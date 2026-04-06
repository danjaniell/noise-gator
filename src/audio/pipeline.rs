use std::sync::{
    atomic::{AtomicBool, AtomicU32, AtomicU64, Ordering},
    Arc, Mutex,
};

use anyhow::Result;
use cpal::traits::{DeviceTrait, StreamTrait};
use cpal::{Stream, StreamConfig};
use ringbuf::traits::{Consumer, Observer, Producer, Split};

use crate::config::RuntimeSettings;
use crate::dsp::autogain::AutoGain;
use crate::dsp::denoise::{Denoiser, FRAME_SIZE};
use crate::dsp::eq::ThreeBandEq;
use crate::dsp::gate::NoiseGate;
use crate::dsp::highpass::HighPassFilter;
use crate::dsp::Processor;

use super::device;
use super::resample::{resample_linear_into, StreamResampler};

/// Monotonically increasing pipeline ID — used to invalidate stale audio callbacks.
static PIPELINE_ID: AtomicU64 = AtomicU64::new(0);

/// Current RMS audio level (f32 bits stored in AtomicU32).
pub static AUDIO_LEVEL: AtomicU32 = AtomicU32::new(0);

/// Thread-safe pipeline handle. Dropping it stops the audio streams.
struct PipelineInner {
    _input_stream: Stream,
    _monitor_stream: Option<Stream>,
    _virtual_stream: Option<Stream>,
}

// SAFETY: cpal::Stream is !Send on some platforms (e.g., macOS CoreAudio uses
// thread-local handles). We wrap PipelineInner in a Mutex<Option<...>> so all
// access is serialized. The streams are never moved between threads — they're
// created on one thread, stored in the Mutex, and dropped from the Mutex.
// No thread-local data escapes the Mutex boundary.
unsafe impl Send for PipelineInner {}

/// Stored device configuration for auto-reconnect.
#[derive(Clone, Default)]
struct DeviceConfig {
    input: Option<String>,
    monitor: Option<String>,
    virtual_dev: Option<String>,
}

/// The running audio pipeline. Owns all cpal streams.
pub struct Pipeline {
    inner: Mutex<Option<PipelineInner>>,
    /// Shared settings that the tray can modify while audio is running.
    pub settings: Arc<RuntimeSettings>,
    /// Set by error callbacks — watchdog thread checks this.
    error_flag: Arc<AtomicBool>,
    /// Last-used device config for auto-restart.
    device_config: Mutex<DeviceConfig>,
}

/// Lock a Mutex, recovering from poison (another thread panicked while holding it).
/// In an audio app, we'd rather work with potentially stale data than crash.
fn lock_or_recover<T>(mutex: &Mutex<T>) -> std::sync::MutexGuard<'_, T> {
    mutex.lock().unwrap_or_else(|poisoned| {
        tracing::warn!("Mutex was poisoned — recovering");
        poisoned.into_inner()
    })
}

impl Pipeline {
    pub fn new(settings: Arc<RuntimeSettings>) -> Self {
        Self {
            inner: Mutex::new(None),
            settings,
            error_flag: Arc::new(AtomicBool::new(false)),
            device_config: Mutex::new(DeviceConfig::default()),
        }
    }

    /// Start (or restart) the audio pipeline.
    pub fn start(
        &self,
        input_name: Option<&str>,
        monitor_name: Option<&str>,
        virtual_name: Option<&str>,
    ) -> Result<()> {
        // Store device config for auto-reconnect
        *lock_or_recover(&self.device_config) = DeviceConfig {
            input: input_name.map(String::from),
            monitor: monitor_name.map(String::from),
            virtual_dev: virtual_name.map(String::from),
        };

        // Stop existing pipeline first
        self.stop();
        self.error_flag.store(false, Ordering::Relaxed);

        let my_id = PIPELINE_ID.fetch_add(1, Ordering::SeqCst) + 1;

        // ── Input device ────────────────────────────────────────────────
        let input_device = device::find_input(input_name)?;
        let in_supported = device::best_f32_config(&input_device, true)?;
        let in_cfg: StreamConfig = in_supported.clone().into();
        let in_channels = in_cfg.channels as usize;
        let in_rate = in_cfg.sample_rate as f64;

        tracing::info!(
            "Input: {} | {} ch | {} Hz | {:?}",
            input_device.description().map(|d| d.name().to_string()).unwrap_or_default(),
            in_channels,
            in_cfg.sample_rate,
            in_supported.sample_format()
        );

        let rnnoise_rate = 48_000f64;
        let settings = Arc::clone(&self.settings);
        let error_flag = Arc::clone(&self.error_flag);

        // ── Build output helper ─────────────────────────────────────────
        struct OutputInfo {
            stream: Stream,
            prod: ringbuf::HeapProd<f32>,
            rate: f64,
        }

        let build_output =
            |device_name: &str, label: &str, pipeline_id: u64| -> Result<OutputInfo> {
                let out_device = device::find_output(Some(device_name))?;
                let supported = out_device
                    .default_output_config()
                    .or_else(|_| device::best_f32_config(&out_device, false))?;
                let cfg: StreamConfig = supported.into();
                let channels = cfg.channels as usize;
                let rate = cfg.sample_rate as f64;

                tracing::info!("{}: {} | {} ch | {} Hz", label, device_name, channels, rate);

                // 3 frames (~30ms at 48kHz) — tight enough for low latency,
                // enough headroom for scheduling jitter.
                let rb = ringbuf::HeapRb::<f32>::new(FRAME_SIZE * 3);
                let (prod, mut cons) = rb.split();

                let is_monitor = label == "Monitor";
                let settings_clone = Arc::clone(&settings);
                let mut temp = vec![0f32; 8192];

                let stream = out_device.build_output_stream(
                    &cfg,
                    move |data: &mut [f32], _| {
                        if PIPELINE_ID.load(Ordering::SeqCst) != pipeline_id {
                            data.fill(0.0);
                            return;
                        }

                        let gain = if is_monitor {
                            settings_clone.output_gain.load(Ordering::Relaxed)
                        } else {
                            1.0f32.to_bits()
                        };
                        let gain = f32::from_bits(gain);
                        let frames = data.len() / channels;

                        // Anti-latency: if ring buffer has more than 2 frames
                        // of backlog beyond what we need right now, skip ahead
                        // to stay near real-time (~20ms max tolerable latency).
                        // Uses temp buffer to drain — no allocation in hot path.
                        let buffered = cons.occupied_len();
                        let max_buffered = FRAME_SIZE * 2;
                        if buffered > max_buffered + frames {
                            let mut remaining = buffered - max_buffered;
                            while remaining > 0 {
                                let drain = remaining.min(temp.len());
                                cons.pop_slice(&mut temp[..drain]);
                                remaining -= drain;
                            }
                        }

                        if frames > temp.len() {
                            temp.resize(frames, 0.0);
                        }
                        let read = cons.pop_slice(&mut temp[..frames]);
                        for (i, ch_frame) in data.chunks_mut(channels).enumerate() {
                            let s = if i < read { temp[i] * gain } else { 0.0 };
                            ch_frame.fill(s);
                        }
                    },
                    {
                        let error_flag = Arc::clone(&error_flag);
                        move |err| {
                            tracing::error!("Output error: {err}");
                            error_flag.store(true, Ordering::Relaxed);
                        }
                    },
                    None,
                )?;
                stream.play()?;
                Ok(OutputInfo { stream, prod, rate })
            };

        // ── Monitor output (optional) ───────────────────────────────────
        let monitor = monitor_name
            .map(|id| build_output(id, "Monitor", my_id))
            .transpose()?;

        // ── Virtual device output ───────────────────────────────────────
        let virt_name = virtual_name
            .map(String::from)
            .or_else(device::detect_virtual_device);

        let virtual_out = virt_name
            .as_deref()
            .filter(|v| monitor_name != Some(*v))
            .map(|id| build_output(id, "Virtual", my_id))
            .transpose()?;

        if virtual_out.is_none() {
            tracing::warn!(
                "No virtual audio device found. Other apps won't receive denoised audio."
            );
        }

        // ── Unpack ──────────────────────────────────────────────────────
        let (monitor_stream, mut mon_prod) = match monitor {
            Some(m) => (Some(m.stream), Some((m.rate, m.prod))),
            None => (None, None),
        };
        let (virtual_stream, mut virt_prod) = match virtual_out {
            Some(v) => (Some(v.stream), Some((v.rate, v.prod))),
            None => (None, None),
        };

        // ── Input callback (DSP chain) ──────────────────────────────────
        let mut highpass = HighPassFilter::default_80hz();
        let mut denoiser = Denoiser::new();
        let eq_settings = self.settings.load_eq_settings();
        let mut eq = ThreeBandEq::new(rnnoise_rate, eq_settings);
        let gate_settings = self.settings.load_gate_settings();
        let mut gate = NoiseGate::with_settings(
            self.settings.hard_mode.load(Ordering::Relaxed),
            gate_settings,
        );
        let autogain_settings = self.settings.load_autogain_settings();
        let mut autogain = AutoGain::new(autogain_settings);

        let max_accum = FRAME_SIZE * 4;
        let mut accumulator: Vec<f32> = Vec::with_capacity(max_accum);
        let mut resample_buf: Vec<f32> = Vec::with_capacity(FRAME_SIZE * 2);

        // Pre-allocated mono downmix buffer — avoids heap allocation in audio callback.
        // Sized for worst case: 4096 stereo frames = 4096 mono samples.
        let mut mono_buf: Vec<f32> = vec![0.0; 8192];

        // Sinc resampler for input (only created if device isn't already 48kHz)
        let needs_resample = (in_rate - rnnoise_rate).abs() >= 1.0;
        let mut input_resampler = if needs_resample {
            Some(StreamResampler::new(
                in_rate as usize,
                rnnoise_rate as usize,
                FRAME_SIZE,
            ))
        } else {
            None
        };

        let input_settings = Arc::clone(&self.settings);

        let input_stream = input_device.build_input_stream(
            &in_cfg,
            move |data: &[f32], _| {
                if PIPELINE_ID.load(Ordering::SeqCst) != my_id {
                    return;
                }

                let i_gain = f32::from_bits(input_settings.input_gain.load(Ordering::Relaxed));
                let denoise_on = input_settings.denoise_enabled.load(Ordering::Relaxed);

                // 1. Downmix to mono + input gain (zero-alloc: reuses pre-allocated buffer)
                let mono_len = data.len() / in_channels;
                if mono_buf.len() < mono_len {
                    mono_buf.resize(mono_len, 0.0);
                }
                for (i, chunk) in data.chunks(in_channels).enumerate() {
                    mono_buf[i] = (chunk.iter().sum::<f32>() / in_channels as f32) * i_gain;
                }
                let mono = &mono_buf[..mono_len];

                // 2. Resample to 48 kHz via sinc resampler (if needed).
                // When no resample needed, use mono slice directly — zero allocation.
                let resampled;
                let at_48k: &[f32] = match input_resampler {
                    Some(ref mut resampler) => {
                        resampled = resampler.process(mono);
                        &resampled
                    }
                    None => mono,
                };

                // 3. RMS level for tray indicator
                if !at_48k.is_empty() {
                    let rms = (at_48k.iter().map(|s| s * s).sum::<f32>() / at_48k.len() as f32)
                        .sqrt();
                    AUDIO_LEVEL.store(rms.to_bits(), Ordering::Relaxed);
                }

                // 4. Accumulate for FRAME_SIZE processing
                accumulator.extend_from_slice(at_48k);
                if accumulator.len() > max_accum {
                    let excess = accumulator.len() - max_accum;
                    accumulator.drain(..excess);
                }

                // 5. Process complete frames
                // Pick up live setting changes
                gate.set_enabled(input_settings.hard_mode.load(Ordering::Relaxed));
                gate.update_settings(input_settings.load_gate_settings());
                eq.update_settings(input_settings.load_eq_settings());
                autogain.update_settings(input_settings.load_autogain_settings());

                let mut read_pos = 0;
                while read_pos + FRAME_SIZE <= accumulator.len() {
                    let mut frame = [0f32; FRAME_SIZE];
                    frame.copy_from_slice(&accumulator[read_pos..read_pos + FRAME_SIZE]);
                    read_pos += FRAME_SIZE;

                    // Pre-filter: remove low-freq rumble before denoise
                    highpass.process(&mut frame);

                    if denoise_on {
                        let result = denoiser.process(&mut frame);
                        if let Some(vad) = result.vad {
                            gate.set_vad(vad);
                        }
                        gate.process(&mut frame);
                        eq.process(&mut frame);
                        autogain.process(&mut frame);
                    }

                    // Push to virtual device
                    if let Some((vrate, ref mut prod)) = virt_prod {
                        if (rnnoise_rate - vrate).abs() < 1.0 {
                            let _ = prod.push_slice(&frame);
                        } else {
                            resample_buf.clear();
                            resample_linear_into(&frame, rnnoise_rate, vrate, &mut resample_buf);
                            let _ = prod.push_slice(&resample_buf);
                        }
                    }

                    // Push to monitor
                    if let Some((mrate, ref mut prod)) = mon_prod {
                        if (rnnoise_rate - mrate).abs() < 1.0 {
                            let _ = prod.push_slice(&frame);
                        } else {
                            resample_buf.clear();
                            resample_linear_into(&frame, rnnoise_rate, mrate, &mut resample_buf);
                            let _ = prod.push_slice(&resample_buf);
                        }
                    }
                }

                if read_pos > 0 {
                    accumulator.drain(..read_pos);
                }
            },
            {
                let error_flag = Arc::clone(&error_flag);
                move |err| {
                    tracing::error!("Input error: {err}");
                    error_flag.store(true, Ordering::Relaxed);
                }
            },
            None,
        )?;

        input_stream.play()?;

        *lock_or_recover(&self.inner) = Some(PipelineInner {
            _input_stream: input_stream,
            _monitor_stream: monitor_stream,
            _virtual_stream: virtual_stream,
        });

        tracing::info!("Pipeline started");
        Ok(())
    }

    pub fn stop(&self) {
        PIPELINE_ID.store(0, Ordering::SeqCst);
        let prev = lock_or_recover(&self.inner).take();
        if prev.is_some() {
            drop(prev);
            std::thread::sleep(std::time::Duration::from_millis(50));
            AUDIO_LEVEL.store(0f32.to_bits(), Ordering::Relaxed);
            tracing::info!("Pipeline stopped");
        }
    }

    pub fn is_running(&self) -> bool {
        lock_or_recover(&self.inner).is_some()
    }

    /// Check if a device error has occurred.
    pub fn has_error(&self) -> bool {
        self.error_flag.load(Ordering::Relaxed)
    }

    /// Attempt to restart the pipeline with the last-used device config.
    /// Falls back to system defaults if the original devices aren't available.
    pub fn try_reconnect(&self) -> Result<()> {
        let config = lock_or_recover(&self.device_config).clone();

        // Try with original devices first
        let result = self.start(
            config.input.as_deref(),
            config.monitor.as_deref(),
            config.virtual_dev.as_deref(),
        );

        if result.is_ok() {
            tracing::info!("Reconnected with original devices");
            return Ok(());
        }

        // Fall back to defaults
        tracing::warn!("Original devices unavailable, falling back to defaults");
        self.start(None, None, None)
    }

    /// Spawn a background watchdog thread that auto-restarts the pipeline
    /// if a device error is detected. Returns a handle to stop the watchdog.
    pub fn start_watchdog(pipeline: Arc<Pipeline>) -> WatchdogHandle {
        let running = Arc::new(AtomicBool::new(true));
        let running_clone = Arc::clone(&running);

        let handle = std::thread::Builder::new()
            .name("audio-watchdog".into())
            .spawn(move || {
                while running_clone.load(Ordering::Relaxed) {
                    std::thread::sleep(std::time::Duration::from_secs(2));

                    if !running_clone.load(Ordering::Relaxed) {
                        break;
                    }

                    if pipeline.has_error() && pipeline.is_running() {
                        tracing::warn!("Device error detected — attempting reconnect...");
                        pipeline.stop();

                        // Wait a bit for the device to settle (e.g., USB re-enumeration)
                        std::thread::sleep(std::time::Duration::from_secs(1));

                        match pipeline.try_reconnect() {
                            Ok(()) => tracing::info!("Auto-reconnect successful"),
                            Err(e) => {
                                tracing::error!("Auto-reconnect failed: {e}");
                                // Will retry on next watchdog cycle
                            }
                        }
                    }
                }
            })
            .expect("Failed to spawn watchdog thread");

        WatchdogHandle {
            running,
            _handle: Some(handle),
        }
    }
}

/// Handle to the watchdog thread. Stops the thread on drop.
pub struct WatchdogHandle {
    running: Arc<AtomicBool>,
    _handle: Option<std::thread::JoinHandle<()>>,
}

impl WatchdogHandle {
    #[allow(dead_code)]
    pub fn stop(&self) {
        self.running.store(false, Ordering::Relaxed);
    }
}

impl Drop for WatchdogHandle {
    fn drop(&mut self) {
        self.running.store(false, Ordering::Relaxed);
        if let Some(handle) = self._handle.take() {
            let _ = handle.join();
        }
    }
}
