//! DeepFilterNet processor — wraps 3 ONNX models via tract-onnx.
//!
//! Feature-gated behind `deepfilter`. When the feature is disabled, this module
//! provides a stub that returns an error on construction.

use super::{ProcessResult, Processor};

#[cfg(feature = "deepfilter")]
mod inner {
    use std::path::Path;
    use std::sync::Arc;

    use anyhow::{anyhow, Result};
    use ndarray::Array4;
    use rustfft::{num_complex::Complex32, FftPlanner};
    use tract_onnx::prelude::*;

    use super::{ProcessResult, Processor};

    /// DeepFilterNet3 constants (from model config.ini).
    const SR: usize = 48_000;
    const FFT_SIZE: usize = 960;
    const HOP_SIZE: usize = 480;
    const FREQ_SIZE: usize = FFT_SIZE / 2 + 1; // 481
    const NB_ERB: usize = 32;
    const NB_DF: usize = 96;
    const DF_ORDER: usize = 5;

    /// ERB band edges for 32 bands at 48kHz with FFT size 960.
    /// Each band spans a range of frequency bins. Precomputed from the
    /// DeepFilterNet erb_fb() function with min_nb_erb_freqs=2.
    static ERB_WIDTHS: [usize; NB_ERB] = [
        2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 3, 3, 3, 4, 4, 5,
        5, 6, 7, 8, 9, 10, 12, 13, 15, 17, 20, 23, 26, 30, 35, 39,
    ];

    type TractModel = SimplePlan<TypedFact, Box<dyn TypedOp>, Graph<TypedFact, Box<dyn TypedOp>>>;

    pub struct DeepFilterProcessor {
        // ONNX models
        encoder: TractModel,
        erb_decoder: TractModel,
        df_decoder: TractModel,
        // FFT
        fft_forward: Arc<dyn rustfft::Fft<f32>>,
        fft_inverse: Arc<dyn rustfft::Fft<f32>>,
        // Windows
        analysis_window: Vec<f32>,
        synthesis_window: Vec<f32>,
        // State buffers
        input_buf: Vec<f32>,          // ring buffer for overlap
        output_buf: Vec<f32>,         // overlap-add output
        fft_scratch: Vec<Complex32>,
        spectrum: Vec<Complex32>,
        // Rolling DF buffer: last DF_ORDER frames of first NB_DF bins
        df_buf: Vec<Vec<Complex32>>,
        // Normalization state
        erb_norm_state: f32,          // running mean for ERB normalization
        spec_norm_state: Vec<f32>,    // running norm per DF bin
        alpha: f32,                   // smoothing coefficient
        // Position tracking
        frame_count: usize,
    }

    // SAFETY: Same rationale as Denoiser — exclusively owned by a single audio
    // callback closure, never shared across threads.
    unsafe impl Send for DeepFilterProcessor {}

    impl DeepFilterProcessor {
        pub fn from_model_dir(model_dir: &Path) -> Result<Self> {
            let enc_path = model_dir.join("enc.onnx");
            let erb_dec_path = model_dir.join("erb_dec.onnx");
            let df_dec_path = model_dir.join("df_dec.onnx");

            if !enc_path.exists() || !erb_dec_path.exists() || !df_dec_path.exists() {
                return Err(anyhow!("Missing ONNX model files in {}", model_dir.display()));
            }

            // Load models with tract — optimize for inference
            let encoder = tract_onnx::onnx()
                .model_for_path(&enc_path)?
                .into_optimized()?
                .into_runnable()?;

            let erb_decoder = tract_onnx::onnx()
                .model_for_path(&erb_dec_path)?
                .into_optimized()?
                .into_runnable()?;

            let df_decoder = tract_onnx::onnx()
                .model_for_path(&df_dec_path)?
                .into_optimized()?
                .into_runnable()?;

            // FFT setup
            let mut planner = FftPlanner::<f32>::new();
            let fft_forward = planner.plan_fft_forward(FFT_SIZE);
            let fft_inverse = planner.plan_fft_inverse(FFT_SIZE);

            // Vorbis window: w(n) = sin(pi/2 * sin^2(pi * n / N))
            let analysis_window: Vec<f32> = (0..FFT_SIZE)
                .map(|n| {
                    let x = std::f32::consts::PI * n as f32 / FFT_SIZE as f32;
                    (std::f32::consts::FRAC_PI_2 * x.sin().powi(2)).sin()
                })
                .collect();

            // Synthesis window (same shape, normalized for overlap-add reconstruction)
            let synthesis_window = analysis_window.clone();

            let alpha = (-(HOP_SIZE as f64) / (SR as f64 * 1.0)).exp() as f32;

            Ok(Self {
                encoder,
                erb_decoder,
                df_decoder,
                fft_forward,
                fft_inverse,
                analysis_window,
                synthesis_window,
                input_buf: vec![0.0; FFT_SIZE],
                output_buf: vec![0.0; FFT_SIZE],
                fft_scratch: vec![Complex32::new(0.0, 0.0); FFT_SIZE],
                spectrum: vec![Complex32::new(0.0, 0.0); FFT_SIZE],
                df_buf: vec![vec![Complex32::new(0.0, 0.0); NB_DF]; DF_ORDER],
                erb_norm_state: 0.0,
                spec_norm_state: vec![0.0; NB_DF],
                alpha,
                frame_count: 0,
            })
        }

        fn process_frame(&mut self, frame: &mut [f32]) {
            // Shift input buffer and insert new samples
            self.input_buf.copy_within(HOP_SIZE.., 0);
            self.input_buf[FFT_SIZE - HOP_SIZE..].copy_from_slice(frame);

            // Apply analysis window and FFT
            let mut fft_buf: Vec<Complex32> = self.input_buf.iter()
                .zip(&self.analysis_window)
                .map(|(&s, &w)| Complex32::new(s * w, 0.0))
                .collect();

            self.fft_forward.process_with_scratch(&mut fft_buf, &mut self.fft_scratch);

            // Store spectrum (only positive frequencies)
            self.spectrum[..FREQ_SIZE].copy_from_slice(&fft_buf[..FREQ_SIZE]);

            // ── Feature extraction ──────────────────────────────────────
            let erb_feat = self.compute_erb_features();
            let spec_feat = self.compute_spec_features();

            // ── Encoder ─────────────────────────────────────────────────
            let enc_result = match self.run_encoder(&erb_feat, &spec_feat) {
                Ok(r) => r,
                Err(e) => {
                    tracing::error!("Encoder inference failed: {e}");
                    frame.copy_from_slice(&self.input_buf[FFT_SIZE - HOP_SIZE..]);
                    return;
                }
            };

            // ── ERB Decoder → gain mask ─────────────────────────────────
            let erb_gains = match self.run_erb_decoder(&enc_result) {
                Ok(g) => g,
                Err(e) => {
                    tracing::error!("ERB decoder failed: {e}");
                    frame.copy_from_slice(&self.input_buf[FFT_SIZE - HOP_SIZE..]);
                    return;
                }
            };

            // Apply ERB gains to spectrum
            self.apply_erb_gains(&erb_gains);

            // ── DF Decoder → deep filtering coefficients ────────────────
            let df_coefs = match self.run_df_decoder(&enc_result) {
                Ok(c) => c,
                Err(e) => {
                    tracing::error!("DF decoder failed: {e}");
                    // Continue with ERB-only enhancement
                    tracing::warn!("Falling back to ERB-only enhancement");
                    self.apply_df_fallback();
                    self.synthesize(frame);
                    return;
                }
            };

            // Apply deep filtering to first NB_DF frequency bins
            self.apply_deep_filtering(&df_coefs);

            // ── Inverse STFT ────────────────────────────────────────────
            self.synthesize(frame);
            self.frame_count += 1;
        }

        fn compute_erb_features(&mut self) -> [f32; NB_ERB] {
            let mut erb = [0.0f32; NB_ERB];
            let mut bin = 0usize;
            for (band, &width) in ERB_WIDTHS.iter().enumerate() {
                let mut power = 0.0f32;
                for b in bin..bin + width {
                    if b < FREQ_SIZE {
                        power += self.spectrum[b].norm_sqr();
                    }
                }
                erb[band] = power / width as f32;
                bin += width;
            }

            // Log + normalization
            for v in &mut erb {
                *v = (*v + 1e-10).log10() * 10.0;
            }

            // Exponential mean normalization
            let mean: f32 = erb.iter().sum::<f32>() / NB_ERB as f32;
            self.erb_norm_state = self.alpha * self.erb_norm_state + (1.0 - self.alpha) * mean;
            for v in &mut erb {
                *v = (*v - self.erb_norm_state) / 40.0;
            }

            erb
        }

        fn compute_spec_features(&mut self) -> Vec<[f32; 2]> {
            let mut features = Vec::with_capacity(NB_DF);
            for i in 0..NB_DF {
                let c = self.spectrum[i];
                let norm_sqr = c.norm_sqr();

                // Update running normalization
                self.spec_norm_state[i] = self.alpha * self.spec_norm_state[i]
                    + (1.0 - self.alpha) * norm_sqr;

                let norm = (self.spec_norm_state[i] + 1e-10).sqrt();
                features.push([c.re / norm, c.im / norm]);
            }
            features
        }

        fn run_encoder(&self, erb: &[f32; NB_ERB], spec: &[[f32; 2]]) -> Result<EncoderOutput> {
            // feat_erb: [1, 1, 1, 32]
            let erb_tensor: Tensor = Array4::from_shape_fn((1, 1, 1, NB_ERB), |(_, _, _, i)| erb[i])
                .into();

            // feat_spec: [1, 2, 1, 96]
            let spec_tensor: Tensor = Array4::from_shape_fn((1, 2, 1, NB_DF), |(_, ch, _, i)| {
                spec[i][ch]
            }).into();

            let result = self.encoder.run(tvec![
                erb_tensor.into(),
                spec_tensor.into(),
            ])?;

            // Outputs: e0, e1, e2, e3, emb, c0, lsnr
            Ok(EncoderOutput {
                e0: result[0].clone(),
                e1: result[1].clone(),
                e2: result[2].clone(),
                e3: result[3].clone(),
                emb: result[4].clone(),
                c0: result[5].clone(),
                _lsnr: result[6].clone(),
            })
        }

        fn run_erb_decoder(&self, enc: &EncoderOutput) -> Result<Vec<f32>> {
            // Inputs: emb, e3, e2, e1, e0
            let result = self.erb_decoder.run(tvec![
                enc.emb.clone(),
                enc.e3.clone(),
                enc.e2.clone(),
                enc.e1.clone(),
                enc.e0.clone(),
            ])?;

            // Output: gain mask [1, 1, 1, 32]
            let gains: Vec<f32> = result[0]
                .to_array_view::<f32>()?
                .iter()
                .copied()
                .collect();
            Ok(gains)
        }

        fn run_df_decoder(&self, enc: &EncoderOutput) -> Result<Vec<[Complex32; DF_ORDER]>> {
            // Inputs: emb, c0
            let result = self.df_decoder.run(tvec![
                enc.emb.clone(),
                enc.c0.clone(),
            ])?;

            // Output: coefficients reshaped to [NB_DF, DF_ORDER, 2] (real/imag)
            let raw: Vec<f32> = result[0]
                .to_array_view::<f32>()?
                .iter()
                .copied()
                .collect();

            let mut coefs = Vec::with_capacity(NB_DF);
            for i in 0..NB_DF {
                let mut c = [Complex32::new(0.0, 0.0); DF_ORDER];
                for j in 0..DF_ORDER {
                    let idx = (i * DF_ORDER + j) * 2;
                    if idx + 1 < raw.len() {
                        c[j] = Complex32::new(raw[idx], raw[idx + 1]);
                    }
                }
                coefs.push(c);
            }
            Ok(coefs)
        }

        fn apply_erb_gains(&mut self, gains: &[f32]) {
            // Interpolate 32 ERB gains back to 481 frequency bins
            let mut bin = 0usize;
            for (band, &width) in ERB_WIDTHS.iter().enumerate() {
                let g = if band < gains.len() {
                    gains[band].clamp(0.0, 1.0)
                } else {
                    1.0
                };
                for b in bin..bin + width {
                    if b < FREQ_SIZE {
                        self.spectrum[b] *= g;
                    }
                }
                bin += width;
            }
        }

        fn apply_deep_filtering(&mut self, coefs: &[[Complex32; DF_ORDER]]) {
            // Shift DF buffer (ring buffer of last DF_ORDER spectra)
            self.df_buf.rotate_left(1);
            let last = self.df_buf.last_mut().unwrap();
            for i in 0..NB_DF {
                last[i] = self.spectrum[i];
            }

            // Apply deep filtering: for each of first NB_DF bins,
            // output = sum(coefs[i][j] * df_buf[j][i]) for j in 0..DF_ORDER
            for i in 0..NB_DF {
                let mut sum = Complex32::new(0.0, 0.0);
                for j in 0..DF_ORDER {
                    let buf_idx = j; // oldest to newest
                    sum += coefs[i][j] * self.df_buf[buf_idx][i];
                }
                self.spectrum[i] = sum;
            }
        }

        fn apply_df_fallback(&self) {
            // No-op: spectrum already has ERB gains applied
        }

        fn synthesize(&mut self, frame: &mut [f32]) {
            // Reconstruct full spectrum (conjugate symmetry)
            let mut ifft_buf = vec![Complex32::new(0.0, 0.0); FFT_SIZE];
            ifft_buf[..FREQ_SIZE].copy_from_slice(&self.spectrum[..FREQ_SIZE]);
            for i in 1..FFT_SIZE / 2 {
                ifft_buf[FFT_SIZE - i] = self.spectrum[i].conj();
            }

            // Inverse FFT
            self.fft_inverse.process_with_scratch(&mut ifft_buf, &mut self.fft_scratch);

            // Normalize + apply synthesis window + overlap-add
            let norm = 1.0 / FFT_SIZE as f32;
            for i in 0..FFT_SIZE {
                let sample = ifft_buf[i].re * norm * self.synthesis_window[i];
                self.output_buf[i] += sample;
            }

            // Output the first HOP_SIZE samples
            frame.copy_from_slice(&self.output_buf[..HOP_SIZE]);

            // Shift overlap buffer
            self.output_buf.copy_within(HOP_SIZE.., 0);
            self.output_buf[FFT_SIZE - HOP_SIZE..].fill(0.0);
        }
    }

    struct EncoderOutput {
        e0: TValue,
        e1: TValue,
        e2: TValue,
        e3: TValue,
        emb: TValue,
        c0: TValue,
        _lsnr: TValue,
    }

    impl Processor for DeepFilterProcessor {
        fn process(&mut self, samples: &mut [f32]) -> ProcessResult {
            for chunk in samples.chunks_exact_mut(HOP_SIZE) {
                self.process_frame(chunk);
            }
            ProcessResult { vad: None }
        }

        fn reset(&mut self) {
            self.input_buf.fill(0.0);
            self.output_buf.fill(0.0);
            self.erb_norm_state = 0.0;
            self.spec_norm_state.fill(0.0);
            for buf in &mut self.df_buf {
                buf.fill(Complex32::new(0.0, 0.0));
            }
            self.frame_count = 0;
        }
    }
}

#[cfg(feature = "deepfilter")]
pub use inner::DeepFilterProcessor;

// ── Stub when deepfilter feature is disabled ────────────────────────────

#[cfg(not(feature = "deepfilter"))]
pub struct DeepFilterProcessor;

#[cfg(not(feature = "deepfilter"))]
impl DeepFilterProcessor {
    pub fn from_model_dir(_path: &std::path::Path) -> anyhow::Result<Self> {
        anyhow::bail!("DeepFilterNet support not compiled in. Rebuild with --features deepfilter")
    }
}

#[cfg(not(feature = "deepfilter"))]
impl Processor for DeepFilterProcessor {
    fn process(&mut self, _samples: &mut [f32]) -> ProcessResult {
        ProcessResult::default()
    }
    fn reset(&mut self) {}
}
