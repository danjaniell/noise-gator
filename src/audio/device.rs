use anyhow::{Result, anyhow};
use cpal::traits::{DeviceTrait, HostTrait};
use cpal::{Device, Host, SampleFormat, SupportedStreamConfig};

/// Known virtual audio device name patterns (checked in priority order).
const VIRTUAL_DEVICE_PATTERNS: &[&str] = &[
    "NoiseGator",
    "BlackHole",
    "VB-Cable",
    "VB-Audio",
    "CABLE Input",
    "Soundflower",
    "Loopback",
];

/// Serializable audio device descriptor.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct AudioDevice {
    pub name: String,
    pub is_default: bool,
}

impl AudioDevice {
    /// Formatted name for UI display (appends "(default)" when applicable).
    pub fn display_name(&self) -> String {
        if self.is_default {
            format!("{} (default)", self.name)
        } else {
            self.name.clone()
        }
    }
}

/// Enumerate input (microphone) devices. Excludes known virtual devices to
/// prevent feedback loops.
pub fn list_input_devices() -> Result<Vec<AudioDevice>> {
    let host = cpal::default_host();
    let default_name = host
        .default_input_device()
        .and_then(|d| device_name(&d))
        .unwrap_or_default();

    Ok(host
        .input_devices()?
        .filter_map(|d| {
            let name = device_name(&d)?;
            if is_virtual_device(&name) {
                return None;
            }
            d.default_input_config().ok()?;
            Some(AudioDevice {
                is_default: name == default_name,
                name,
            })
        })
        .collect())
}

/// Enumerate output devices.
pub fn list_output_devices() -> Result<Vec<AudioDevice>> {
    let host = cpal::default_host();
    let default_name = host
        .default_output_device()
        .and_then(|d| device_name(&d))
        .unwrap_or_default();

    Ok(host
        .output_devices()?
        .filter_map(|d| {
            let name = device_name(&d)?;
            d.default_output_config().ok()?;
            Some(AudioDevice {
                is_default: name == default_name,
                name,
            })
        })
        .collect())
}

/// Auto-detect an installed virtual audio device.
pub fn detect_virtual_device() -> Option<String> {
    let host = cpal::default_host();
    host.output_devices().ok()?.find_map(|d| {
        let name = device_name(&d)?;
        if is_virtual_device(&name) {
            Some(name)
        } else {
            None
        }
    })
}

/// Get device name via the cpal 0.17 description API.
fn device_name(d: &Device) -> Option<String> {
    d.description().ok().map(|desc| desc.name().to_string())
}

fn is_virtual_device(name: &str) -> bool {
    VIRTUAL_DEVICE_PATTERNS
        .iter()
        .any(|pattern| name.contains(pattern))
}

/// Find an input device by name, or return the system default.
pub fn find_input(name: Option<&str>) -> Result<Device> {
    let host = cpal::default_host();
    find_device(&host, name, true)
}

/// Find an output device by name, or return the system default.
pub fn find_output(name: Option<&str>) -> Result<Device> {
    let host = cpal::default_host();
    find_device(&host, name, false)
}

fn find_device(host: &Host, name: Option<&str>, is_input: bool) -> Result<Device> {
    match name {
        None => if is_input {
            host.default_input_device()
        } else {
            host.default_output_device()
        }
        .ok_or_else(|| {
            anyhow!(
                "No default {} device",
                if is_input { "input" } else { "output" }
            )
        }),
        Some(id) => {
            let devices: Vec<_> = if is_input {
                host.input_devices()?.collect()
            } else {
                host.output_devices()?.collect()
            };
            let kind = if is_input { "Input" } else { "Output" };

            // Exact match first
            if let Some(d) = devices
                .iter()
                .find(|d| device_name(d).map(|n| n == id).unwrap_or(false))
            {
                return Ok(d.clone());
            }

            // Substring fallback: prefer longest matching name to avoid greedy short matches.
            // For input devices, skip virtual devices to prevent feedback loops.
            let id_lower = id.to_lowercase();
            let mut best: Option<(&Device, String)> = None;
            for d in &devices {
                if let Some(n) = device_name(d) {
                    if is_input && is_virtual_device(&n) {
                        continue;
                    }
                    let n_lower = n.to_lowercase();
                    if id_lower.contains(&n_lower) || n_lower.contains(&id_lower) {
                        let is_better =
                            best.as_ref().map_or(true, |(_, prev)| n.len() > prev.len());
                        if is_better {
                            best = Some((d, n));
                        }
                    }
                }
            }
            if let Some((d, matched_name)) = best {
                tracing::warn!("{kind} device '{id}' not found exactly, matched '{matched_name}'");
                return Ok(d.clone());
            }

            Err(anyhow!("{kind} device '{id}' not found"))
        }
    }
}

/// Pick the best audio config for a device, preferring F32 then I16, targeting 48 kHz
/// and low channel count. Returns the actual supported format — caller must handle
/// sample conversion if the format is not F32.
pub fn best_f32_config(device: &Device, is_input: bool) -> Result<SupportedStreamConfig> {
    let configs: Vec<_> = if is_input {
        device.supported_input_configs()?.collect()
    } else {
        device.supported_output_configs()?.collect()
    };

    // Score a config: prefer F32 > I16 > others, then 48kHz, then low channel count.
    let score = |c: &cpal::SupportedStreamConfigRange| -> i64 {
        let fmt_score: i64 = match c.sample_format() {
            SampleFormat::F32 => 0,
            SampleFormat::I16 => 1_000_000,
            _ => 2_000_000,
        };
        let rate_diff = (c.max_sample_rate() as i64 - 48_000).abs();
        let ch_score = if c.channels() <= 2 {
            0
        } else {
            c.channels() as i64 * 1000
        };
        fmt_score + rate_diff + ch_score
    };

    let best = configs
        .iter()
        .filter(|c| matches!(c.sample_format(), SampleFormat::F32 | SampleFormat::I16))
        .min_by_key(|c| score(c))
        .map(|c| {
            let rate = c.max_sample_rate().min(48_000).max(c.min_sample_rate());
            (*c).with_sample_rate(rate)
        });

    if let Some(cfg) = best {
        return Ok(cfg);
    }

    // Fallback to device default
    if is_input {
        device.default_input_config().map_err(|e| anyhow!(e))
    } else {
        device.default_output_config().map_err(|e| anyhow!(e))
    }
}
