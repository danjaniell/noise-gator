use super::{ProcessResult, Processor};

/// Automatic gain normalization applied after denoise.
///
/// Measures the RMS level of each frame and applies makeup gain to bring
/// the signal up to a target RMS. Uses slow-attack / fast-release smoothing
/// to avoid pumping artifacts while still catching level drops quickly.
///
/// This prevents the "quiet voice after denoise" problem where RNNoise
/// strips energy along with noise.
pub struct AutoGain {
    enabled: bool,
    settings: AutoGainSettings,
    /// Smoothed RMS measurement (slow-moving for stable gain).
    smoothed_rms: f32,
    /// Current applied gain (smoothed to avoid clicks).
    current_gain: f32,
}

/// Auto-gain parameters.
#[derive(Debug, Clone, Copy, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct AutoGainSettings {
    /// Target RMS level (0.0–1.0). Default: 0.1 (~-20 dBFS).
    pub target_rms: f32,
    /// Maximum gain applied. Prevents amplifying silence/noise. Default: 10.0 (20dB).
    pub max_gain: f32,
    /// Minimum gain (never attenuate below this). Default: 1.0 (unity).
    pub min_gain: f32,
    /// RMS smoothing coefficient (0.0–1.0). Higher = slower response.
    pub rms_smoothing: f32,
    /// Gain smoothing coefficient. Higher = slower gain changes.
    pub gain_smoothing: f32,
    /// RMS level below which we don't apply gain (silence threshold).
    /// Prevents amplifying noise floor. Default: 0.001.
    pub silence_threshold: f32,
    pub enabled: bool,
}

impl Default for AutoGainSettings {
    fn default() -> Self {
        Self {
            target_rms: 0.1,
            max_gain: 4.0,
            min_gain: 1.0,
            rms_smoothing: 0.95,
            gain_smoothing: 0.99,
            silence_threshold: 0.005,
            enabled: true,
        }
    }
}

impl AutoGain {
    pub fn new(settings: AutoGainSettings) -> Self {
        Self {
            enabled: settings.enabled,
            settings,
            smoothed_rms: 0.0,
            current_gain: 1.0,
        }
    }

    pub fn update_settings(&mut self, new: AutoGainSettings) {
        self.enabled = new.enabled;
        self.settings = new;
    }

}

impl Processor for AutoGain {
    fn process(&mut self, samples: &mut [f32]) -> ProcessResult {
        if !self.enabled || samples.is_empty() {
            return ProcessResult::default();
        }

        // Measure frame RMS
        let rms = (samples.iter().map(|s| s * s).sum::<f32>() / samples.len() as f32).sqrt();

        // Smooth the RMS measurement
        self.smoothed_rms = self.settings.rms_smoothing * self.smoothed_rms
            + (1.0 - self.settings.rms_smoothing) * rms;

        // Calculate target gain
        let target_gain = if self.smoothed_rms < self.settings.silence_threshold {
            // Below silence threshold — don't amplify noise floor
            1.0
        } else {
            (self.settings.target_rms / self.smoothed_rms)
                .clamp(self.settings.min_gain, self.settings.max_gain)
        };

        // Smooth gain changes (asymmetric: fast attack for gain reduction, slow for increase)
        let coeff = if target_gain < self.current_gain {
            // Reducing gain — respond faster to avoid clipping
            1.0 - (1.0 - self.settings.gain_smoothing) * 4.0
        } else {
            self.settings.gain_smoothing
        }
        .clamp(0.0, 0.999);

        self.current_gain = coeff * self.current_gain + (1.0 - coeff) * target_gain;

        // Apply gain
        for s in samples.iter_mut() {
            *s *= self.current_gain;
            // Soft clip to prevent any overshoot
            *s = soft_clip(*s);
        }

        ProcessResult::default()
    }

    fn reset(&mut self) {
        self.smoothed_rms = 0.0;
        self.current_gain = 1.0;
    }
}

/// Soft clipper (tanh-style) — prevents hard clipping artifacts.
#[inline]
fn soft_clip(x: f32) -> f32 {
    if x.abs() < 0.9 {
        x
    } else {
        x.signum() * (1.0 - (-x.abs() * 2.0).exp())
    }
}
