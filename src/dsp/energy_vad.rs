use super::{ProcessResult, Processor, time_to_coeff};

/// Simple RMS-based voice activity detector.
///
/// Used in DeepFilter mode (which lacks neural VAD). Computes the RMS energy
/// of each frame and applies smoothing to produce a binary VAD signal.
///
/// The threshold is tuned for post-denoise audio — DeepFilter's output is
/// already clean, so a simple energy check works well.
pub struct EnergyVad {
    threshold: f32,
    smoothed_rms: f32,
    attack_coeff: f32,
    release_coeff: f32,
}

impl EnergyVad {
    /// Create with sensible defaults for post-denoise audio.
    pub fn new() -> Self {
        Self {
            threshold: 0.01,
            smoothed_rms: 0.0,
            attack_coeff: time_to_coeff(2.0),
            release_coeff: time_to_coeff(80.0),
        }
    }
}

impl Processor for EnergyVad {
    fn process(&mut self, samples: &mut [f32]) -> ProcessResult {
        if samples.is_empty() {
            return ProcessResult { vad: Some(0.0) };
        }

        let rms = (samples.iter().map(|s| s * s).sum::<f32>() / samples.len() as f32).sqrt();

        let coeff = if rms > self.smoothed_rms {
            self.attack_coeff
        } else {
            self.release_coeff
        };
        self.smoothed_rms = coeff * self.smoothed_rms + (1.0 - coeff) * rms;

        let vad = if self.smoothed_rms >= self.threshold {
            1.0
        } else {
            0.0
        };

        // Does NOT modify samples — measurement-only processor
        ProcessResult { vad: Some(vad) }
    }

    fn reset(&mut self) {
        self.smoothed_rms = 0.0;
    }
}
