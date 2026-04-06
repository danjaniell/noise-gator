use std::f64::consts::PI;

use super::{ProcessResult, Processor};

/// 2nd-order Butterworth high-pass filter.
///
/// Placed before the denoiser to strip low-frequency rumble (AC hum, desk
/// vibration, traffic) so RNNoise doesn't waste model capacity on it.
pub struct HighPassFilter {
    b0: f64,
    b1: f64,
    b2: f64,
    a1: f64,
    a2: f64,
    // State (Direct Form II Transposed)
    z1: f64,
    z2: f64,
    enabled: bool,
}

impl HighPassFilter {
    /// Create a 2nd-order Butterworth high-pass at the given cutoff frequency.
    pub fn new(cutoff_hz: f64, sample_rate: f64) -> Self {
        let w0 = 2.0 * PI * cutoff_hz / sample_rate;
        let cos_w0 = w0.cos();
        let sin_w0 = w0.sin();
        // Q = 1/√2 for Butterworth (maximally flat passband)
        let alpha = sin_w0 / (2.0 * std::f64::consts::FRAC_1_SQRT_2.recip());

        let a0 = 1.0 + alpha;
        let b0 = f64::midpoint(1.0, cos_w0) / a0;
        let b1 = (-(1.0 + cos_w0)) / a0;
        let b2 = f64::midpoint(1.0, cos_w0) / a0;
        let a1 = (-2.0 * cos_w0) / a0;
        let a2 = (1.0 - alpha) / a0;

        Self {
            b0,
            b1,
            b2,
            a1,
            a2,
            z1: 0.0,
            z2: 0.0,
            enabled: true,
        }
    }

    /// Default: 80 Hz cutoff at 48 kHz sample rate.
    pub fn default_80hz() -> Self {
        Self::new(80.0, 48_000.0)
    }

    #[allow(dead_code)]
    pub fn set_enabled(&mut self, enabled: bool) {
        self.enabled = enabled;
    }
}

impl Processor for HighPassFilter {
    fn process(&mut self, samples: &mut [f32]) -> ProcessResult {
        if !self.enabled {
            return ProcessResult::default();
        }

        for s in samples.iter_mut() {
            let x = *s as f64;
            let y = self.b0 * x + self.z1;
            self.z1 = self.b1 * x - self.a1 * y + self.z2;
            self.z2 = self.b2 * x - self.a2 * y;
            *s = y as f32;
        }

        ProcessResult::default()
    }

    fn reset(&mut self) {
        self.z1 = 0.0;
        self.z2 = 0.0;
    }
}
