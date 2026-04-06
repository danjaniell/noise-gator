use std::f64::consts::PI;

use super::{ProcessResult, Processor};

/// Biquad filter coefficients + state (Direct Form II Transposed).
#[derive(Clone)]
struct Biquad {
    b0: f64,
    b1: f64,
    b2: f64,
    a1: f64,
    a2: f64,
    z1: f64,
    z2: f64,
}

impl Biquad {
    fn low_shelf(freq: f64, gain_db: f64, sample_rate: f64) -> Self {
        let a = 10f64.powf(gain_db / 40.0);
        let w0 = 2.0 * PI * freq / sample_rate;
        let (cos_w0, sin_w0) = (w0.cos(), w0.sin());
        let alpha = sin_w0 / 2.0 * ((a + 1.0 / a) * (1.0 / 0.9 - 1.0) + 2.0).sqrt();
        let a0 = (a + 1.0) + (a - 1.0) * cos_w0 + 2.0 * alpha * a.sqrt();

        Self {
            b0: a * ((a + 1.0) - (a - 1.0) * cos_w0 + 2.0 * alpha * a.sqrt()) / a0,
            b1: 2.0 * a * ((a - 1.0) - (a + 1.0) * cos_w0) / a0,
            b2: a * ((a + 1.0) - (a - 1.0) * cos_w0 - 2.0 * alpha * a.sqrt()) / a0,
            a1: -2.0 * ((a - 1.0) + (a + 1.0) * cos_w0) / a0,
            a2: ((a + 1.0) + (a - 1.0) * cos_w0 - 2.0 * alpha * a.sqrt()) / a0,
            z1: 0.0,
            z2: 0.0,
        }
    }

    fn high_shelf(freq: f64, gain_db: f64, sample_rate: f64) -> Self {
        let a = 10f64.powf(gain_db / 40.0);
        let w0 = 2.0 * PI * freq / sample_rate;
        let (cos_w0, sin_w0) = (w0.cos(), w0.sin());
        let alpha = sin_w0 / 2.0 * ((a + 1.0 / a) * (1.0 / 0.9 - 1.0) + 2.0).sqrt();
        let a0 = (a + 1.0) - (a - 1.0) * cos_w0 + 2.0 * alpha * a.sqrt();

        Self {
            b0: a * ((a + 1.0) + (a - 1.0) * cos_w0 + 2.0 * alpha * a.sqrt()) / a0,
            b1: -2.0 * a * ((a - 1.0) + (a + 1.0) * cos_w0) / a0,
            b2: a * ((a + 1.0) + (a - 1.0) * cos_w0 - 2.0 * alpha * a.sqrt()) / a0,
            a1: 2.0 * ((a - 1.0) - (a + 1.0) * cos_w0) / a0,
            a2: ((a + 1.0) - (a - 1.0) * cos_w0 - 2.0 * alpha * a.sqrt()) / a0,
            z1: 0.0,
            z2: 0.0,
        }
    }

    fn peaking(freq: f64, gain_db: f64, q: f64, sample_rate: f64) -> Self {
        let a = 10f64.powf(gain_db / 40.0);
        let w0 = 2.0 * PI * freq / sample_rate;
        let (cos_w0, sin_w0) = (w0.cos(), w0.sin());
        let alpha = sin_w0 / (2.0 * q);
        let a0 = 1.0 + alpha / a;

        Self {
            b0: (1.0 + alpha * a) / a0,
            b1: (-2.0 * cos_w0) / a0,
            b2: (1.0 - alpha * a) / a0,
            a1: (-2.0 * cos_w0) / a0,
            a2: (1.0 - alpha / a) / a0,
            z1: 0.0,
            z2: 0.0,
        }
    }

    #[inline]
    fn process_sample(&mut self, x: f64) -> f64 {
        let y = self.b0 * x + self.z1;
        self.z1 = self.b1 * x - self.a1 * y + self.z2;
        self.z2 = self.b2 * x - self.a2 * y;
        y
    }

    #[allow(dead_code)]
    fn reset(&mut self) {
        self.z1 = 0.0;
        self.z2 = 0.0;
    }
}

/// 3-band parametric EQ: bass (low shelf), mid (peaking), treble (high shelf).
pub struct ThreeBandEq {
    sample_rate: f64,
    bass: Biquad,
    mid: Biquad,
    treble: Biquad,
    settings: EqSettings,
}

/// EQ band gains in dB. Range: -12.0 to +12.0.
#[derive(Debug, Clone, Copy, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct EqSettings {
    pub bass_db: f32,
    pub mid_db: f32,
    pub treble_db: f32,
    pub enabled: bool,
}

impl Default for EqSettings {
    fn default() -> Self {
        Self {
            bass_db: 3.0,
            mid_db: 1.5,
            treble_db: -2.5,
            enabled: true,
        }
    }
}

impl ThreeBandEq {
    pub fn new(sample_rate: f64, settings: EqSettings) -> Self {
        Self {
            sample_rate,
            bass: Biquad::low_shelf(300.0, settings.bass_db as f64, sample_rate),
            mid: Biquad::peaking(2500.0, settings.mid_db as f64, 1.2, sample_rate),
            treble: Biquad::high_shelf(6000.0, settings.treble_db as f64, sample_rate),
            settings,
        }
    }

    /// Update EQ gains. Rebuilds only the filters whose values changed.
    pub fn update_settings(&mut self, new: EqSettings) {
        if (new.bass_db - self.settings.bass_db).abs() > f32::EPSILON {
            self.bass = Biquad::low_shelf(300.0, new.bass_db as f64, self.sample_rate);
        }
        if (new.mid_db - self.settings.mid_db).abs() > f32::EPSILON {
            self.mid = Biquad::peaking(2500.0, new.mid_db as f64, 1.2, self.sample_rate);
        }
        if (new.treble_db - self.settings.treble_db).abs() > f32::EPSILON {
            self.treble = Biquad::high_shelf(6000.0, new.treble_db as f64, self.sample_rate);
        }
        self.settings = new;
    }

    #[allow(dead_code)]
    pub fn settings(&self) -> &EqSettings {
        &self.settings
    }
}

impl Processor for ThreeBandEq {
    fn process(&mut self, samples: &mut [f32]) -> ProcessResult {
        if !self.settings.enabled {
            return ProcessResult::default();
        }

        // All bands at 0 dB — skip
        let s = &self.settings;
        if s.bass_db.abs() < f32::EPSILON
            && s.mid_db.abs() < f32::EPSILON
            && s.treble_db.abs() < f32::EPSILON
        {
            return ProcessResult::default();
        }

        for sample in samples.iter_mut() {
            let mut x = *sample as f64;
            if self.settings.bass_db.abs() > f32::EPSILON {
                x = self.bass.process_sample(x);
            }
            if self.settings.mid_db.abs() > f32::EPSILON {
                x = self.mid.process_sample(x);
            }
            if self.settings.treble_db.abs() > f32::EPSILON {
                x = self.treble.process_sample(x);
            }
            *sample = x as f32;
        }

        ProcessResult::default()
    }

    fn reset(&mut self) {
        self.bass.reset();
        self.mid.reset();
        self.treble.reset();
    }
}
