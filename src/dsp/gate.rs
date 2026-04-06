use super::{ProcessResult, Processor};

/// Configurable VAD-based noise gate with proper attack/release/hold envelope.
///
/// Unlike a simple threshold gate, this implements:
/// - **Attack**: how fast the gate opens when voice is detected (ms)
/// - **Release**: how slowly the gate closes after voice stops (ms)
/// - **Hold**: minimum time the gate stays open after voice detection (ms)
/// - **Threshold**: VAD probability below which the gate engages
/// - **Floor**: gain applied when gated (not zero — preserves naturalness)
///
/// Must be placed after the denoiser in the processing chain.
pub struct NoiseGate {
    enabled: bool,
    settings: GateSettings,
    /// Current gain envelope (0.0–1.0).
    envelope: f32,
    /// Samples remaining in hold phase.
    hold_counter: usize,
    /// Cached VAD from the denoiser.
    last_vad: f32,
    /// Pre-computed coefficients from settings.
    attack_coeff: f32,
    release_coeff: f32,
    hold_samples: usize,
}

/// Noise gate parameters. All times in milliseconds.
#[derive(Debug, Clone, Copy, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct GateSettings {
    /// VAD probability threshold (0.0–1.0). Below this = gated.
    pub threshold: f32,
    /// Attack time in ms — how fast the gate opens.
    pub attack_ms: f32,
    /// Release time in ms — how slowly the gate closes.
    pub release_ms: f32,
    /// Hold time in ms — gate stays open at least this long after voice.
    pub hold_ms: f32,
    /// Floor gain when gated (0.0 = full silence, 0.05 = natural).
    pub floor: f32,
}

impl Default for GateSettings {
    fn default() -> Self {
        Self {
            threshold: 0.20,
            attack_ms: 5.0,
            release_ms: 150.0,
            hold_ms: 100.0,
            floor: 0.05,
        }
    }
}

const SAMPLE_RATE: f32 = 48_000.0;

/// Convert a time constant in ms to a one-pole smoothing coefficient.
/// coeff = exp(-1 / (time_ms * sample_rate / 1000))
fn time_to_coeff(ms: f32) -> f32 {
    if ms <= 0.0 {
        return 0.0;
    }
    (-1.0 / (ms * SAMPLE_RATE / 1000.0)).exp()
}

impl NoiseGate {
    #[allow(dead_code)]
    pub fn new(enabled: bool) -> Self {
        let settings = GateSettings::default();
        Self::with_settings(enabled, settings)
    }

    pub fn with_settings(enabled: bool, settings: GateSettings) -> Self {
        let attack_coeff = time_to_coeff(settings.attack_ms);
        let release_coeff = time_to_coeff(settings.release_ms);
        let hold_samples = (settings.hold_ms * SAMPLE_RATE / 1000.0) as usize;

        Self {
            enabled,
            settings,
            envelope: 1.0,
            hold_counter: 0,
            last_vad: 1.0,
            attack_coeff,
            release_coeff,
            hold_samples,
        }
    }

    pub fn set_enabled(&mut self, enabled: bool) {
        self.enabled = enabled;
        if !enabled {
            self.envelope = 1.0;
            self.hold_counter = 0;
        }
    }

    #[allow(dead_code)]
    pub fn is_enabled(&self) -> bool {
        self.enabled
    }

    /// Update gate settings at runtime (e.g., from tray menu).
    pub fn update_settings(&mut self, new: GateSettings) {
        if new != self.settings {
            self.attack_coeff = time_to_coeff(new.attack_ms);
            self.release_coeff = time_to_coeff(new.release_ms);
            self.hold_samples = (new.hold_ms * SAMPLE_RATE / 1000.0) as usize;
            self.settings = new;
        }
    }

    #[allow(dead_code)]
    pub fn settings(&self) -> &GateSettings {
        &self.settings
    }

    /// Feed VAD probability from the denoiser. Call before [`process`].
    pub fn set_vad(&mut self, vad: f32) {
        self.last_vad = vad;
    }
}

impl Processor for NoiseGate {
    fn process(&mut self, samples: &mut [f32]) -> ProcessResult {
        if !self.enabled {
            return ProcessResult::default();
        }

        let voice_detected = self.last_vad >= self.settings.threshold;

        // Update hold counter
        if voice_detected {
            self.hold_counter = self.hold_samples;
        }

        let in_hold = self.hold_counter > 0;

        for s in samples.iter_mut() {
            // Target gain: 1.0 if voice or in hold phase, floor otherwise
            let target = if voice_detected || in_hold {
                1.0
            } else {
                self.settings.floor
            };

            // Apply attack or release coefficient
            let coeff = if target > self.envelope {
                self.attack_coeff // opening — fast
            } else {
                self.release_coeff // closing — slow
            };

            // One-pole smoother: envelope = coeff * envelope + (1 - coeff) * target
            self.envelope = coeff * self.envelope + (1.0 - coeff) * target;

            *s *= self.envelope;

            // Decrement hold counter per-sample for accurate timing
            if self.hold_counter > 0 {
                self.hold_counter -= 1;
            }
        }

        ProcessResult::default()
    }

    fn reset(&mut self) {
        self.envelope = 1.0;
        self.hold_counter = 0;
        self.last_vad = 1.0;
    }
}
