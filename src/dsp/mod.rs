pub mod autogain;
pub mod deepfilter;
pub mod denoise;
pub mod energy_vad;
pub mod eq;
pub mod gate;
pub mod highpass;

/// Trait for any DSP processor that operates on audio frames in-place.
pub trait Processor: Send {
    /// Process a buffer of f32 samples in-place. Returns optional metadata.
    fn process(&mut self, samples: &mut [f32]) -> ProcessResult;

    /// Reset internal state (e.g., filter history, smoothing).
    #[allow(dead_code)]
    fn reset(&mut self);
}

/// Metadata returned from a processing step.
#[derive(Debug, Clone, Default)]
pub struct ProcessResult {
    /// Voice activity probability (0.0–1.0), if the processor computes it.
    pub vad: Option<f32>,
}

