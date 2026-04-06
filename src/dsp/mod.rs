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
    fn reset(&mut self);
}

/// Metadata returned from a processing step.
#[derive(Debug, Clone, Default)]
pub struct ProcessResult {
    /// Voice activity probability (0.0–1.0), if the processor computes it.
    pub vad: Option<f32>,
}

/// A chain of processors applied sequentially.
#[allow(dead_code)]
pub struct ProcessorChain {
    stages: Vec<Box<dyn Processor>>,
}

#[allow(dead_code)]
impl ProcessorChain {
    pub fn new() -> Self {
        Self { stages: Vec::new() }
    }

    pub fn add(mut self, processor: Box<dyn Processor>) -> Self {
        self.stages.push(processor);
        self
    }

    /// Process samples through all stages. Returns the last non-None VAD value.
    pub fn process(&mut self, samples: &mut [f32]) -> ProcessResult {
        let mut last_vad = None;
        for stage in &mut self.stages {
            let result = stage.process(samples);
            if result.vad.is_some() {
                last_vad = result.vad;
            }
        }
        ProcessResult { vad: last_vad }
    }

    pub fn reset(&mut self) {
        for stage in &mut self.stages {
            stage.reset();
        }
    }
}
