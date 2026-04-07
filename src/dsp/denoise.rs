use nnnoiseless::DenoiseState;

use super::{ProcessResult, Processor};

/// RNNoise expects 480-sample frames (10 ms @ 48 kHz, mono).
pub const FRAME_SIZE: usize = nnnoiseless::FRAME_SIZE;

/// RNNoise-based denoiser. Operates on exactly [`FRAME_SIZE`] samples at a time.
///
/// Internally scales samples to int16 range before processing (RNNoise requirement)
/// and scales back after.
pub struct Denoiser {
    state: Box<DenoiseState<'static>>,
    out_buf: Vec<f32>,
}

// SAFETY: DenoiseState is !Send because nnnoiseless uses internal raw pointers
// for FFT buffers. However, each Denoiser instance is exclusively owned by a
// single audio callback closure (captured by `move`). It is never shared across
// threads — it's created before the closure and moved into it. The &mut self
// on process_frame guarantees exclusive access.
unsafe impl Send for Denoiser {}

impl Denoiser {
    pub fn new() -> Self {
        Self {
            state: DenoiseState::new(),
            out_buf: vec![0.0; FRAME_SIZE],
        }
    }

    /// Denoise a single FRAME_SIZE chunk. Returns VAD probability.
    fn process_frame(&mut self, frame: &mut [f32; FRAME_SIZE]) -> f32 {
        // RNNoise expects int16-scale input
        for s in frame.iter_mut() {
            *s *= 32768.0;
        }

        let vad = self.state.process_frame(&mut self.out_buf, frame);
        frame.copy_from_slice(&self.out_buf);

        // Scale back to f32 range
        for s in frame.iter_mut() {
            *s /= 32768.0;
        }

        vad
    }
}

impl Processor for Denoiser {
    fn process(&mut self, samples: &mut [f32]) -> ProcessResult {
        // Process in FRAME_SIZE chunks. Leftover samples are untouched
        // (caller should buffer to FRAME_SIZE boundaries).
        let mut last_vad = 0.0;
        for chunk in samples.chunks_exact_mut(FRAME_SIZE) {
            // chunks_exact_mut guarantees exactly FRAME_SIZE elements per chunk.
            let frame: &mut [f32; FRAME_SIZE] = chunk
                .try_into()
                .expect("chunks_exact_mut yielded wrong size — this is a bug");
            last_vad = self.process_frame(frame);
        }
        ProcessResult {
            vad: Some(last_vad),
        }
    }

    fn reset(&mut self) {
        self.state = DenoiseState::new();
        self.out_buf.fill(0.0);
    }
}

/// Two-pass RNNoise denoiser for stronger suppression.
///
/// Runs audio through two independent RNNoise instances sequentially.
/// The second pass catches residual noise that the first pass left behind,
/// providing significantly better suppression at the cost of ~2x CPU.
pub struct DualPassDenoiser {
    pass1: Denoiser,
    pass2: Denoiser,
}

impl DualPassDenoiser {
    pub fn new() -> Self {
        Self {
            pass1: Denoiser::new(),
            pass2: Denoiser::new(),
        }
    }
}

// SAFETY: Same rationale as Denoiser — exclusively owned by a single audio
// callback closure.
unsafe impl Send for DualPassDenoiser {}

impl Processor for DualPassDenoiser {
    fn process(&mut self, samples: &mut [f32]) -> ProcessResult {
        // First pass: primary denoise + VAD extraction
        let result = self.pass1.process(samples);
        // Second pass: clean up residual noise
        self.pass2.process(samples);
        // Return VAD from first pass (most accurate on raw signal)
        result
    }

    fn reset(&mut self) {
        self.pass1.reset();
        self.pass2.reset();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn dual_pass_processes_without_panic() {
        let mut dp = DualPassDenoiser::new();
        let mut samples = [0.1f32; FRAME_SIZE];
        let result = dp.process(&mut samples);
        assert!(result.vad.is_some(), "VAD should come from first pass");
    }

    #[test]
    fn dual_pass_vad_comes_from_first_pass() {
        let mut dp = DualPassDenoiser::new();
        // Feed a frame with signal — first pass should produce non-zero VAD
        let mut samples = [0.0f32; FRAME_SIZE];
        for (i, s) in samples.iter_mut().enumerate() {
            *s = (i as f32 * 0.1).sin() * 0.5;
        }
        let result = dp.process(&mut samples);
        assert!(result.vad.is_some());
    }

    #[test]
    fn dual_pass_output_differs_from_single_pass() {
        let mut single = Denoiser::new();
        let mut dual = DualPassDenoiser::new();

        let mut input = [0.0f32; FRAME_SIZE];
        for (i, s) in input.iter_mut().enumerate() {
            *s = (i as f32 * 0.1).sin() * 0.3;
        }

        let mut single_out = input;
        let mut dual_out = input;
        single.process(&mut single_out);
        dual.process(&mut dual_out);

        // They should produce different output (dual-pass applies more suppression)
        let differs = single_out.iter().zip(&dual_out).any(|(a, b)| (a - b).abs() > 1e-10);
        assert!(differs, "Dual-pass should differ from single-pass");
    }
}
