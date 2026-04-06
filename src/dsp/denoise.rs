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
