use audioadapter_buffers::direct::InterleavedSlice;
use rubato::{Fft, FixedSync, Resampler};

/// Wraps rubato 2.0's FFT-based resampler for streaming use.
///
/// `rubato` operates on fixed-size chunks. This wrapper handles buffering
/// input samples until a full chunk is available, then resamples in bulk.
pub struct StreamResampler {
    resampler: Fft<f32>,
    input_buf: Vec<f32>,
    output_buf: Vec<f32>,
    chunk_size: usize,
}

impl StreamResampler {
    /// Create a resampler from `from_rate` to `to_rate`.
    /// `chunk_size` controls the internal processing block size — smaller = lower
    /// latency but slightly more overhead. 480 (one RNNoise frame) is a good default.
    pub fn new(from_rate: usize, to_rate: usize, chunk_size: usize) -> Self {
        // 1 channel mono, 1 sub-chunk, fixed input size
        let resampler = Fft::new(from_rate, to_rate, chunk_size, 1, 1, FixedSync::Input)
            .expect("Failed to create resampler");
        let max_output = resampler.output_frames_max();
        Self {
            resampler,
            input_buf: Vec::with_capacity(chunk_size * 2),
            output_buf: vec![0.0f32; max_output],
            chunk_size,
        }
    }

    /// Feed input samples and get resampled output.
    /// May return empty if not enough input has accumulated for a full chunk.
    pub fn process(&mut self, input: &[f32]) -> Vec<f32> {
        self.input_buf.extend_from_slice(input);

        let mut output = Vec::new();

        while self.input_buf.len() >= self.chunk_size {
            let chunk: Vec<f32> = self.input_buf.drain(..self.chunk_size).collect();

            // Ensure output buffer is large enough
            let needed = self.resampler.output_frames_next();
            if self.output_buf.len() < needed {
                self.output_buf.resize(needed, 0.0);
            }

            // rubato 2.0: InterleavedSlice adapters for mono (1 channel)
            let input_adapter = InterleavedSlice::new(&chunk, 1, self.chunk_size).unwrap();
            let mut output_adapter =
                InterleavedSlice::new_mut(&mut self.output_buf, 1, needed).unwrap();

            match self
                .resampler
                .process_into_buffer(&input_adapter, &mut output_adapter, None)
            {
                Ok((_in_frames, out_frames)) => {
                    output.extend_from_slice(&self.output_buf[..out_frames]);
                }
                Err(e) => {
                    tracing::warn!("Resampler error: {e}");
                }
            }
        }

        output
    }

    /// Reset internal state (call when restarting pipeline).
    #[allow(dead_code)]
    pub fn reset(&mut self) {
        self.resampler.reset();
        self.input_buf.clear();
    }
}

/// Simple linear interpolation resampler for output-side resampling where
/// latency matters less and chunk alignment is awkward.
pub fn resample_linear_into(input: &[f32], from_rate: f64, to_rate: f64, out: &mut Vec<f32>) {
    if input.is_empty() {
        return;
    }
    let ratio = from_rate / to_rate;
    let out_len = (input.len() as f64 / ratio).ceil() as usize;
    out.reserve(out_len);
    for i in 0..out_len {
        let src = i as f64 * ratio;
        let lo = src.floor() as usize;
        let hi = (lo + 1).min(input.len() - 1);
        let t = (src - lo as f64) as f32;
        out.push(input[lo] * (1.0 - t) + input[hi] * t);
    }
}
