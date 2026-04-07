//! DeepFilterNet processor — wraps 3 ONNX models via ort (ONNX Runtime).
//!
//! Feature-gated behind `deepfilter`. When the feature is disabled, this module
//! provides a stub that returns an error on construction.

use super::{ProcessResult, Processor};

#[cfg(feature = "deepfilter")]
mod inner {
    use std::path::Path;
    use std::sync::Arc;

    use anyhow::{Result, anyhow};
    use ndarray::Array4;
    use ort::session::Session;
    use ort::value::TensorRef;
    use rustfft::{FftPlanner, num_complex::Complex32};

    /// Helper to extract a named output as an owned Vec<f32> with shape info.
    fn extract_output(
        outputs: &ort::session::SessionOutputs<'_>,
        name: &str,
    ) -> Result<(Vec<usize>, Vec<f32>)> {
        let value = outputs
            .get(name)
            .ok_or_else(|| anyhow!("Model missing expected output '{name}'"))?;
        let (shape, data) = value.try_extract_tensor::<f32>()?;
        let shape_usize: Vec<usize> = shape.iter().map(|&d| d as usize).collect();
        Ok((shape_usize, data.to_vec()))
    }

    use super::{ProcessResult, Processor};

    /// DeepFilterNet3 constants (from model config.ini).
    const SR: usize = 48_000;
    const FFT_SIZE: usize = 960;
    const HOP_SIZE: usize = 480;
    const FREQ_SIZE: usize = FFT_SIZE / 2 + 1; // 481
    const NB_ERB: usize = 32;
    const NB_DF: usize = 96;
    const DF_ORDER: usize = 5;

    /// Maximum consecutive inference errors before disabling DeepFilter.
    const MAX_ERRORS: usize = 10;

    /// ERB band edges for 32 bands at 48kHz with FFT size 960.
    static ERB_WIDTHS: [usize; NB_ERB] = [
        2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 3, 3, 3, 4, 4, 5, 5, 6, 7, 8, 9, 10, 12, 13, 15, 17, 20,
        23, 26, 30, 35, 39,
    ];

    pub struct DeepFilterProcessor {
        // ONNX Runtime sessions
        encoder: Session,
        erb_decoder: Session,
        df_decoder: Session,
        // FFT
        fft_forward: Arc<dyn rustfft::Fft<f32>>,
        fft_inverse: Arc<dyn rustfft::Fft<f32>>,
        // Windows
        analysis_window: Vec<f32>,
        synthesis_window: Vec<f32>,
        // State buffers
        input_buf: Vec<f32>,
        output_buf: Vec<f32>,
        fft_buf: Vec<Complex32>,
        ifft_buf: Vec<Complex32>,
        fft_scratch: Vec<Complex32>,
        spectrum: Vec<Complex32>,
        // Rolling DF buffer: last DF_ORDER frames of first NB_DF bins
        df_buf: Vec<Vec<Complex32>>,
        // Normalization state
        erb_norm_state: f32,
        spec_norm_state: Vec<f32>,
        alpha: f32,
        // Position tracking
        frame_count: usize,
        // Error tracking
        error_count: usize,
    }

    // SAFETY: Exclusively owned by a single audio callback closure.
    unsafe impl Send for DeepFilterProcessor {}

    /// Encoder output tensors as owned (shape, data) pairs for passing to decoders.
    struct EncoderOutput {
        e0: (Vec<usize>, Vec<f32>),
        e1: (Vec<usize>, Vec<f32>),
        e2: (Vec<usize>, Vec<f32>),
        e3: (Vec<usize>, Vec<f32>),
        emb: (Vec<usize>, Vec<f32>),
        c0: (Vec<usize>, Vec<f32>),
    }

    impl EncoderOutput {
        /// Create a TensorRef from one of the stored outputs.
        fn tensor_ref(field: &(Vec<usize>, Vec<f32>)) -> Result<TensorRef<'_, f32>> {
            Ok(TensorRef::from_array_view(
                ndarray::ArrayViewD::from_shape(field.0.as_slice(), field.1.as_slice())
                    .map_err(|e| anyhow!("Shape mismatch: {e}"))?,
            )?)
        }
    }

    impl DeepFilterProcessor {
        pub fn from_model_dir(model_dir: &Path) -> Result<Self> {
            let enc_path = model_dir.join("enc.onnx");
            let erb_dec_path = model_dir.join("erb_dec.onnx");
            let df_dec_path = model_dir.join("df_dec.onnx");

            if !enc_path.exists() || !erb_dec_path.exists() || !df_dec_path.exists() {
                return Err(anyhow!(
                    "Missing ONNX model files in {}",
                    model_dir.display()
                ));
            }

            use ort::session::builder::GraphOptimizationLevel;

            let encoder = Session::builder()?
                .with_optimization_level(GraphOptimizationLevel::Level3)?
                .with_intra_threads(1)?
                .commit_from_file(&enc_path)?;

            let erb_decoder = Session::builder()?
                .with_optimization_level(GraphOptimizationLevel::Level3)?
                .with_intra_threads(1)?
                .commit_from_file(&erb_dec_path)?;

            let df_decoder = Session::builder()?
                .with_optimization_level(GraphOptimizationLevel::Level3)?
                .with_intra_threads(1)?
                .commit_from_file(&df_dec_path)?;

            // Log model info
            tracing::info!(
                "DeepFilter encoder: {} inputs, {} outputs",
                encoder.inputs.len(),
                encoder.outputs.len()
            );

            // FFT setup
            let mut planner = FftPlanner::<f32>::new();
            let fft_forward = planner.plan_fft_forward(FFT_SIZE);
            let fft_inverse = planner.plan_fft_inverse(FFT_SIZE);

            // sqrt-Hann window — matches DeepFilterNet training STFT.
            // analysis * synthesis must satisfy COLA for perfect reconstruction.
            let analysis_window: Vec<f32> = (0..FFT_SIZE)
                .map(|n| {
                    let hann = 0.5 * (1.0 - (2.0 * std::f32::consts::PI * n as f32 / FFT_SIZE as f32).cos());
                    hann.sqrt()
                })
                .collect();
            let synthesis_window = analysis_window.clone();

            let alpha = (-(HOP_SIZE as f64) / (SR as f64 * 1.0)).exp() as f32;

            Ok(Self {
                encoder,
                erb_decoder,
                df_decoder,
                fft_forward,
                fft_inverse,
                analysis_window,
                synthesis_window,
                input_buf: vec![0.0; FFT_SIZE],
                output_buf: vec![0.0; FFT_SIZE],
                fft_buf: vec![Complex32::new(0.0, 0.0); FFT_SIZE],
                ifft_buf: vec![Complex32::new(0.0, 0.0); FFT_SIZE],
                fft_scratch: vec![Complex32::new(0.0, 0.0); FFT_SIZE],
                spectrum: vec![Complex32::new(0.0, 0.0); FFT_SIZE],
                df_buf: vec![vec![Complex32::new(0.0, 0.0); NB_DF]; DF_ORDER],
                erb_norm_state: 0.0,
                spec_norm_state: vec![0.0; NB_DF],
                alpha,
                frame_count: 0,
                error_count: 0,
            })
        }

        pub fn is_failed(&self) -> bool {
            self.error_count >= MAX_ERRORS
        }

        fn process_frame(&mut self, frame: &mut [f32]) {
            self.input_buf.copy_within(HOP_SIZE.., 0);
            self.input_buf[FFT_SIZE - HOP_SIZE..].copy_from_slice(frame);

            for (buf, (&s, &w)) in self
                .fft_buf
                .iter_mut()
                .zip(self.input_buf.iter().zip(&self.analysis_window))
            {
                *buf = Complex32::new(s * w, 0.0);
            }

            self.fft_forward
                .process_with_scratch(&mut self.fft_buf, &mut self.fft_scratch);

            self.spectrum[..FREQ_SIZE].copy_from_slice(&self.fft_buf[..FREQ_SIZE]);

            let erb_feat = self.compute_erb_features();
            let spec_feat = self.compute_spec_features();

            let enc_result = match self.run_encoder(&erb_feat, &spec_feat) {
                Ok(r) => r,
                Err(e) => {
                    tracing::error!("Encoder inference failed: {e}");
                    frame.copy_from_slice(&self.input_buf[FFT_SIZE - HOP_SIZE..]);
                    self.error_count += 1;
                    return;
                }
            };

            let erb_gains = match self.run_erb_decoder(&enc_result) {
                Ok(g) => g,
                Err(e) => {
                    tracing::error!("ERB decoder failed: {e}");
                    frame.copy_from_slice(&self.input_buf[FFT_SIZE - HOP_SIZE..]);
                    self.error_count += 1;
                    return;
                }
            };

            // Save original noisy spectrum for DF buffer BEFORE ERB gains modify it.
            // Deep filtering uses learned filters on the raw noisy signal, not
            // the ERB-enhanced one.
            self.df_buf.rotate_left(1);
            let last = self.df_buf.last_mut().unwrap();
            last[..NB_DF].copy_from_slice(&self.spectrum[..NB_DF]);

            self.apply_erb_gains(&erb_gains);

            let df_coefs = match self.run_df_decoder(&enc_result) {
                Ok(c) => c,
                Err(e) => {
                    tracing::error!("DF decoder failed: {e}");
                    tracing::warn!("Falling back to ERB-only enhancement");
                    self.synthesize(frame);
                    return;
                }
            };

            self.apply_deep_filtering(&df_coefs);
            self.synthesize(frame);
            self.frame_count += 1;
            self.error_count = 0;
        }

        fn compute_erb_features(&mut self) -> [f32; NB_ERB] {
            let mut erb = [0.0f32; NB_ERB];
            let mut bin = 0usize;
            for (band, &width) in ERB_WIDTHS.iter().enumerate() {
                let mut power = 0.0f32;
                for b in bin..bin + width {
                    if b < FREQ_SIZE {
                        power += self.spectrum[b].norm_sqr();
                    }
                }
                erb[band] = power / width as f32;
                bin += width;
            }

            for v in &mut erb {
                *v = (*v + 1e-10).log10() * 10.0;
            }

            let mean: f32 = erb.iter().sum::<f32>() / NB_ERB as f32;
            self.erb_norm_state = self.alpha * self.erb_norm_state + (1.0 - self.alpha) * mean;
            for v in &mut erb {
                *v = (*v - self.erb_norm_state) / 40.0;
            }

            erb
        }

        fn compute_spec_features(&mut self) -> [[f32; 2]; NB_DF] {
            let mut features = [[0.0f32; 2]; NB_DF];
            for i in 0..NB_DF {
                let c = self.spectrum[i];
                let norm_sqr = c.norm_sqr();
                self.spec_norm_state[i] =
                    self.alpha * self.spec_norm_state[i] + (1.0 - self.alpha) * norm_sqr;
                let norm = (self.spec_norm_state[i] + 1e-10).sqrt();
                features[i] = [c.re / norm, c.im / norm];
            }
            features
        }

        fn run_encoder(
            &mut self,
            erb: &[f32; NB_ERB],
            spec: &[[f32; 2]; NB_DF],
        ) -> Result<EncoderOutput> {
            let erb_array = Array4::from_shape_fn((1, 1, 1, NB_ERB), |(_, _, _, i)| erb[i]);
            let spec_array =
                Array4::from_shape_fn((1, 2, 1, NB_DF), |(_, ch, _, i)| spec[i][ch]);

            let erb_ref = TensorRef::from_array_view(erb_array.view())?;
            let spec_ref = TensorRef::from_array_view(spec_array.view())?;

            let outputs = self.encoder.run(ort::inputs![
                "feat_erb" => erb_ref,
                "feat_spec" => spec_ref
            ])?;

            Ok(EncoderOutput {
                e0: extract_output(&outputs, "e0")?,
                e1: extract_output(&outputs, "e1")?,
                e2: extract_output(&outputs, "e2")?,
                e3: extract_output(&outputs, "e3")?,
                emb: extract_output(&outputs, "emb")?,
                c0: extract_output(&outputs, "c0")?,
            })
        }

        fn run_erb_decoder(&mut self, enc: &EncoderOutput) -> Result<Vec<f32>> {
            let outputs = self.erb_decoder.run(ort::inputs![
                "emb" => EncoderOutput::tensor_ref(&enc.emb)?,
                "e3" => EncoderOutput::tensor_ref(&enc.e3)?,
                "e2" => EncoderOutput::tensor_ref(&enc.e2)?,
                "e1" => EncoderOutput::tensor_ref(&enc.e1)?,
                "e0" => EncoderOutput::tensor_ref(&enc.e0)?
            ])?;

            let value = outputs
                .get("m")
                .ok_or_else(|| anyhow!("ERB decoder missing output 'm'"))?;
            let (_, gains) = value.try_extract_tensor::<f32>()?;
            Ok(gains.to_vec())
        }

        fn run_df_decoder(&mut self, enc: &EncoderOutput) -> Result<Vec<[Complex32; DF_ORDER]>> {
            let outputs = self.df_decoder.run(ort::inputs![
                "emb" => EncoderOutput::tensor_ref(&enc.emb)?,
                "c0" => EncoderOutput::tensor_ref(&enc.c0)?
            ])?;

            let value = outputs
                .get("coefs")
                .ok_or_else(|| anyhow!("DF decoder missing output 'coefs'"))?;
            let (_, raw) = value.try_extract_tensor::<f32>()?;

            let mut coefs = Vec::with_capacity(NB_DF);
            for i in 0..NB_DF {
                let mut c = [Complex32::new(0.0, 0.0); DF_ORDER];
                for j in 0..DF_ORDER {
                    let idx = (i * DF_ORDER + j) * 2;
                    if idx + 1 < raw.len() {
                        c[j] = Complex32::new(raw[idx], raw[idx + 1]);
                    }
                }
                coefs.push(c);
            }
            Ok(coefs)
        }

        fn apply_erb_gains(&mut self, gains: &[f32]) {
            let mut bin = 0usize;
            for (band, &width) in ERB_WIDTHS.iter().enumerate() {
                let g = if band < gains.len() {
                    gains[band].clamp(0.0, 1.0)
                } else {
                    1.0
                };
                for b in bin..bin + width {
                    if b < FREQ_SIZE {
                        self.spectrum[b] *= g;
                    }
                }
                bin += width;
            }
        }

        fn apply_deep_filtering(&mut self, coefs: &[[Complex32; DF_ORDER]]) {
            // df_buf is already populated (before ERB gains) with the noisy spectrum.
            // coefs[bin][0] applies to the newest frame (last in df_buf),
            // coefs[bin][DF_ORDER-1] applies to the oldest.
            let last_idx = DF_ORDER - 1;
            for i in 0..NB_DF {
                let mut sum = Complex32::new(0.0, 0.0);
                for j in 0..DF_ORDER {
                    sum += coefs[i][j] * self.df_buf[last_idx - j][i];
                }
                self.spectrum[i] = sum;
            }
        }

        fn synthesize(&mut self, frame: &mut [f32]) {
            self.ifft_buf.fill(Complex32::new(0.0, 0.0));
            self.ifft_buf[..FREQ_SIZE].copy_from_slice(&self.spectrum[..FREQ_SIZE]);
            for i in 1..FFT_SIZE / 2 {
                self.ifft_buf[FFT_SIZE - i] = self.spectrum[i].conj();
            }

            self.fft_inverse
                .process_with_scratch(&mut self.ifft_buf, &mut self.fft_scratch);

            let norm = 1.0 / FFT_SIZE as f32;
            for i in 0..FFT_SIZE {
                let sample = self.ifft_buf[i].re * norm * self.synthesis_window[i];
                self.output_buf[i] += sample;
            }

            frame.copy_from_slice(&self.output_buf[..HOP_SIZE]);
            self.output_buf.copy_within(HOP_SIZE.., 0);
            self.output_buf[FFT_SIZE - HOP_SIZE..].fill(0.0);
        }
    }

    impl Processor for DeepFilterProcessor {
        fn process(&mut self, samples: &mut [f32]) -> ProcessResult {
            if self.is_failed() {
                return ProcessResult { vad: None };
            }

            for chunk in samples.chunks_exact_mut(HOP_SIZE) {
                self.process_frame(chunk);

                if self.is_failed() {
                    tracing::error!(
                        "DeepFilter inference failed {} times — disabling. \
                         Audio will pass through unprocessed.",
                        self.error_count
                    );
                    return ProcessResult { vad: None };
                }
            }
            ProcessResult { vad: None }
        }

        fn reset(&mut self) {
            self.input_buf.fill(0.0);
            self.output_buf.fill(0.0);
            self.fft_buf.fill(Complex32::new(0.0, 0.0));
            self.ifft_buf.fill(Complex32::new(0.0, 0.0));
            self.erb_norm_state = 0.0;
            self.spec_norm_state.fill(0.0);
            for buf in &mut self.df_buf {
                buf.fill(Complex32::new(0.0, 0.0));
            }
            self.frame_count = 0;
            self.error_count = 0;
        }
    }
}

#[cfg(feature = "deepfilter")]
pub use inner::DeepFilterProcessor;

// ── Stub when deepfilter feature is disabled ────────────────────────────

#[cfg(not(feature = "deepfilter"))]
pub struct DeepFilterProcessor;

#[cfg(not(feature = "deepfilter"))]
impl DeepFilterProcessor {
    pub fn from_model_dir(_path: &std::path::Path) -> anyhow::Result<Self> {
        anyhow::bail!("DeepFilterNet support not compiled in. Rebuild with --features deepfilter")
    }

    pub fn is_failed(&self) -> bool {
        true
    }
}

#[cfg(not(feature = "deepfilter"))]
impl Processor for DeepFilterProcessor {
    fn process(&mut self, _samples: &mut [f32]) -> ProcessResult {
        ProcessResult::default()
    }
    fn reset(&mut self) {}
}
