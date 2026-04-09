use std::sync::Arc;
use std::sync::atomic::Ordering;

use crate::audio::device;
use crate::audio::pipeline::{AUDIO_LEVEL, OUTPUT_LEVEL, Pipeline};
use crate::config::{DenoiseEngine, RuntimeSettings};
use crate::dsp::eq::EqSettings;

// ── Constants ────────────────────────────────────────────────────────────────

const GATE_THRESHOLD_MAX: f32 = 0.50;
const GATE_THRESHOLD_RANGE: f32 = 0.45;
const GATE_FLOOR_ACTIVE: f32 = 0.05;
const GATE_FLOOR_BYPASS: f32 = 1.0;

const METER_ATTACK_COEFF: f32 = 0.3;
const METER_RELEASE_COEFF: f32 = 0.05;
const METER_YELLOW_DB: f32 = -12.0;
const METER_RED_DB: f32 = -6.0;

// ── UI-only state (not shared with audio thread) ────────────────────────────

struct UiState {
    suppression_pct: f32,
    gate_sensitivity: f32,
    clarity_pct: f32,
    advanced_eq: bool,
    bass_db: f32,
    mid_db: f32,
    treble_db: f32,
    eq_defaults: EqSettings,
    input_meter: f32,
    output_meter: f32,
    input_devices: Vec<(String, String)>,
    output_devices: Vec<(String, String)>,
    selected_input: String,
    selected_output: String,
}

impl UiState {
    fn from_settings(
        settings: &RuntimeSettings,
        input_device: Option<&str>,
        output_device: Option<&str>,
    ) -> Self {
        let suppression = f32::from_bits(settings.suppression_level.load(Ordering::Relaxed));
        let threshold = f32::from_bits(settings.gate_threshold.load(Ordering::Relaxed));
        let eq_bass = settings.eq_bass_db10.load(Ordering::Relaxed) as f32 / 10.0;
        let eq_mid = settings.eq_mid_db10.load(Ordering::Relaxed) as f32 / 10.0;
        let eq_treble = settings.eq_treble_db10.load(Ordering::Relaxed) as f32 / 10.0;

        let gate_sensitivity =
            ((GATE_THRESHOLD_MAX - threshold) / GATE_THRESHOLD_RANGE * 100.0).clamp(0.0, 100.0);
        let eq_defaults = EqSettings::default();
        // Guard against divide-by-zero if bass default is ever changed to 0
        let clarity_pct = if eq_defaults.bass_db.abs() > f32::EPSILON {
            (eq_bass / eq_defaults.bass_db * 100.0).clamp(0.0, 100.0)
        } else {
            100.0
        };

        Self {
            suppression_pct: suppression * 100.0,
            gate_sensitivity,
            clarity_pct,
            advanced_eq: settings.advanced_eq.load(Ordering::Relaxed),
            bass_db: eq_bass,
            mid_db: eq_mid,
            treble_db: eq_treble,
            eq_defaults,
            input_meter: 0.0,
            output_meter: 0.0,
            input_devices: Vec::new(),
            output_devices: Vec::new(),
            selected_input: input_device.unwrap_or_default().to_string(),
            selected_output: output_device.unwrap_or_default().to_string(),
        }
    }
}

// ── Settings application ────────────────────────────────────────────────────

/// Full settings window rendered via eframe/egui.
pub struct SettingsApp {
    settings: Arc<RuntimeSettings>,
    pipeline: Arc<Pipeline>,
    ui: UiState,
    devices_loaded: bool,
}

impl SettingsApp {
    pub fn new(
        settings: Arc<RuntimeSettings>,
        pipeline: Arc<Pipeline>,
        input_device: Option<&str>,
        output_device: Option<&str>,
    ) -> Self {
        let ui = UiState::from_settings(&settings, input_device, output_device);
        Self {
            settings,
            pipeline,
            ui,
            devices_loaded: false,
        }
    }

    /// Populate device lists from the system. Called once on first frame.
    fn load_devices(&mut self) {
        match device::list_input_devices() {
            Ok(inputs) => {
                self.ui.input_devices =
                    inputs.iter().map(|d| (d.name.clone(), d.display_name())).collect();
            }
            Err(e) => tracing::warn!("Failed to list input devices: {e}"),
        }
        match device::list_output_devices() {
            Ok(outputs) => {
                self.ui.output_devices =
                    outputs.iter().map(|d| (d.name.clone(), d.display_name())).collect();
            }
            Err(e) => tracing::warn!("Failed to list output devices: {e}"),
        }
    }

    /// Stop + restart the pipeline with the current device selections.
    fn restart_pipeline(&self) {
        self.pipeline.stop();
        let input = &self.ui.selected_input;
        let monitor = &self.ui.selected_output;
        if let Err(e) = self.pipeline.start(
            if input.is_empty() { None } else { Some(input.as_str()) },
            if monitor.is_empty() { None } else { Some(monitor.as_str()) },
            None,
        ) {
            tracing::error!("Pipeline restart failed: {e}");
        }
    }
}

impl SettingsApp {
    /// Run one frame of the settings UI (called by the window host).
    pub fn ui(&mut self, ctx: &egui::Context) {
        if !self.devices_loaded {
            self.load_devices();
            self.devices_loaded = true;
        }

        egui::CentralPanel::default().show(ctx, |ui| {
            egui::ScrollArea::vertical().show(ui, |ui| {
                ui.heading("Noise Gator");
                ui.add_space(4.0);

                self.render_meters(ui);
                ui.add_space(4.0);
                self.render_devices(ui);
                self.render_suppression(ui);
                self.render_gate(ui);
                self.render_clarity(ui);
                self.render_autogain(ui);
            });
        });

        // Mark settings dirty so audio callback picks up any changes
        self.settings.mark_dirty();

        // Repaints driven by tray event loop (ControlFlow::wait_duration)
    }
}

// ── Section renderers (split out for readability) ───────────────────────────

impl SettingsApp {
    fn render_meters(&mut self, ui: &mut egui::Ui) {
        let raw_in = f32::from_bits(AUDIO_LEVEL.load(Ordering::Relaxed));
        let raw_out = f32::from_bits(OUTPUT_LEVEL.load(Ordering::Relaxed));

        // Exponential smoothing: fast attack, slow release
        self.ui.input_meter = smooth(self.ui.input_meter, raw_in);
        self.ui.output_meter = smooth(self.ui.output_meter, raw_out);

        let in_db = rms_to_db(self.ui.input_meter);
        let out_db = rms_to_db(self.ui.output_meter);
        let in_norm = db_to_norm(in_db);
        let out_norm = db_to_norm(out_db);

        level_meter(ui, "In", in_norm, in_db);
        level_meter(ui, "Out", out_norm, out_db);
    }

    fn render_devices(&mut self, ui: &mut egui::Ui) {
        let id = ui.make_persistent_id("devices_section");
        egui::collapsing_header::CollapsingState::load_with_default_open(ui.ctx(), id, true)
            .show_header(ui, |ui| {
                ui.label(egui::RichText::new("Devices").strong());
            })
            .body(|ui| {
                // Input device
                let mut input_changed = false;
                ui.horizontal(|ui| {
                    ui.label("Input:");
                    let current_display = if self.ui.selected_input.is_empty() {
                        "System Default"
                    } else {
                        self.ui
                            .input_devices
                            .iter()
                            .find(|(name, _)| *name == self.ui.selected_input)
                            .map_or("(unknown)", |(_, display)| display.as_str())
                    };

                    egui::ComboBox::from_id_salt("input_device")
                        .selected_text(current_display)
                        .width(250.0)
                        .show_ui(ui, |ui| {
                            // "System Default" option — follows OS default device
                            if ui
                                .selectable_value(
                                    &mut self.ui.selected_input,
                                    String::new(),
                                    "System Default",
                                )
                                .changed()
                            {
                                input_changed = true;
                            }
                            for (name, display) in &self.ui.input_devices {
                                if ui
                                    .selectable_value(
                                        &mut self.ui.selected_input,
                                        name.clone(),
                                        display,
                                    )
                                    .changed()
                                {
                                    input_changed = true;
                                }
                            }
                        });
                });

                // Monitor (output) device
                let mut output_changed = false;
                ui.horizontal(|ui| {
                    ui.label("Monitor:");
                    let current_display = if self.ui.selected_output.is_empty() {
                        "None"
                    } else {
                        self.ui
                            .output_devices
                            .iter()
                            .find(|(name, _)| *name == self.ui.selected_output)
                            .map_or("(unknown)", |(_, display)| display.as_str())
                    };

                    egui::ComboBox::from_id_salt("output_device")
                        .selected_text(current_display)
                        .width(250.0)
                        .show_ui(ui, |ui| {
                            // "None" option
                            if ui
                                .selectable_value(
                                    &mut self.ui.selected_output,
                                    String::new(),
                                    "None",
                                )
                                .changed()
                            {
                                output_changed = true;
                            }
                            for (name, display) in &self.ui.output_devices {
                                if ui
                                    .selectable_value(
                                        &mut self.ui.selected_output,
                                        name.clone(),
                                        display,
                                    )
                                    .changed()
                                {
                                    output_changed = true;
                                }
                            }
                        });
                });

                if input_changed || output_changed {
                    self.restart_pipeline();
                }
            });
    }

    fn render_suppression(&mut self, ui: &mut egui::Ui) {
        let id = ui.make_persistent_id("suppression_section");
        egui::collapsing_header::CollapsingState::load_with_default_open(ui.ctx(), id, true)
            .show_header(ui, |ui| {
                ui.label(egui::RichText::new("Noise Suppression").strong());
            })
            .body(|ui| {
                // Engine selection
                let mut current_engine = self.settings.load_engine();
                let mut engine_changed = false;

                ui.horizontal(|ui| {
                    ui.label("Engine:");
                    if ui
                        .radio_value(
                            &mut current_engine,
                            DenoiseEngine::DeepFilter,
                            "DeepFilterNet",
                        )
                        .changed()
                    {
                        engine_changed = true;
                    }
                    if ui
                        .radio_value(&mut current_engine, DenoiseEngine::RNNoise, "RNNoise")
                        .changed()
                    {
                        engine_changed = true;
                    }
                });

                if engine_changed {
                    self.settings
                        .engine
                        .store(current_engine as u8, Ordering::Relaxed);
                    self.restart_pipeline();
                }

                // Suppression level slider
                ui.horizontal(|ui| {
                    ui.label("Level:");
                    let slider = egui::Slider::new(&mut self.ui.suppression_pct, 0.0..=100.0)
                        .step_by(1.0)
                        .suffix("%");
                    if ui.add(slider).changed() {
                        self.settings.suppression_level.store(
                            (self.ui.suppression_pct / 100.0).to_bits(),
                            Ordering::Relaxed,
                        );
                    }
                });
            });
    }

    fn render_gate(&mut self, ui: &mut egui::Ui) {
        let id = ui.make_persistent_id("gate_section");
        egui::collapsing_header::CollapsingState::load_with_default_open(ui.ctx(), id, true)
            .show_header(ui, |ui| {
                ui.label(egui::RichText::new("Voice Gate").strong());
            })
            .body(|ui| {
                // Enabled checkbox — gate_floor: enabled → 0.05, disabled → 1.0
                let floor = f32::from_bits(self.settings.gate_floor.load(Ordering::Relaxed));
                let mut gate_enabled = floor < GATE_FLOOR_BYPASS;

                if ui.checkbox(&mut gate_enabled, "Enabled").changed() {
                    let new_floor = if gate_enabled { GATE_FLOOR_ACTIVE } else { GATE_FLOOR_BYPASS };
                    self.settings
                        .gate_floor
                        .store(new_floor.to_bits(), Ordering::Relaxed);
                }

                // Sensitivity slider (only when gate is on)
                ui.add_enabled_ui(gate_enabled, |ui| {
                    ui.horizontal(|ui| {
                        ui.label("Sensitivity:");
                        let slider = egui::Slider::new(&mut self.ui.gate_sensitivity, 0.0..=100.0)
                            .step_by(1.0)
                            .suffix("%");
                        if ui.add(slider).changed() {
                            let threshold = GATE_THRESHOLD_MAX
                                - (self.ui.gate_sensitivity / 100.0) * GATE_THRESHOLD_RANGE;
                            self.settings
                                .gate_threshold
                                .store(threshold.to_bits(), Ordering::Relaxed);
                        }
                    });
                    ui.small("(Higher = more sensitive, gates less)");
                });
            });
    }

    fn render_clarity(&mut self, ui: &mut egui::Ui) {
        let id = ui.make_persistent_id("clarity_section");
        egui::collapsing_header::CollapsingState::load_with_default_open(ui.ctx(), id, true)
            .show_header(ui, |ui| {
                ui.label(egui::RichText::new("Voice Clarity").strong());
            })
            .body(|ui| {
                let mut eq_enabled = self.settings.eq_enabled.load(Ordering::Relaxed);
                if ui.checkbox(&mut eq_enabled, "Enabled").changed() {
                    self.settings.eq_enabled.store(eq_enabled, Ordering::Relaxed);
                }

                ui.add_enabled_ui(eq_enabled, |ui| {
                    if self.ui.advanced_eq {
                        self.render_clarity_advanced(ui);
                    } else {
                        self.render_clarity_simple(ui);
                    }
                });
            });
    }

    fn render_clarity_simple(&mut self, ui: &mut egui::Ui) {
        if ui.small_button("\u{25b8} Show Advanced").clicked() {
            self.ui.advanced_eq = true;
            self.settings.advanced_eq.store(true, Ordering::Relaxed);
        }

        ui.horizontal(|ui| {
            ui.label("Clarity:");
            let slider = egui::Slider::new(&mut self.ui.clarity_pct, 0.0..=100.0)
                .step_by(1.0)
                .suffix("%");
            if ui.add(slider).changed() {
                let pct = self.ui.clarity_pct / 100.0;
                let d = &self.ui.eq_defaults;
                self.ui.bass_db = pct * d.bass_db;
                self.ui.mid_db = pct * d.mid_db;
                self.ui.treble_db = pct * d.treble_db;
                store_eq(&self.settings, self.ui.bass_db, self.ui.mid_db, self.ui.treble_db);
            }
        });
    }

    fn render_clarity_advanced(&mut self, ui: &mut egui::Ui) {
        if ui.small_button("\u{25be} Hide Advanced").clicked() {
            self.ui.advanced_eq = false;
            self.settings.advanced_eq.store(false, Ordering::Relaxed);
        }

        let mut eq_changed = false;
        eq_changed |= eq_band_slider(ui, "Bass (300 Hz):", &mut self.ui.bass_db);
        eq_changed |= eq_band_slider(ui, "Mid (2.5 kHz):", &mut self.ui.mid_db);
        eq_changed |= eq_band_slider(ui, "Treble (6 kHz):", &mut self.ui.treble_db);

        if eq_changed {
            store_eq(&self.settings, self.ui.bass_db, self.ui.mid_db, self.ui.treble_db);
        }

        if ui.button("Reset to defaults").clicked() {
            let d = &self.ui.eq_defaults;
            self.ui.bass_db = d.bass_db;
            self.ui.mid_db = d.mid_db;
            self.ui.treble_db = d.treble_db;
            store_eq(&self.settings, d.bass_db, d.mid_db, d.treble_db);
        }
    }

    fn render_autogain(&mut self, ui: &mut egui::Ui) {
        let id = ui.make_persistent_id("autogain_section");
        egui::collapsing_header::CollapsingState::load_with_default_open(ui.ctx(), id, true)
            .show_header(ui, |ui| {
                ui.label(egui::RichText::new("Auto Gain").strong());
            })
            .body(|ui| {
                let mut enabled = self.settings.autogain_enabled.load(Ordering::Relaxed);
                if ui.checkbox(&mut enabled, "Enabled").changed() {
                    self.settings
                        .autogain_enabled
                        .store(enabled, Ordering::Relaxed);
                }
            });
    }
}

// ── Helpers ─────────────────────────────────────────────────────────────────

/// Store all three EQ band atomics at once.
fn store_eq(settings: &RuntimeSettings, bass: f32, mid: f32, treble: f32) {
    #[allow(clippy::cast_possible_truncation)]
    {
        settings
            .eq_bass_db10
            .store((bass * 10.0) as i32, Ordering::Relaxed);
        settings
            .eq_mid_db10
            .store((mid * 10.0) as i32, Ordering::Relaxed);
        settings
            .eq_treble_db10
            .store((treble * 10.0) as i32, Ordering::Relaxed);
    }
}

/// Exponential smoothing: fast attack, slow release.
fn smooth(current: f32, raw: f32) -> f32 {
    if raw > current {
        current * (1.0 - METER_ATTACK_COEFF) + raw * METER_ATTACK_COEFF
    } else {
        current * (1.0 - METER_RELEASE_COEFF) + raw * METER_RELEASE_COEFF
    }
}

/// Convert RMS amplitude to dB, clamped to [-60, 0].
fn rms_to_db(rms: f32) -> f32 {
    (20.0 * rms.max(1e-10).log10()).clamp(-60.0, 0.0)
}

/// Normalize dB value to 0.0..=1.0 range for progress bar.
fn db_to_norm(db: f32) -> f32 {
    (db + 60.0) / 60.0
}

/// Color-code a meter based on dB level.
fn meter_color(db: f32) -> egui::Color32 {
    if db < METER_YELLOW_DB {
        egui::Color32::from_rgb(76, 175, 80) // green
    } else if db < METER_RED_DB {
        egui::Color32::from_rgb(255, 193, 7) // yellow
    } else {
        egui::Color32::from_rgb(244, 67, 54) // red
    }
}

/// Draw a thin, non-interactive level meter bar (not a slider/progress bar).
fn level_meter(ui: &mut egui::Ui, label: &str, normalized: f32, db: f32) {
    ui.horizontal(|ui| {
        ui.label(format!("{label}:"));

        let bar_width = 220.0;
        let bar_height = 6.0;
        let (rect, _response) = ui.allocate_exact_size(
            egui::vec2(bar_width, bar_height),
            egui::Sense::hover(), // non-interactive
        );

        if ui.is_rect_visible(rect) {
            let painter = ui.painter();
            // Dark background track
            painter.rect_filled(rect, 2.0, egui::Color32::from_gray(40));
            // Filled portion
            let fill_width = rect.width() * normalized.clamp(0.0, 1.0);
            if fill_width > 0.5 {
                let fill_rect = egui::Rect::from_min_size(rect.min, egui::vec2(fill_width, bar_height));
                painter.rect_filled(fill_rect, 2.0, meter_color(db));
            }
        }

        ui.label(format!("{db:.0} dB"));
    });
}

/// Render a single EQ band slider (-12 to +12 dB). Returns true if changed.
fn eq_band_slider(ui: &mut egui::Ui, label: &str, value: &mut f32) -> bool {
    let mut changed = false;
    ui.horizontal(|ui| {
        ui.label(label);
        if ui
            .add(egui::Slider::new(value, -12.0..=12.0).step_by(0.5).suffix(" dB"))
            .changed()
        {
            changed = true;
        }
    });
    changed
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn smooth_attack_moves_toward_raw() {
        let result = smooth(0.0, 1.0);
        assert!((result - METER_ATTACK_COEFF).abs() < f32::EPSILON);
    }

    #[test]
    fn smooth_release_moves_slowly() {
        let result = smooth(1.0, 0.0);
        assert!((result - (1.0 - METER_RELEASE_COEFF)).abs() < f32::EPSILON);
    }

    #[test]
    fn smooth_no_change_when_equal() {
        let result = smooth(0.5, 0.5);
        assert!((result - 0.5).abs() < f32::EPSILON);
    }

    #[test]
    fn rms_to_db_zero_clamps_to_minus_60() {
        assert!((rms_to_db(0.0) - (-60.0)).abs() < 0.1);
    }

    #[test]
    fn rms_to_db_unity_is_zero() {
        assert!((rms_to_db(1.0) - 0.0).abs() < f32::EPSILON);
    }

    #[test]
    fn rms_to_db_known_value() {
        // 0.1 RMS = -20 dB
        assert!((rms_to_db(0.1) - (-20.0)).abs() < 0.01);
    }

    #[test]
    fn db_to_norm_boundaries() {
        assert!((db_to_norm(-60.0) - 0.0).abs() < f32::EPSILON);
        assert!((db_to_norm(0.0) - 1.0).abs() < f32::EPSILON);
        assert!((db_to_norm(-30.0) - 0.5).abs() < f32::EPSILON);
    }

    #[test]
    fn meter_color_thresholds() {
        assert_eq!(meter_color(-20.0), egui::Color32::from_rgb(76, 175, 80));
        assert_eq!(meter_color(-9.0), egui::Color32::from_rgb(255, 193, 7));
        assert_eq!(meter_color(-3.0), egui::Color32::from_rgb(244, 67, 54));
    }

    #[test]
    fn gate_sensitivity_round_trip() {
        // Forward: sensitivity -> threshold
        let sensitivity = 60.0;
        let threshold = GATE_THRESHOLD_MAX - (sensitivity / 100.0) * GATE_THRESHOLD_RANGE;
        // Reverse: threshold -> sensitivity
        let recovered = ((GATE_THRESHOLD_MAX - threshold) / GATE_THRESHOLD_RANGE * 100.0)
            .clamp(0.0, 100.0);
        assert!((recovered - sensitivity).abs() < 0.01);
    }

    #[test]
    fn gate_sensitivity_boundaries() {
        // 0% sensitivity -> max threshold (0.50)
        let t0 = GATE_THRESHOLD_MAX - (0.0 / 100.0) * GATE_THRESHOLD_RANGE;
        assert!((t0 - 0.50).abs() < f32::EPSILON);
        // 100% sensitivity -> min threshold (0.05)
        let t100 = GATE_THRESHOLD_MAX - (100.0 / 100.0) * GATE_THRESHOLD_RANGE;
        assert!((t100 - 0.05).abs() < f32::EPSILON);
    }
}

