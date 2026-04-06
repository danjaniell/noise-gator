use std::sync::atomic::Ordering;
use std::sync::Arc;

use muda::{
    CheckMenuItem, Menu, MenuEvent, MenuItem, PredefinedMenuItem, Submenu,
};
use winit::event_loop::ActiveEventLoop;

use crate::audio::device;
use crate::audio::pipeline::Pipeline;
use crate::config::{Config, DenoiseEngine};

/// Holds references to menu items we need to update dynamically.
pub struct MenuState {
    pub toggle_id: muda::MenuId,
    pub denoise_id: muda::MenuId,
    pub hard_mode_id: muda::MenuId,
    pub eq_enabled_id: muda::MenuId,
    pub quit_id: muda::MenuId,
    pub input_ids: Vec<(muda::MenuId, String)>,
    pub output_ids: Vec<(muda::MenuId, String)>,
    pub engine_rnnoise_id: muda::MenuId,
    pub engine_deepfilter_id: muda::MenuId,
}

pub fn build_menu(config: &Config, pipeline: &Arc<Pipeline>) -> (Menu, MenuState) {
    let menu = Menu::new();

    // ── Toggle on/off ───────────────────────────────────────────────
    let is_running = pipeline.is_running();
    let toggle = MenuItem::new(
        if is_running { "Stop" } else { "Start" },
        true,
        None,
    );
    let toggle_id = toggle.id().clone();
    let _ = menu.append(&toggle);

    let _ = menu.append(&PredefinedMenuItem::separator());

    // ── Input device submenu ────────────────────────────────────────
    let input_sub = Submenu::new("Input Device", true);
    let mut input_ids = Vec::new();
    if let Ok(devices) = device::list_input_devices() {
        for dev in &devices {
            let label = if dev.is_default {
                format!("{} (default)", dev.name)
            } else {
                dev.name.clone()
            };
            let is_selected = config
                .input_device
                .as_deref()
                .map_or(dev.is_default, |s| s == dev.name);
            let item = CheckMenuItem::new(&label, true, is_selected, None);
            input_ids.push((item.id().clone(), dev.name.clone()));
            let _ = input_sub.append(&item);
        }
    }
    let _ = menu.append(&input_sub);

    // ── Output device submenu ───────────────────────────────────────
    let output_sub = Submenu::new("Monitor Output", true);
    let mut output_ids = Vec::new();
    let none_item = CheckMenuItem::new(
        "None (no monitoring)",
        true,
        config.output_device.is_none(),
        None,
    );
    output_ids.push((none_item.id().clone(), String::new()));
    let _ = output_sub.append(&none_item);

    if let Ok(devices) = device::list_output_devices() {
        for dev in &devices {
            let is_selected = config
                .output_device
                .as_deref()
                .is_some_and(|s| s == dev.name);
            let item = CheckMenuItem::new(&dev.name, true, is_selected, None);
            output_ids.push((item.id().clone(), dev.name.clone()));
            let _ = output_sub.append(&item);
        }
    }
    let _ = menu.append(&output_sub);

    let _ = menu.append(&PredefinedMenuItem::separator());

    // ── DSP toggles ─────────────────────────────────────────────────
    let denoise = CheckMenuItem::new(
        "Noise Cancellation",
        true,
        pipeline.settings.denoise_enabled.load(Ordering::Relaxed),
        None,
    );
    let denoise_id = denoise.id().clone();
    let _ = menu.append(&denoise);

    let hard_mode = CheckMenuItem::new(
        "Hard Mode (aggressive gate)",
        true,
        pipeline.settings.hard_mode.load(Ordering::Relaxed),
        None,
    );
    let hard_mode_id = hard_mode.id().clone();
    let _ = menu.append(&hard_mode);

    let eq_enabled = CheckMenuItem::new(
        "EQ (warmth correction)",
        true,
        pipeline.settings.eq_enabled.load(Ordering::Relaxed),
        None,
    );
    let eq_enabled_id = eq_enabled.id().clone();
    let _ = menu.append(&eq_enabled);

    let _ = menu.append(&PredefinedMenuItem::separator());

    // ── Engine selection ────────────────────────────────────────────
    let engine_sub = Submenu::new("Engine", true);

    let is_rnnoise = config.engine == DenoiseEngine::RNNoise;
    let rnnoise_item = CheckMenuItem::new("RNNoise (default)", true, is_rnnoise, None);
    let engine_rnnoise_id = rnnoise_item.id().clone();
    let _ = engine_sub.append(&rnnoise_item);

    let df_label = if crate::models::is_deepfilter_available() {
        "DeepFilterNet"
    } else {
        "DeepFilterNet (Download ~8MB)"
    };
    let df_item = CheckMenuItem::new(df_label, true, !is_rnnoise, None);
    let engine_deepfilter_id = df_item.id().clone();
    let _ = engine_sub.append(&df_item);

    let _ = menu.append(&engine_sub);

    let _ = menu.append(&PredefinedMenuItem::separator());

    // ── Quit ────────────────────────────────────────────────────────
    let quit = MenuItem::new("Quit", true, None);
    let quit_id = quit.id().clone();
    let _ = menu.append(&quit);

    let state = MenuState {
        toggle_id,
        denoise_id,
        hard_mode_id,
        eq_enabled_id,
        quit_id,
        input_ids,
        output_ids,
        engine_rnnoise_id,
        engine_deepfilter_id,
    };

    (menu, state)
}

pub fn handle_event(
    event: &MenuEvent,
    state: &MenuState,
    pipeline: &Arc<Pipeline>,
    config: &mut Config,
    event_loop: &ActiveEventLoop,
) {
    let id = event.id();

    // ── Quit ────────────────────────────────────────────────────────
    if *id == state.quit_id {
        tracing::info!("Quit requested");
        pipeline.stop();
        // Save config before exit
        let updated = pipeline.settings.to_config(config);
        if let Err(e) = updated.save() {
            tracing::error!("Failed to save config: {e}");
        }
        event_loop.exit();
        return;
    }

    // ── Toggle pipeline ─────────────────────────────────────────────
    if *id == state.toggle_id {
        if pipeline.is_running() {
            pipeline.stop();
        } else if let Err(e) = pipeline.start(
            config.input_device.as_deref(),
            config.output_device.as_deref(),
            config.virtual_device.as_deref(),
        ) {
            tracing::error!("Failed to start pipeline: {e}");
        }
        return;
    }

    // ── Denoise toggle ──────────────────────────────────────────────
    if *id == state.denoise_id {
        let current = pipeline.settings.denoise_enabled.load(Ordering::Relaxed);
        pipeline
            .settings
            .denoise_enabled
            .store(!current, Ordering::Relaxed);
        tracing::info!("Denoise: {}", !current);
        return;
    }

    // ── Hard mode toggle ────────────────────────────────────────────
    if *id == state.hard_mode_id {
        let current = pipeline.settings.hard_mode.load(Ordering::Relaxed);
        pipeline
            .settings
            .hard_mode
            .store(!current, Ordering::Relaxed);
        tracing::info!("Hard mode: {}", !current);
        return;
    }

    // ── EQ toggle ───────────────────────────────────────────────────
    if *id == state.eq_enabled_id {
        let current = pipeline.settings.eq_enabled.load(Ordering::Relaxed);
        pipeline
            .settings
            .eq_enabled
            .store(!current, Ordering::Relaxed);
        tracing::info!("EQ: {}", !current);
        return;
    }

    // ── Engine selection ────────────────────────────────────────────
    if *id == state.engine_rnnoise_id {
        if config.engine != DenoiseEngine::RNNoise {
            config.engine = DenoiseEngine::RNNoise;
            pipeline.settings.engine.store(0, Ordering::Relaxed);
            tracing::info!("Switching to RNNoise");
            pipeline.stop();
            if let Err(e) = pipeline.start(
                config.input_device.as_deref(),
                config.output_device.as_deref(),
                config.virtual_device.as_deref(),
            ) {
                tracing::error!("Failed to restart pipeline: {e}");
            }
        }
        return;
    }

    if *id == state.engine_deepfilter_id {
        if config.engine != DenoiseEngine::DeepFilter {
            tracing::info!("Switching to DeepFilterNet...");
            pipeline.stop();

            // Download model if needed (blocking with user notification)
            if !crate::models::is_deepfilter_available() {
                show_message("Noise Gator", "Downloading DeepFilterNet model (~8MB).\nThis may take a moment.");
                match crate::models::ensure_deepfilter_model() {
                    Ok(_) => {
                        tracing::info!("DeepFilterNet model installed.");
                    }
                    Err(e) => {
                        tracing::error!("Model download failed: {e}");
                        show_message("Noise Gator", &format!("Download failed: {e}"));
                        // Restart with RNNoise
                        let _ = pipeline.start(
                            config.input_device.as_deref(),
                            config.output_device.as_deref(),
                            config.virtual_device.as_deref(),
                        );
                        return;
                    }
                }
            }

            config.engine = DenoiseEngine::DeepFilter;
            pipeline.settings.engine.store(1, Ordering::Relaxed);

            if let Err(e) = pipeline.start(
                config.input_device.as_deref(),
                config.output_device.as_deref(),
                config.virtual_device.as_deref(),
            ) {
                tracing::error!("Failed to restart with DeepFilter: {e}");
                // Fall back to RNNoise
                config.engine = DenoiseEngine::RNNoise;
                pipeline.settings.engine.store(0, Ordering::Relaxed);
                let _ = pipeline.start(
                    config.input_device.as_deref(),
                    config.output_device.as_deref(),
                    config.virtual_device.as_deref(),
                );
            }
        }
        return;
    }

    // ── Input device selection ──────────────────────────────────────
    for (item_id, device_name) in &state.input_ids {
        if id == item_id {
            config.input_device = Some(device_name.clone());
            tracing::info!("Input device: {device_name}");
            if pipeline.is_running() {
                pipeline.stop();
                if let Err(e) = pipeline.start(
                    Some(device_name.as_str()),
                    config.output_device.as_deref(),
                    config.virtual_device.as_deref(),
                ) {
                    tracing::error!("Failed to restart pipeline: {e}");
                }
            }
            return;
        }
    }

    // ── Output device selection ─────────────────────────────────────
    for (item_id, device_name) in &state.output_ids {
        if id == item_id {
            config.output_device = if device_name.is_empty() {
                None
            } else {
                Some(device_name.clone())
            };
            tracing::info!(
                "Monitor output: {}",
                config.output_device.as_deref().unwrap_or("none")
            );
            if pipeline.is_running() {
                pipeline.stop();
                if let Err(e) = pipeline.start(
                    config.input_device.as_deref(),
                    config.output_device.as_deref(),
                    config.virtual_device.as_deref(),
                ) {
                    tracing::error!("Failed to restart pipeline: {e}");
                }
            }
            return;
        }
    }
}

/// Show a simple message box. Best-effort — logs on failure.
#[cfg(target_os = "windows")]
fn show_message(title: &str, message: &str) {
    use windows::core::HSTRING;
    use windows::Win32::UI::WindowsAndMessaging::{MessageBoxW, MB_OK, MB_ICONINFORMATION};
    let title = HSTRING::from(title);
    let message = HSTRING::from(message);
    unsafe {
        let _ = MessageBoxW(None, &message, &title, MB_OK | MB_ICONINFORMATION);
    }
}

#[cfg(not(target_os = "windows"))]
fn show_message(_title: &str, message: &str) {
    tracing::info!("{message}");
}
