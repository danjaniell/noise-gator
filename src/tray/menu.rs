use std::sync::Arc;
use std::sync::atomic::Ordering;

use muda::{CheckMenuItem, Menu, MenuEvent, MenuItem, PredefinedMenuItem, Submenu};
use winit::event_loop::ActiveEventLoop;

use crate::audio::device;
use crate::audio::pipeline::Pipeline;
use crate::config::{Config, DenoiseEngine};

/// Holds references to menu items we need to update dynamically.
pub struct MenuState {
    pub toggle_id: muda::MenuId,
    pub denoise_id: muda::MenuId,
    pub eq_enabled_id: muda::MenuId,
    pub settings_id: muda::MenuId,
    pub quit_id: muda::MenuId,
    pub input_items: Vec<(CheckMenuItem, String)>,
    pub output_items: Vec<(CheckMenuItem, String)>,
    pub engine_rnnoise_id: muda::MenuId,
    pub engine_deepfilter_id: muda::MenuId,
    // Keep CheckMenuItem refs to control checked state programmatically
    pub engine_rnnoise_item: CheckMenuItem,
    pub engine_deepfilter_item: CheckMenuItem,
}

pub fn build_menu(config: &Config, pipeline: &Arc<Pipeline>) -> (Menu, MenuState) {
    let menu = Menu::new();

    // ── Toggle on/off ───────────────────────────────────────────────
    let is_running = pipeline.is_running();
    let toggle = MenuItem::new(if is_running { "Stop" } else { "Start" }, true, None);
    let toggle_id = toggle.id().clone();
    let _ = menu.append(&toggle);

    let _ = menu.append(&PredefinedMenuItem::separator());

    // ── Input device submenu ────────────────────────────────────────
    let input_sub = Submenu::new("Input Device", true);
    let mut input_items = Vec::new();
    // "System Default" option — uses whatever the OS default input is
    let sys_default_input = CheckMenuItem::new(
        "System Default",
        true,
        config.input_device.is_none(),
        None,
    );
    input_items.push((sys_default_input.clone(), String::new()));
    let _ = input_sub.append(&sys_default_input);
    let _ = input_sub.append(&PredefinedMenuItem::separator());

    if let Ok(devices) = device::list_input_devices() {
        for dev in &devices {
            let is_selected = config
                .input_device
                .as_deref()
                .is_some_and(|s| s == dev.name);
            let item = CheckMenuItem::new(&dev.display_name(), true, is_selected, None);
            input_items.push((item.clone(), dev.name.clone()));
            let _ = input_sub.append(&item);
        }
    }
    let _ = menu.append(&input_sub);

    // ── Output device submenu ───────────────────────────────────────
    let output_sub = Submenu::new("Monitor Output", true);
    let mut output_items = Vec::new();
    let none_item = CheckMenuItem::new(
        "None (no monitoring)",
        true,
        config.output_device.is_none(),
        None,
    );
    output_items.push((none_item.clone(), String::new()));
    let _ = output_sub.append(&none_item);

    if let Ok(devices) = device::list_output_devices() {
        for dev in &devices {
            let is_selected = config
                .output_device
                .as_deref()
                .is_some_and(|s| s == dev.name);
            let item = CheckMenuItem::new(&dev.name, true, is_selected, None);
            output_items.push((item.clone(), dev.name.clone()));
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
    let engine_rnnoise_item = CheckMenuItem::new("RNNoise (lightweight)", true, is_rnnoise, None);
    let engine_rnnoise_id = engine_rnnoise_item.id().clone();
    let _ = engine_sub.append(&engine_rnnoise_item);

    let df_label = if crate::models::is_deepfilter_available() {
        "DeepFilterNet"
    } else {
        "DeepFilterNet (Download ~8MB)"
    };
    let engine_deepfilter_item = CheckMenuItem::new(df_label, true, !is_rnnoise, None);
    let engine_deepfilter_id = engine_deepfilter_item.id().clone();
    let _ = engine_sub.append(&engine_deepfilter_item);

    let _ = menu.append(&engine_sub);

    let _ = menu.append(&PredefinedMenuItem::separator());

    // ── Settings window ────────────────────────────────────────────
    let settings_item = MenuItem::new("Settings...", true, None);
    let settings_id = settings_item.id().clone();
    let _ = menu.append(&settings_item);

    // ── Quit ────────────────────────────────────────────────────────
    let quit = MenuItem::new("Quit", true, None);
    let quit_id = quit.id().clone();
    let _ = menu.append(&quit);

    let state = MenuState {
        toggle_id,
        denoise_id,
        eq_enabled_id,
        settings_id,
        quit_id,
        input_items,
        output_items,
        engine_rnnoise_id,
        engine_deepfilter_id,
        engine_rnnoise_item,
        engine_deepfilter_item,
    };

    (menu, state)
}

/// Set engine checkmarks — ensures exactly one is checked.
fn set_engine_checked(state: &MenuState, engine: DenoiseEngine) {
    let is_rnnoise = engine == DenoiseEngine::RNNoise;
    state.engine_rnnoise_item.set_checked(is_rnnoise);
    state.engine_deepfilter_item.set_checked(!is_rnnoise);
}

/// Revert to RNNoise: update config, checkmarks, and restart pipeline.
fn revert_to_rnnoise(
    state: &MenuState,
    pipeline: &Arc<Pipeline>,
    config: &mut Config,
) {
    config.engine = DenoiseEngine::RNNoise;
    pipeline
        .settings
        .engine
        .store(DenoiseEngine::RNNoise.as_u8(), Ordering::Relaxed);
    set_engine_checked(state, DenoiseEngine::RNNoise);
    let _ = pipeline.start(
        config.input_device.as_deref(),
        config.output_device.as_deref(),
        config.virtual_device.as_deref(),
    );
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

    // ── Settings window ────────────────────────────────────────────
    // Handled by TrayApp via the open_settings flag — see tray/mod.rs

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
            tracing::info!("Switching to RNNoise");
            pipeline.stop();
            revert_to_rnnoise(state, pipeline, config);
        } else {
            set_engine_checked(state, DenoiseEngine::RNNoise);
        }
        return;
    }

    if *id == state.engine_deepfilter_id {
        if config.engine != DenoiseEngine::DeepFilter {
            tracing::info!("Switching to DeepFilterNet...");
            pipeline.stop();

            // Initialize ONNX Runtime + download model if needed
            if !crate::models::is_ort_available() || !crate::models::is_deepfilter_available() {
                show_message(
                    "Noise Gator",
                    "Downloading DeepFilterNet dependencies.\nThis may take a moment.",
                );
            }

            if let Err(e) = crate::models::init_ort() {
                tracing::error!("ONNX Runtime init failed: {e}");
                show_message("Noise Gator", &format!("ONNX Runtime setup failed: {e}"));
                revert_to_rnnoise(state, pipeline, config);
                return;
            }

            if !crate::models::is_deepfilter_available() {
                match crate::models::ensure_deepfilter_model() {
                    Ok(_) => tracing::info!("DeepFilterNet model installed."),
                    Err(e) => {
                        tracing::error!("Model download failed: {e}");
                        show_message("Noise Gator", &format!("Download failed: {e}"));
                        revert_to_rnnoise(state, pipeline, config);
                        return;
                    }
                }
            }

            config.engine = DenoiseEngine::DeepFilter;
            pipeline
                .settings
                .engine
                .store(DenoiseEngine::DeepFilter.as_u8(), Ordering::Relaxed);
            set_engine_checked(state, DenoiseEngine::DeepFilter);

            if let Err(e) = pipeline.start(
                config.input_device.as_deref(),
                config.output_device.as_deref(),
                config.virtual_device.as_deref(),
            ) {
                tracing::error!("Failed to restart with DeepFilter: {e}");
                revert_to_rnnoise(state, pipeline, config);
            }
        } else {
            set_engine_checked(state, DenoiseEngine::DeepFilter);
        }
        return;
    }

    // ── Input device selection ──────────────────────────────────────
    for (item, device_name) in &state.input_items {
        if *id == *item.id() {
            // Empty name = "System Default" → config None
            config.input_device = if device_name.is_empty() {
                None
            } else {
                Some(device_name.clone())
            };
            // Enforce mutual exclusivity: uncheck all, check selected
            for (other, _) in &state.input_items {
                other.set_checked(*other.id() == *item.id());
            }
            tracing::info!(
                "Input device: {}",
                config.input_device.as_deref().unwrap_or("system default")
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

    // ── Output device selection ─────────────────────────────────────
    for (item, device_name) in &state.output_items {
        if *id == *item.id() {
            config.output_device = if device_name.is_empty() {
                None
            } else {
                Some(device_name.clone())
            };
            // Enforce mutual exclusivity: uncheck all, check selected
            for (other, _) in &state.output_items {
                other.set_checked(*other.id() == *item.id());
            }
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
    use windows::Win32::UI::WindowsAndMessaging::{MB_ICONINFORMATION, MB_OK, MessageBoxW};
    use windows::core::HSTRING;
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
