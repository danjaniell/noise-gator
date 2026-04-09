mod menu;

use std::sync::Arc;
use std::time::Duration;

use anyhow::Result;
use tray_icon::TrayIconBuilder;
use winit::application::ApplicationHandler;
use winit::event::WindowEvent;
use winit::event_loop::{ActiveEventLoop, ControlFlow, EventLoop};
use winit::window::WindowId;

use crate::audio::pipeline::Pipeline;
use crate::config::Config;
use crate::ui::window::SettingsWindowState;

use menu::MenuState;

/// Run the system tray event loop (blocks until quit).
pub fn run(pipeline: Arc<Pipeline>, config: Config) -> Result<()> {
    let event_loop = EventLoop::new()?;

    let mut app = TrayApp {
        pipeline,
        config,
        menu_state: None,
        tray_icon: None,
        settings_window: None,
        open_settings: false,
    };

    event_loop.run_app(&mut app)?;
    Ok(())
}

struct TrayApp {
    pipeline: Arc<Pipeline>,
    config: Config,
    menu_state: Option<MenuState>,
    tray_icon: Option<tray_icon::TrayIcon>,
    settings_window: Option<SettingsWindowState>,
    open_settings: bool,
}

impl ApplicationHandler for TrayApp {
    fn resumed(&mut self, _event_loop: &ActiveEventLoop) {
        if self.tray_icon.is_some() {
            return; // Already initialized
        }

        let (menu, menu_state) = menu::build_menu(&self.config, &self.pipeline);
        self.menu_state = Some(menu_state);

        let icon = load_icon();

        match TrayIconBuilder::new()
            .with_menu(Box::new(menu))
            .with_tooltip("Noise Gator")
            .with_icon(icon)
            .build()
        {
            Ok(tray) => {
                self.tray_icon = Some(tray);
                tracing::info!("System tray initialized");
            }
            Err(e) => {
                tracing::error!("Failed to create tray icon: {e}");
            }
        }
    }

    fn window_event(
        &mut self,
        _event_loop: &ActiveEventLoop,
        window_id: WindowId,
        event: WindowEvent,
    ) {
        let is_settings = self
            .settings_window
            .as_ref()
            .is_some_and(|sw| sw.window_id() == window_id);

        if !is_settings {
            return;
        }

        match event {
            WindowEvent::CloseRequested => {
                self.settings_window.take(); // Drop impl handles cleanup
                tracing::info!("Settings window closed");
            }
            WindowEvent::RedrawRequested => {
                if let Some(ref mut sw) = self.settings_window {
                    sw.render();
                }
            }
            ref other => {
                if let Some(ref mut sw) = self.settings_window {
                    sw.on_window_event(other);
                }
            }
        }
    }

    fn about_to_wait(&mut self, event_loop: &ActiveEventLoop) {
        // Drain menu events FIRST — before rebuilding the menu from tray
        // icon events. Rebuilding creates new MenuIds; if we rebuild first,
        // clicks on the old (visible) menu carry stale IDs that no longer
        // match menu_state, so the event is silently dropped.
        while let Ok(event) = muda::MenuEvent::receiver().try_recv() {
            if let Some(ref menu_state) = self.menu_state {
                if *event.id() == menu_state.settings_id {
                    self.open_settings = true;
                } else {
                    menu::handle_event(
                        &event,
                        menu_state,
                        &self.pipeline,
                        &mut self.config,
                        event_loop,
                    );
                }
            }
        }

        // Rebuild menu on tray icon click so device list stays fresh
        // (e.g., Bluetooth headset connected after app started).
        // This runs AFTER menu event processing to avoid invalidating IDs.
        while let Ok(_event) = tray_icon::TrayIconEvent::receiver().try_recv() {
            let (menu, menu_state) = menu::build_menu(&self.config, &self.pipeline);
            self.menu_state = Some(menu_state);
            if let Some(ref tray) = self.tray_icon {
                let _ = tray.set_menu(Some(Box::new(menu)));
            }
        }

        // Open settings window after processing events (needs ActiveEventLoop)
        if self.open_settings {
            self.open_settings = false;
            if self.settings_window.is_none() {
                tracing::info!("Opening settings window");
                self.settings_window = SettingsWindowState::new(
                    event_loop,
                    Arc::clone(&self.pipeline.settings),
                    Arc::clone(&self.pipeline),
                    self.config.input_device.as_deref(),
                    self.config.output_device.as_deref(),
                );
            }
        }

        // Drive meter animation when settings window is open
        if let Some(ref sw) = self.settings_window {
            event_loop.set_control_flow(ControlFlow::wait_duration(Duration::from_millis(33)));
            sw.request_redraw();
        } else {
            event_loop.set_control_flow(ControlFlow::Wait);
        }
    }
}

/// Load the pre-sized 32x32 app icon from the embedded PNG for use as the system tray icon.
fn load_icon() -> tray_icon::Icon {
    let png_bytes = include_bytes!("../../img/noise-gator-32.png");
    let cursor = std::io::Cursor::new(png_bytes.as_slice());
    let decoder = png::Decoder::new(cursor);
    let mut reader = decoder.read_info().expect("Failed to read PNG header");
    let mut buf = vec![0u8; reader.output_buffer_size().expect("PNG output buffer size")];
    let info = reader.next_frame(&mut buf).expect("Failed to decode PNG");
    let rgba = buf[..info.buffer_size()].to_vec();
    tray_icon::Icon::from_rgba(rgba, info.width, info.height).expect("Failed to create tray icon")
}
