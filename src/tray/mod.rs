mod menu;

use std::sync::Arc;

use anyhow::Result;
use tray_icon::TrayIconBuilder;
use winit::application::ApplicationHandler;
use winit::event::WindowEvent;
use winit::event_loop::{ActiveEventLoop, EventLoop};
use winit::window::WindowId;

use crate::audio::pipeline::Pipeline;
use crate::config::Config;

use menu::MenuState;

/// Run the system tray event loop (blocks until quit).
pub fn run(pipeline: Arc<Pipeline>, config: Config) -> Result<()> {
    let event_loop = EventLoop::new()?;

    let mut app = TrayApp {
        pipeline,
        config,
        menu_state: None,
        _tray_icon: None,
    };

    event_loop.run_app(&mut app)?;
    Ok(())
}

struct TrayApp {
    pipeline: Arc<Pipeline>,
    config: Config,
    menu_state: Option<MenuState>,
    _tray_icon: Option<tray_icon::TrayIcon>,
}

impl ApplicationHandler for TrayApp {
    fn resumed(&mut self, _event_loop: &ActiveEventLoop) {
        if self._tray_icon.is_some() {
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
                self._tray_icon = Some(tray);
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
        _window_id: WindowId,
        _event: WindowEvent,
    ) {
    }

    fn about_to_wait(&mut self, _event_loop: &ActiveEventLoop) {
        // Process tray menu events
        if let Ok(event) = muda::MenuEvent::receiver().try_recv() {
            if let Some(ref menu_state) = self.menu_state {
                menu::handle_event(
                    &event,
                    menu_state,
                    &self.pipeline,
                    &mut self.config,
                    _event_loop,
                );
            }
        }
    }
}

/// Generate a simple colored icon in memory (no external file needed).
fn load_icon() -> tray_icon::Icon {
    let size = 32u32;
    let mut rgba = vec![0u8; (size * size * 4) as usize];

    // Draw a green circle on transparent background
    let center = size as f32 / 2.0;
    let radius = center - 2.0;
    for y in 0..size {
        for x in 0..size {
            let dx = x as f32 - center;
            let dy = y as f32 - center;
            let dist = (dx * dx + dy * dy).sqrt();
            let idx = ((y * size + x) * 4) as usize;
            if dist <= radius {
                // Green: #4CAF50
                rgba[idx] = 0x4C;     // R
                rgba[idx + 1] = 0xAF; // G
                rgba[idx + 2] = 0x50; // B
                rgba[idx + 3] = 0xFF; // A
            }
        }
    }

    tray_icon::Icon::from_rgba(rgba, size, size).expect("Failed to create tray icon")
}
