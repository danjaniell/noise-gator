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

/// Load the app icon from the embedded PNG for use as the system tray icon.
fn load_icon() -> tray_icon::Icon {
    let png_bytes = include_bytes!("../../img/noise-gator.png");
    let img = image::load_from_memory_with_format(png_bytes, image::ImageFormat::Png)
        .expect("Failed to decode embedded icon PNG");
    let resized = img.resize(32, 32, image::imageops::FilterType::Lanczos3);
    let rgba = resized.to_rgba8();
    let (w, h) = rgba.dimensions();
    tray_icon::Icon::from_rgba(rgba.into_raw(), w, h).expect("Failed to create tray icon")
}
