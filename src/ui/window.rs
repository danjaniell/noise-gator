//! Settings window backed by glutin + egui_glow, sharing the tray event loop.
//!
//! Every GL/glutin API call requires unsafe — hence the file-level allow.

#![allow(unsafe_code)]

use std::num::NonZeroU32;
use std::sync::Arc;

use egui_glow::EguiGlow;
use glow::HasContext as _;
use glutin::config::ConfigTemplateBuilder;
use glutin::context::{ContextApi, ContextAttributesBuilder, NotCurrentGlContext, PossiblyCurrentGlContext};
use glutin::display::{GetGlDisplay, GlDisplay};
use glutin::surface::GlSurface;
use raw_window_handle::HasWindowHandle;
use winit::event_loop::ActiveEventLoop;
use winit::window::{Window, WindowId};

use crate::audio::pipeline::Pipeline;
use crate::config::RuntimeSettings;

use super::settings::SettingsApp;

// ── GL context wrapper ─────────────────────────────────────────────────────

struct GlutinContext {
    window: Window,
    gl_context: Option<glutin::context::PossiblyCurrentContext>,
    gl_surface: glutin::surface::Surface<glutin::surface::WindowSurface>,
}

impl GlutinContext {
    /// Create a new GL-backed window on the given event loop.
    ///
    /// # Safety
    /// - Must be called on the main thread (winit requirement).
    /// - `event_loop` must be in the resumed/active state.
    /// - Calls into platform GL display/context/surface APIs.
    ///
    /// # Errors
    /// Returns `Err` if GL config, context, or surface creation fails.
    unsafe fn new(event_loop: &ActiveEventLoop) -> Result<Self, String> {
        let window_attrs = Window::default_attributes()
            .with_resizable(true)
            .with_inner_size(winit::dpi::LogicalSize::new(400.0_f64, 550.0))
            .with_min_inner_size(winit::dpi::LogicalSize::new(350.0_f64, 400.0))
            .with_title("Noise Gator Settings")
            .with_visible(false);

        let config_template = ConfigTemplateBuilder::new()
            .prefer_hardware_accelerated(None)
            .with_depth_size(0)
            .with_stencil_size(0)
            .with_transparency(false);

        let (mut window, gl_config) = glutin_winit::DisplayBuilder::new()
            .with_preference(glutin_winit::ApiPreference::FallbackEgl)
            .with_window_attributes(Some(window_attrs.clone()))
            .build(event_loop, config_template, |mut configs| {
                configs.next().expect("no matching GL config")
            })
            .map_err(|e| format!("GL display: {e}"))?;

        let gl_display = gl_config.display();

        let raw_handle = window
            .as_ref()
            .map(|w| w.window_handle().expect("window handle").as_raw());

        let context_attrs = ContextAttributesBuilder::new().build(raw_handle);
        let fallback_attrs = ContextAttributesBuilder::new()
            .with_context_api(ContextApi::Gles(None))
            .build(raw_handle);

        let not_current = unsafe {
            gl_display
                .create_context(&gl_config, &context_attrs)
                .or_else(|_| gl_display.create_context(&gl_config, &fallback_attrs))
                .map_err(|e| format!("GL context: {e}"))?
        };

        let window = window.take().unwrap_or_else(|| {
            glutin_winit::finalize_window(event_loop, window_attrs, &gl_config)
                .expect("failed to finalize window")
        });

        let (w, h): (u32, u32) = window.inner_size().into();
        let w = NonZeroU32::new(w).unwrap_or(NonZeroU32::MIN);
        let h = NonZeroU32::new(h).unwrap_or(NonZeroU32::MIN);

        let surface_attrs =
            glutin::surface::SurfaceAttributesBuilder::<glutin::surface::WindowSurface>::new()
                .build(
                    window.window_handle().expect("window handle").as_raw(),
                    w,
                    h,
                );

        let gl_surface = unsafe {
            gl_display
                .create_window_surface(&gl_config, &surface_attrs)
                .map_err(|e| format!("GL surface: {e}"))?
        };

        let gl_context = not_current
            .make_current(&gl_surface)
            .map_err(|e| format!("GL make_current: {e}"))?;

        gl_surface
            .set_swap_interval(
                &gl_context,
                glutin::surface::SwapInterval::Wait(NonZeroU32::MIN),
            )
            .ok();

        Ok(Self {
            window,
            gl_context: Some(gl_context),
            gl_surface,
        })
    }

    fn resize(&self, size: winit::dpi::PhysicalSize<u32>) {
        if let (Some(w), Some(h)) = (NonZeroU32::new(size.width), NonZeroU32::new(size.height)) {
            if let Some(ref ctx) = self.gl_context {
                self.gl_surface.resize(ctx, w, h);
            }
        }
    }

    fn swap_buffers(&self) {
        if let Some(ref ctx) = self.gl_context {
            self.gl_surface.swap_buffers(ctx).ok();
        }
    }

    fn get_proc_address(&self, addr: &std::ffi::CStr) -> *const std::ffi::c_void {
        self.gl_context
            .as_ref()
            .map(|ctx| ctx.display().get_proc_address(addr))
            .unwrap_or(std::ptr::null())
    }

    /// Release the GL context before drop to ensure clean teardown.
    fn make_not_current(&mut self) {
        if let Some(ctx) = self.gl_context.take() {
            ctx.make_not_current().ok();
        }
    }
}

impl Drop for GlutinContext {
    fn drop(&mut self) {
        self.make_not_current();
    }
}

// ── Public settings window state ───────────────────────────────────────────

/// Everything needed to drive the settings window inside the tray event loop.
pub struct SettingsWindowState {
    glutin_ctx: GlutinContext,
    gl: Arc<glow::Context>,
    egui_glow: EguiGlow,
    pub app: SettingsApp,
}

impl SettingsWindowState {
    /// Create and show the settings window. Must be called from within an
    /// `ApplicationHandler` method that has access to `ActiveEventLoop`.
    /// Returns `None` if GL initialization fails.
    pub fn new(
        event_loop: &ActiveEventLoop,
        settings: Arc<RuntimeSettings>,
        pipeline: Arc<Pipeline>,
    ) -> Option<Self> {
        let glutin_ctx = match unsafe { GlutinContext::new(event_loop) } {
            Ok(ctx) => ctx,
            Err(e) => {
                tracing::error!("Failed to create settings window: {e}");
                return None;
            }
        };

        let gl = unsafe {
            Arc::new(glow::Context::from_loader_function(|s| {
                let s = std::ffi::CString::new(s).expect("CString for GL proc");
                glutin_ctx.get_proc_address(&s)
            }))
        };

        let egui_glow = EguiGlow::new(event_loop, gl.clone(), None, None, true);
        egui_glow.egui_ctx.set_visuals(egui::Visuals::dark());

        glutin_ctx.window.set_visible(true);

        Some(Self {
            glutin_ctx,
            gl,
            egui_glow,
            app: SettingsApp::new(settings, pipeline),
        })
    }

    /// The winit `WindowId` so the tray can route events.
    pub fn window_id(&self) -> WindowId {
        self.glutin_ctx.window.id()
    }

    /// Forward a winit event; returns `true` if egui wants a repaint.
    pub fn on_window_event(&mut self, event: &winit::event::WindowEvent) -> bool {
        if let winit::event::WindowEvent::Resized(size) = event {
            self.glutin_ctx.resize(*size);
        }
        let resp = self
            .egui_glow
            .on_window_event(&self.glutin_ctx.window, event);
        if resp.repaint {
            self.glutin_ctx.window.request_redraw();
        }
        resp.repaint
    }

    /// Run one egui frame and paint it.
    pub fn render(&mut self) {
        let app = &mut self.app;
        self.egui_glow.run(&self.glutin_ctx.window, |ctx| {
            app.ui(ctx);
        });

        unsafe {
            self.gl.clear_color(0.1, 0.1, 0.1, 1.0);
            self.gl.clear(glow::COLOR_BUFFER_BIT);
        }

        self.egui_glow.paint(&self.glutin_ctx.window);
        self.glutin_ctx.swap_buffers();
    }

    /// Ask the OS to schedule a `RedrawRequested` for the settings window.
    pub fn request_redraw(&self) {
        self.glutin_ctx.window.request_redraw();
    }
}

impl Drop for SettingsWindowState {
    fn drop(&mut self) {
        self.egui_glow.destroy();
        // GlutinContext::drop handles make_not_current
    }
}
