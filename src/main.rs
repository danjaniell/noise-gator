#![cfg_attr(
    all(not(debug_assertions), target_os = "windows"),
    windows_subsystem = "windows"
)]

mod audio;
mod config;
mod driver;
mod dsp;
mod tray;

use std::sync::Arc;

use clap::Parser;

use audio::device;
use audio::pipeline::Pipeline;
use config::cli::Cli;
use config::{Config, RuntimeSettings};

fn main() {
    // Parse CLI first (before logging — --list-devices should be clean)
    let cli = Cli::parse();

    // Init logging
    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::from_default_env()
                .add_directive("noise_gator=info".parse().unwrap()),
        )
        .init();

    // ── List devices mode ───────────────────────────────────────────
    if cli.list_devices {
        print_devices();
        return;
    }

    // ── Uninstall driver mode ───────────────────────────────────────
    if cli.uninstall_driver {
        match driver::uninstall() {
            Ok(()) => println!("Driver uninstalled."),
            Err(e) => {
                eprintln!("Failed to uninstall driver: {e}");
                std::process::exit(1);
            }
        }
        return;
    }

    // ── Load config (CLI overrides file) ────────────────────────────
    let mut config = if let Some(ref path) = cli.config {
        match config::Config::load_from_path(path) {
            Ok(c) => c,
            Err(e) => {
                tracing::warn!("Failed to load config from {path}: {e}");
                Config::default()
            }
        }
    } else {
        Config::load()
    };

    // CLI overrides
    if cli.input.is_some() {
        config.input_device = cli.input;
    }
    if cli.monitor.is_some() {
        config.output_device = cli.monitor;
    }
    if cli.virtual_device.is_some() {
        config.virtual_device = cli.virtual_device;
    }
    if cli.no_denoise {
        config.denoise_enabled = false;
    }
    if cli.hard_mode {
        config.hard_mode = true;
    }

    // ── Ensure virtual audio driver ─────────────────────────────────
    if !cli.skip_driver {
        if let Err(e) = driver::ensure_installed() {
            tracing::error!("Driver setup failed: {e}");
            eprintln!(
                "Virtual audio driver setup failed: {e}\n\
                 Run with --skip-driver to bypass, or install manually."
            );
            std::process::exit(1);
        }
    }

    // ── Build pipeline ──────────────────────────────────────────────
    let settings = Arc::new(RuntimeSettings::from_config(&config));
    let pipeline = Arc::new(Pipeline::new(Arc::clone(&settings)));

    // Auto-start the pipeline
    if let Err(e) = pipeline.start(
        config.input_device.as_deref(),
        config.output_device.as_deref(),
        config.virtual_device.as_deref(),
    ) {
        tracing::error!("Failed to start audio pipeline: {e}");
        eprintln!("Failed to start: {e}");
        std::process::exit(1);
    }

    // ── Start device watchdog for auto-reconnect ─────────────────
    let _watchdog = Pipeline::start_watchdog(Arc::clone(&pipeline));

    // ── Run mode ────────────────────────────────────────────────────
    if cli.headless {
        run_headless(&pipeline);
    } else if let Err(e) = tray::run(Arc::clone(&pipeline), config) {
        tracing::error!("Tray error: {e}");
        eprintln!("Tray error: {e}");
    }

    // Cleanup
    pipeline.stop();
    let final_config = pipeline.settings.to_config(&Config::load());
    if let Err(e) = final_config.save() {
        tracing::warn!("Failed to save config on exit: {e}");
    }
}

fn print_devices() {
    println!("=== Input Devices ===");
    match device::list_input_devices() {
        Ok(devices) => {
            for d in &devices {
                let marker = if d.is_default { " (default)" } else { "" };
                println!("  {}{}", d.name, marker);
            }
        }
        Err(e) => eprintln!("  Error: {e}"),
    }

    println!("\n=== Output Devices ===");
    match device::list_output_devices() {
        Ok(devices) => {
            for d in &devices {
                let marker = if d.is_default { " (default)" } else { "" };
                println!("  {}{}", d.name, marker);
            }
        }
        Err(e) => eprintln!("  Error: {e}"),
    }

    if let Some(virt) = device::detect_virtual_device() {
        println!("\n=== Virtual Device ===");
        println!("  {virt}");
    }
}

fn run_headless(pipeline: &Pipeline) {
    tracing::info!("Running headless. Press Ctrl+C to stop.");

    let running = Arc::new(std::sync::atomic::AtomicBool::new(true));
    let r = Arc::clone(&running);

    ctrlc_handler(move || {
        r.store(false, std::sync::atomic::Ordering::Relaxed);
    });

    while running.load(std::sync::atomic::Ordering::Relaxed) {
        std::thread::sleep(std::time::Duration::from_millis(100));
    }

    pipeline.stop();
}

fn ctrlc_handler<F: FnOnce() + Send + 'static>(f: F) {
    let f = std::sync::Mutex::new(Some(f));
    let _ = ctrlc::set_handler(move || {
        if let Some(f) = f.lock().unwrap().take() {
            f();
        }
    });
}
