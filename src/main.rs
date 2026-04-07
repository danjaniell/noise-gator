#![cfg_attr(
    all(not(debug_assertions), target_os = "windows"),
    windows_subsystem = "windows"
)]

mod audio;
mod config;
mod driver;
mod dsp;
mod models;
mod tray;
mod ui;

use std::sync::Arc;

use clap::Parser;

use audio::device;
use audio::pipeline::Pipeline;
use config::cli::Cli;
use config::{Config, DenoiseEngine, RuntimeSettings};

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

    // ── Single instance guard ──────────────────────────────────────
    let _instance_guard = enforce_single_instance();

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
    if let Some(ref eng) = cli.engine {
        match eng.to_lowercase().as_str() {
            "deepfilter" | "deep-filter" | "df" => config.engine = DenoiseEngine::DeepFilter,
            "rnnoise" | "rnn" => config.engine = DenoiseEngine::RNNoise,
            other => tracing::warn!("Unknown engine '{other}', using default"),
        }
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

    // ── Initialize ONNX Runtime (for DeepFilter) ─────────────────────
    if config.engine == DenoiseEngine::DeepFilter {
        if let Err(e) = models::init_ort() {
            tracing::error!("Failed to initialize ONNX Runtime: {e}");
            tracing::warn!("Falling back to RNNoise engine");
            config.engine = DenoiseEngine::RNNoise;
        }
    }

    // ── Build pipeline ──────────────────────────────────────────────
    let settings = Arc::new(RuntimeSettings::from_config(&config));
    let pipeline = Arc::new(Pipeline::new(Arc::clone(&settings)));

    // Auto-start the pipeline — fall back to default devices if configured ones
    // are unavailable (e.g., Bluetooth headset not connected).
    if let Err(e) = pipeline.start(
        config.input_device.as_deref(),
        config.output_device.as_deref(),
        config.virtual_device.as_deref(),
    ) {
        tracing::warn!("Failed to start with configured devices: {e}");
        tracing::info!("Falling back to default devices");
        config.input_device = None;
        config.output_device = None;
        if let Err(e2) = pipeline.start(None, None, config.virtual_device.as_deref()) {
            tracing::error!("Failed to start audio pipeline: {e2}");
            eprintln!("Failed to start: {e2}");
            std::process::exit(1);
        }
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

/// Ensures only one instance of noise-gator runs at a time.
/// Returns a guard that holds the OS mutex — when dropped (or process exits),
/// the mutex is released so a new instance can start.
#[cfg(target_os = "windows")]
fn enforce_single_instance() -> windows::Win32::Foundation::HANDLE {
    use windows::Win32::System::Threading::CreateMutexW;
    use windows::core::HSTRING;

    let name = HSTRING::from("Global\\NoiseGator_SingleInstance");
    let handle = unsafe { CreateMutexW(None, true, &name) };

    match handle {
        Ok(h) => {
            // If ERROR_ALREADY_EXISTS, another instance owns the mutex
            let last_err = unsafe { windows::Win32::Foundation::GetLastError() };
            if last_err == windows::Win32::Foundation::ERROR_ALREADY_EXISTS {
                eprintln!("Noise Gator is already running.");
                std::process::exit(0);
            }
            h
        }
        Err(_) => {
            // CreateMutex failed — another instance likely running, or access denied
            eprintln!("Noise Gator is already running.");
            std::process::exit(0);
        }
    }
}

#[cfg(not(target_os = "windows"))]
fn enforce_single_instance() -> std::fs::File {
    use std::fs::File;
    use std::io::Write;
    use std::os::unix::io::AsRawFd;

    let lock_path = std::env::var("XDG_RUNTIME_DIR")
        .unwrap_or_else(|_| "/tmp".to_string())
        + "/noise-gator.lock";

    // Open-or-create the lock file (not truncated — flock is on the fd, not content)
    let file = File::options()
        .write(true)
        .create(true)
        .truncate(false)
        .open(&lock_path)
        .unwrap_or_else(|e| {
            eprintln!("Cannot open lock file {lock_path}: {e}");
            std::process::exit(1);
        });

    // Non-blocking exclusive lock — fails immediately if another instance holds it
    let rc = unsafe { libc::flock(file.as_raw_fd(), libc::LOCK_EX | libc::LOCK_NB) };
    if rc != 0 {
        eprintln!("Noise Gator is already running.");
        std::process::exit(0);
    }

    // Write PID for diagnostics (optional, doesn't affect locking)
    let mut f = &file;
    let _ = f.write_all(format!("{}", std::process::id()).as_bytes());

    // Return the File — the lock is held as long as this fd is open.
    // When the process exits (even on crash/SIGKILL), the OS releases the flock.
    file
}
