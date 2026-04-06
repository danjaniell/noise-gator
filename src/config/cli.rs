use clap::Parser;

/// Noise Gator — lightweight real-time microphone noise cancellation.
#[derive(Parser, Debug)]
#[command(name = "noise-gator", version, about)]
pub struct Cli {
    /// Input microphone device name
    #[arg(short, long)]
    pub input: Option<String>,

    /// Monitoring output device name (hear yourself)
    #[arg(short, long)]
    pub monitor: Option<String>,

    /// Virtual output device name (for other apps). Auto-detected if omitted.
    #[arg(short, long)]
    pub virtual_device: Option<String>,

    /// Disable noise cancellation (passthrough mode)
    #[arg(long)]
    pub no_denoise: bool,

    /// Enable aggressive VAD-based noise gate
    #[arg(long)]
    pub hard_mode: bool,

    /// List available audio devices and exit
    #[arg(long)]
    pub list_devices: bool,

    /// Skip virtual audio driver check/install
    #[arg(long)]
    pub skip_driver: bool,

    /// Uninstall the virtual audio driver and exit
    #[arg(long)]
    pub uninstall_driver: bool,

    /// Path to config file (default: platform config dir)
    #[arg(long)]
    pub config: Option<String>,

    /// Run headless (no system tray, useful for scripts)
    #[arg(long)]
    pub headless: bool,
}
