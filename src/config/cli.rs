/// Noise Gator — lightweight real-time microphone noise cancellation.
#[derive(Debug, Default)]
pub struct Cli {
    /// Input microphone device name
    pub input: Option<String>,
    /// Monitoring output device name (hear yourself)
    pub monitor: Option<String>,
    /// Virtual output device name (for other apps). Auto-detected if omitted.
    pub virtual_device: Option<String>,
    /// Disable noise cancellation (passthrough mode)
    pub no_denoise: bool,
    /// Enable aggressive VAD-based noise gate
    pub hard_mode: bool,
    /// List available audio devices and exit
    pub list_devices: bool,
    /// Skip virtual audio driver check/install
    pub skip_driver: bool,
    /// Uninstall the virtual audio driver and exit
    pub uninstall_driver: bool,
    /// Path to config file (default: platform config dir)
    pub config: Option<String>,
    /// Run headless (no system tray, useful for scripts)
    pub headless: bool,
    /// Denoise engine: rnnoise (default) or deepfilter
    pub engine: Option<String>,
}

impl Cli {
    pub fn parse() -> Self {
        let mut pargs = pico_args::Arguments::from_env();

        if pargs.contains(["-h", "--help"]) {
            print_help();
            std::process::exit(0);
        }

        if pargs.contains(["-V", "--version"]) {
            println!("{} {}", env!("CARGO_PKG_NAME"), env!("CARGO_PKG_VERSION"));
            std::process::exit(0);
        }

        let cli = Cli {
            input: pargs.opt_value_from_str(["-i", "--input"]).unwrap_or(None),
            monitor: pargs.opt_value_from_str(["-m", "--monitor"]).unwrap_or(None),
            virtual_device: pargs.opt_value_from_str(["-v", "--virtual-device"]).unwrap_or(None),
            no_denoise: pargs.contains("--no-denoise"),
            hard_mode: pargs.contains("--hard-mode"),
            list_devices: pargs.contains("--list-devices"),
            skip_driver: pargs.contains("--skip-driver"),
            uninstall_driver: pargs.contains("--uninstall-driver"),
            config: pargs.opt_value_from_str("--config").unwrap_or(None),
            headless: pargs.contains("--headless"),
            engine: pargs.opt_value_from_str("--engine").unwrap_or(None),
        };

        let remaining = pargs.finish();
        if !remaining.is_empty() {
            eprintln!("Error: unrecognized arguments: {remaining:?}");
            eprintln!("Run with --help for usage.");
            std::process::exit(1);
        }

        cli
    }
}

fn print_help() {
    println!(
        "\
{name} {version}
{desc}

USAGE:
    {name} [OPTIONS]

OPTIONS:
    -i, --input <NAME>           Input microphone device name
    -m, --monitor <NAME>         Monitoring output device name (hear yourself)
    -v, --virtual-device <NAME>  Virtual output device name (auto-detected if omitted)
        --no-denoise             Disable noise cancellation (passthrough mode)
        --hard-mode              Enable aggressive VAD-based noise gate
        --list-devices           List available audio devices and exit
        --skip-driver            Skip virtual audio driver check/install
        --uninstall-driver       Uninstall the virtual audio driver and exit
        --config <PATH>          Path to config file (default: platform config dir)
        --headless               Run headless (no system tray, useful for scripts)
        --engine <ENGINE>        Denoise engine: rnnoise (default) or deepfilter
    -h, --help                   Print help
    -V, --version                Print version",
        name = env!("CARGO_PKG_NAME"),
        version = env!("CARGO_PKG_VERSION"),
        desc = env!("CARGO_PKG_DESCRIPTION"),
    );
}
