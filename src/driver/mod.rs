#[cfg(target_os = "windows")]
mod windows;

#[cfg(target_os = "macos")]
mod macos;

use anyhow::Result;

/// Check if a virtual audio driver is installed on this system.
pub fn is_installed() -> bool {
    #[cfg(target_os = "windows")]
    {
        windows::is_installed()
    }
    #[cfg(target_os = "macos")]
    {
        macos::is_installed()
    }
    #[cfg(not(any(target_os = "macos", target_os = "windows")))]
    {
        false
    }
}

/// Download and install the virtual audio driver.
/// On Windows: downloads VB-Cable from official site and runs the installer.
/// On macOS: installs a bundled HAL plugin.
pub fn ensure_installed() -> Result<()> {
    if is_installed() {
        tracing::info!("Virtual audio driver already installed");
        return Ok(());
    }

    tracing::info!("Virtual audio driver not found — installing...");

    #[cfg(target_os = "windows")]
    {
        windows::download_and_install()
    }
    #[cfg(target_os = "macos")]
    {
        macos::install()
    }
    #[cfg(not(any(target_os = "macos", target_os = "windows")))]
    {
        Err(anyhow::anyhow!(
            "Virtual audio driver auto-install is not supported on this platform. \
             Please install a virtual audio loopback device manually (e.g., PulseAudio null sink)."
        ))
    }
}

/// Uninstall the virtual audio driver.
pub fn uninstall() -> Result<()> {
    #[cfg(target_os = "windows")]
    {
        windows::uninstall()
    }
    #[cfg(target_os = "macos")]
    {
        macos::uninstall()
    }
    #[cfg(not(any(target_os = "macos", target_os = "windows")))]
    {
        Err(anyhow::anyhow!(
            "Driver uninstallation is not supported on this platform"
        ))
    }
}
