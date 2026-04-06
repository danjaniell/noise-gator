use std::path::Path;
use std::process::Command;

use anyhow::{Result, anyhow};

const DRIVER_NAME: &str = "NoiseGatorDriver.driver";
const INSTALL_DIR: &str = "/Library/Audio/Plug-Ins/HAL";

/// BlackHole download URL (open-source virtual audio driver for macOS).
const BLACKHOLE_URL: &str =
    "https://github.com/ExistentialAudio/BlackHole/releases/latest/download/BlackHole2ch.pkg";

pub fn is_installed() -> bool {
    // Check for our own driver or BlackHole
    let own_driver = format!("{}/{}/Contents/Info.plist", INSTALL_DIR, DRIVER_NAME);
    if Path::new(&own_driver).exists() {
        return true;
    }
    // Also check for BlackHole
    Path::new("/Library/Audio/Plug-Ins/HAL/BlackHole2ch.driver/Contents/Info.plist").exists()
}

pub fn install() -> Result<()> {
    // Download BlackHole (open-source, no license issues)
    let tmp_pkg = "/tmp/BlackHole2ch.pkg";

    tracing::info!("Downloading BlackHole virtual audio driver...");
    let response = reqwest::blocking::get(BLACKHOLE_URL)
        .map_err(|e| anyhow!("Failed to download BlackHole: {e}"))?;

    if !response.status().is_success() {
        return Err(anyhow!("Download failed: {}", response.status()));
    }

    let bytes = response
        .bytes()
        .map_err(|e| anyhow!("Failed to read download: {e}"))?;
    std::fs::write(tmp_pkg, &bytes).map_err(|e| anyhow!("Failed to write pkg: {e}"))?;

    // Install with admin prompt
    let script = format!(
        r#"do shell script "installer -pkg '{}' -target /" with administrator privileges"#,
        tmp_pkg
    );

    let output = Command::new("osascript")
        .arg("-e")
        .arg(&script)
        .output()
        .map_err(|e| anyhow!("Failed to run installer: {e}"))?;

    let _ = std::fs::remove_file(tmp_pkg);

    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr);
        if stderr.contains("User canceled") || stderr.contains("-128") {
            return Err(anyhow!("Permission denied."));
        }
        return Err(anyhow!("Installation failed: {stderr}"));
    }

    tracing::info!("BlackHole installed. Restarting CoreAudio...");
    let _ = Command::new("killall").arg("coreaudiod").output();
    std::thread::sleep(std::time::Duration::from_secs(2));

    Ok(())
}

pub fn uninstall() -> Result<()> {
    let script = format!(
        r#"do shell script "rm -rf '{0}/{1}' && killall coreaudiod" with administrator privileges"#,
        INSTALL_DIR, DRIVER_NAME
    );

    let output = Command::new("osascript")
        .arg("-e")
        .arg(&script)
        .output()
        .map_err(|e| anyhow!("Failed to run osascript: {e}"))?;

    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr);
        if stderr.contains("User canceled") || stderr.contains("-128") {
            return Err(anyhow!("Permission denied."));
        }
        return Err(anyhow!("Uninstall failed: {stderr}"));
    }

    tracing::info!("Virtual audio driver uninstalled.");
    Ok(())
}
