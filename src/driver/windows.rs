use std::os::windows::process::CommandExt;
use std::process::Command;

use anyhow::{Result, anyhow};
use cpal::traits::{DeviceTrait, HostTrait};

const CREATE_NO_WINDOW: u32 = 0x0800_0000;

/// VB-Cable download URL (official VB-Audio site).
const VBCABLE_URL: &str = "https://download.vb-audio.com/Download_CABLE/VBCABLE_Driver_Pack45.zip";

pub fn is_installed() -> bool {
    let host = cpal::default_host();
    host.input_devices().ok().is_some_and(|mut devs| {
        devs.any(|d| {
            d.description()
                .map(|desc| {
                    let n = desc.name();
                    n.contains("NoiseGator") || n.contains("CABLE") || n.contains("VB-Audio")
                })
                .unwrap_or(false)
        })
    })
}

pub fn download_and_install() -> Result<()> {
    let tmp_dir = std::env::temp_dir().join("noise-gator-driver");
    let _ = std::fs::create_dir_all(&tmp_dir);
    let zip_path = tmp_dir.join("VBCABLE_Driver_Pack.zip");

    // Download
    tracing::info!("Downloading VB-Cable from {}", VBCABLE_URL);
    let response = reqwest::blocking::get(VBCABLE_URL)
        .map_err(|e| anyhow!("Failed to download VB-Cable: {e}"))?;

    if !response.status().is_success() {
        return Err(anyhow!(
            "Download failed with status: {}",
            response.status()
        ));
    }

    let bytes = response
        .bytes()
        .map_err(|e| anyhow!("Failed to read download: {e}"))?;
    std::fs::write(&zip_path, &bytes).map_err(|e| anyhow!("Failed to write zip: {e}"))?;

    // Extract using PowerShell
    let extract_cmd = format!(
        "Expand-Archive -Path '{}' -DestinationPath '{}' -Force",
        zip_path.display(),
        tmp_dir.display()
    );
    let extract_status = Command::new("powershell")
        .args(["-NoProfile", "-NonInteractive", "-Command", &extract_cmd])
        .creation_flags(CREATE_NO_WINDOW)
        .status()
        .map_err(|e| anyhow!("Failed to extract zip: {e}"))?;

    if !extract_status.success() {
        return Err(anyhow!("Failed to extract VB-Cable zip"));
    }

    // Find the x64 installer
    let setup_exe = tmp_dir.join("VBCABLE_Setup_x64.exe");
    if !setup_exe.exists() {
        return Err(anyhow!(
            "VBCABLE_Setup_x64.exe not found in extracted archive"
        ));
    }

    // Run installer (will show UAC prompt)
    tracing::info!("Launching VB-Cable installer...");
    let status = Command::new(&setup_exe)
        .creation_flags(CREATE_NO_WINDOW)
        .status()
        .map_err(|e| anyhow!("Failed to launch VB-Cable installer: {e}"))?;

    if !status.success() {
        let code = status.code().unwrap_or(-1);
        if code == 1 {
            return Err(anyhow!("Installation was cancelled by user."));
        }
        return Err(anyhow!("VB-Cable installer exited with code {code}"));
    }

    tracing::info!("VB-Cable installed. Waiting for Windows Audio to register...");
    std::thread::sleep(std::time::Duration::from_secs(5));

    // Rename CABLE → NoiseGator in the registry
    rename_device()?;

    // Cleanup
    let _ = std::fs::remove_dir_all(&tmp_dir);

    tracing::info!("Virtual audio driver installed and configured.");
    Ok(())
}

fn rename_device() -> Result<()> {
    let rename_ps1 = r"$ErrorActionPreference = 'SilentlyContinue'
Stop-Service -Name 'AudioSrv' -Force
Stop-Service -Name 'AudioEndpointBuilder' -Force
Start-Sleep -Seconds 1
$k = '{b3f8fa53-0004-438e-9003-51a46e139bfc},6'
$cap = 'HKLM:\SOFTWARE\Microsoft\Windows\CurrentVersion\MMDevices\Audio\Capture'
Get-ChildItem $cap | ForEach-Object {
    $p = Join-Path $_.PSPath 'Properties'
    $v = (Get-ItemProperty $p -Name $k -EA 0).$k
    if ($v -like '*CABLE*') { Set-ItemProperty $p $k 'NoiseGator' }
}
$ren = 'HKLM:\SOFTWARE\Microsoft\Windows\CurrentVersion\MMDevices\Audio\Render'
Get-ChildItem $ren | ForEach-Object {
    $p = Join-Path $_.PSPath 'Properties'
    $v = (Get-ItemProperty $p -Name $k -EA 0).$k
    if ($v -like '*CABLE*') { Set-ItemProperty $p $k 'NoiseGator [Internal]' }
}
Start-Service -Name 'AudioEndpointBuilder'
Start-Service -Name 'AudioSrv'
";

    let script_path = std::env::temp_dir().join("noisegator_rename.ps1");
    std::fs::write(&script_path, rename_ps1)
        .map_err(|e| anyhow!("Failed to write rename script: {e}"))?;

    let elevate = format!(
        "Start-Process powershell -Verb RunAs -WindowStyle Hidden -Wait \
         -ArgumentList '-NonInteractive -WindowStyle Hidden -ExecutionPolicy Bypass -File \"{}\"'",
        script_path.display()
    );

    let _ = Command::new("powershell")
        .args([
            "-NoProfile",
            "-NonInteractive",
            "-WindowStyle",
            "Hidden",
            "-Command",
            &elevate,
        ])
        .creation_flags(CREATE_NO_WINDOW)
        .output();

    let _ = std::fs::remove_file(&script_path);
    Ok(())
}

pub fn uninstall() -> Result<()> {
    let script = r#"
$ErrorActionPreference = 'SilentlyContinue'
$driverOutput = & pnputil /enum-drivers /class "Media" 2>&1
$lines = $driverOutput -split "`r?`n"
$currentInf = $null
foreach ($line in $lines) {
    if ($line -match 'Published Name\s*:\s*(\S+\.inf)') { $currentInf = $Matches[1] }
    if (($line -match 'VB-Audio' -or $line -match 'vbMme' -or $line -match 'VBCABLE') -and $currentInf) {
        & pnputil /delete-driver $currentInf /uninstall /force
        $currentInf = $null
    }
}
"#;

    let script_path = std::env::temp_dir().join("noisegator_uninstall.ps1");
    std::fs::write(&script_path, script)
        .map_err(|e| anyhow!("Failed to write uninstall script: {e}"))?;

    let elevate = format!(
        "Start-Process powershell -ArgumentList '-NoProfile -ExecutionPolicy Bypass -File \"{}\"' \
         -Verb RunAs -Wait",
        script_path.display()
    );

    tracing::info!("Uninstalling VB-Cable driver...");

    let output = Command::new("powershell")
        .args(["-NoProfile", "-NonInteractive", "-Command", &elevate])
        .output()
        .map_err(|e| anyhow!("Failed to launch uninstaller: {e}"))?;

    let _ = std::fs::remove_file(&script_path);

    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr);
        if stderr.contains("canceled") || stderr.contains("cancelled") {
            return Err(anyhow!("Permission denied. Driver was not removed."));
        }
        return Err(anyhow!("Driver uninstallation failed: {stderr}"));
    }

    tracing::info!("VB-Cable driver uninstalled.");
    Ok(())
}
