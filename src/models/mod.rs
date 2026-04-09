//! Download and cache management for DeepFilterNet ONNX model files
//! and ONNX Runtime shared library.

use std::path::PathBuf;
#[cfg(feature = "deepfilter")]
use std::path::Path;

use anyhow::{Result, anyhow};

#[cfg(feature = "deepfilter")]
const DEEPFILTER_MODEL_URL: &str =
    "https://github.com/Rikorose/DeepFilterNet/raw/84d57ec2c08fe08e68a13fb32a58cd7092060a0f/models/DeepFilterNet3_onnx.tar.gz";

#[cfg(feature = "deepfilter")]
const DEEPFILTER_SHA256: &str =
    "c94d91f70911001c946e0fabb4aa9adc37045f45a03b56008cb0c8244cb63616";

const MODELS_DIR: &str = "models";

pub fn deepfilter_model_dir() -> PathBuf {
    model_base_dir().join("deepfilter3")
}

pub fn is_deepfilter_available() -> bool {
    let dir = deepfilter_model_dir();
    dir.join("enc.onnx").exists()
        && dir.join("erb_dec.onnx").exists()
        && dir.join("df_dec.onnx").exists()
        && dir.join("config.ini").exists()
}

#[cfg(feature = "deepfilter")]
pub fn ensure_deepfilter_model() -> Result<PathBuf> {
    let model_dir = deepfilter_model_dir();

    if is_deepfilter_available() {
        tracing::info!(
            "DeepFilterNet model already cached at {}",
            model_dir.display()
        );
        return Ok(model_dir);
    }

    tracing::info!("Downloading DeepFilterNet model from {DEEPFILTER_MODEL_URL}");

    let response = ureq::get(DEEPFILTER_MODEL_URL)
        .call()
        .map_err(|e| anyhow!("Failed to download DeepFilterNet model: {e}"))?;

    let bytes = response
        .into_body()
        .with_config()
        .limit(50 * 1024 * 1024) // 50 MB — model tar.gz is ~8 MB
        .read_to_vec()
        .map_err(|e| anyhow!("Failed to read model download: {e}"))?;

    verify_sha256(&bytes, DEEPFILTER_SHA256)?;

    std::fs::create_dir_all(&model_dir)?;
    extract_tar_gz(&bytes, &model_dir)?;

    if !is_deepfilter_available() {
        return Err(anyhow!(
            "Model extraction succeeded but expected files not found in {}",
            model_dir.display()
        ));
    }

    tracing::info!("DeepFilterNet model installed to {}", model_dir.display());
    Ok(model_dir)
}

fn model_base_dir() -> PathBuf {
    crate::config::Config::path()
        .parent()
        .map(|p| p.join(MODELS_DIR))
        .unwrap_or_else(|| PathBuf::from(MODELS_DIR))
}

#[cfg(feature = "deepfilter")]
fn verify_sha256(data: &[u8], expected: &str) -> Result<()> {
    use sha2::{Digest, Sha256};
    let hash = Sha256::digest(data);
    let hex = format!("{hash:x}");
    if hex != expected {
        return Err(anyhow!("SHA256 mismatch: expected {expected}, got {hex}"));
    }
    Ok(())
}


#[cfg(feature = "deepfilter")]
fn extract_tar_gz(data: &[u8], dest: &Path) -> Result<()> {
    let gz = flate2::read::GzDecoder::new(data);
    let mut archive = tar::Archive::new(gz);
    archive.set_preserve_permissions(false);

    for entry in archive.entries()? {
        let mut entry = entry?;
        // Only extract regular files — skip symlinks, hardlinks, directories
        if !entry.header().entry_type().is_file() {
            continue;
        }
        let path = entry.path()?;
        if let Some(file_name) = path.file_name() {
            let target = dest.join(file_name);
            let mut file = std::fs::File::create(&target)?;
            std::io::copy(&mut entry, &mut file)?;
        }
    }
    Ok(())
}

// ── ONNX Runtime shared library management ─────────────────────────────

/// Expected DLL/dylib filename per platform.
#[cfg(all(feature = "deepfilter", target_os = "windows"))]
const ORT_LIB_NAME: &str = "onnxruntime.dll";
#[cfg(all(feature = "deepfilter", target_os = "macos"))]
const ORT_LIB_NAME: &str = "libonnxruntime.dylib";
#[cfg(all(feature = "deepfilter", target_os = "linux"))]
const ORT_LIB_NAME: &str = "libonnxruntime.so";

/// Platform suffix used in ONNX Runtime release asset names.
#[cfg(all(feature = "deepfilter", target_os = "windows", target_arch = "x86_64"))]
const ORT_PLATFORM: &str = "win-x64";
#[cfg(all(feature = "deepfilter", target_os = "macos", target_arch = "x86_64"))]
const ORT_PLATFORM: &str = "osx-x86_64";
#[cfg(all(feature = "deepfilter", target_os = "macos", target_arch = "aarch64"))]
const ORT_PLATFORM: &str = "osx-arm64";
#[cfg(all(feature = "deepfilter", target_os = "linux", target_arch = "x86_64"))]
const ORT_PLATFORM: &str = "linux-x64";

/// Resolve the download URL for the latest ONNX Runtime release.
/// Queries the GitHub API to find the newest version and matching asset.
#[cfg(feature = "deepfilter")]
fn resolve_ort_download_url() -> Result<String> {
    let response = ureq::get("https://api.github.com/repos/microsoft/onnxruntime/releases/latest")
        .header("User-Agent", "noise-gator")
        .call()
        .map_err(|e| anyhow!("Failed to query ONNX Runtime releases: {e}"))?;

    let body = response
        .into_body()
        .read_to_string()
        .map_err(|e| anyhow!("Failed to read GitHub API response: {e}"))?;

    let resp: serde_json::Value = serde_json::from_str(&body)
        .map_err(|e| anyhow!("Failed to parse GitHub API response: {e}"))?;

    let needle = format!("onnxruntime-{ORT_PLATFORM}");
    let assets = resp["assets"]
        .as_array()
        .ok_or_else(|| anyhow!("GitHub API response missing 'assets' array"))?;

    for asset in assets {
        let name = asset["name"].as_str().unwrap_or("");
        let url = asset["browser_download_url"].as_str().unwrap_or("");

        if !name.contains(&needle) || name.contains("gpu") || name.contains("GPU") {
            continue;
        }
        if url.ends_with(".zip") || url.ends_with(".tgz") {
            tracing::info!("Resolved ONNX Runtime URL: {url}");
            return Ok(url.to_string());
        }
    }

    Err(anyhow!(
        "No ONNX Runtime asset found for platform '{ORT_PLATFORM}'"
    ))
}

/// Directory where ONNX Runtime library is cached.
#[cfg(feature = "deepfilter")]
fn ort_lib_dir() -> PathBuf {
    model_base_dir().join("ort")
}

/// Full path to the ONNX Runtime shared library.
#[cfg(feature = "deepfilter")]
pub fn ort_lib_path() -> PathBuf {
    ort_lib_dir().join(ORT_LIB_NAME)
}

/// Check if ONNX Runtime is already available.
#[cfg(feature = "deepfilter")]
pub fn is_ort_available() -> bool {
    ort_lib_path().exists()
}

/// Download and extract ONNX Runtime if not already cached.
#[cfg(feature = "deepfilter")]
pub fn ensure_onnxruntime() -> Result<PathBuf> {
    let lib_path = ort_lib_path();

    if lib_path.exists() {
        tracing::info!("ONNX Runtime already cached at {}", lib_path.display());
        return Ok(lib_path);
    }

    let url = resolve_ort_download_url()?;
    tracing::info!("Downloading ONNX Runtime from {url}");

    let response = ureq::get(&url)
        .call()
        .map_err(|e| anyhow!("Failed to download ONNX Runtime: {e}"))?;

    let bytes = response
        .into_body()
        .with_config()
        .limit(100 * 1024 * 1024) // 100 MB — ORT archive is ~15-20 MB
        .read_to_vec()
        .map_err(|e| anyhow!("Failed to read ONNX Runtime download: {e}"))?;

    let dest = ort_lib_dir();
    std::fs::create_dir_all(&dest)?;

    if url.ends_with(".zip") {
        extract_zip_lib(&bytes, &dest)?;
    } else {
        extract_tgz_lib(&bytes, &dest)?;
    }

    if !lib_path.exists() {
        return Err(anyhow!(
            "ONNX Runtime extraction succeeded but {} not found",
            ORT_LIB_NAME
        ));
    }

    tracing::info!("ONNX Runtime installed to {}", lib_path.display());
    Ok(lib_path)
}

/// Extract the shared library from the ONNX Runtime zip archive.
/// The zip contains a directory like `onnxruntime-win-x64-1.21.1/lib/onnxruntime.dll`.
#[cfg(feature = "deepfilter")]
fn extract_zip_lib(data: &[u8], dest: &Path) -> Result<()> {
    use std::io::Cursor;

    let reader = Cursor::new(data);
    let mut archive =
        zip::ZipArchive::new(reader).map_err(|e| anyhow!("Failed to open zip: {e}"))?;

    for i in 0..archive.len() {
        let mut file = archive
            .by_index(i)
            .map_err(|e| anyhow!("Failed to read zip entry: {e}"))?;

        let name = file.name().to_string();

        // Look for the shared library in the lib/ directory
        if name.ends_with(ORT_LIB_NAME) {
            let target = dest.join(ORT_LIB_NAME);
            let mut out = std::fs::File::create(&target)?;
            std::io::copy(&mut file, &mut out)?;
            tracing::debug!("Extracted {name} → {}", target.display());
            return Ok(());
        }
    }

    Err(anyhow!(
        "{} not found in ONNX Runtime zip archive",
        ORT_LIB_NAME
    ))
}

/// Extract the shared library from an ONNX Runtime .tgz archive (macOS/Linux).
#[cfg(feature = "deepfilter")]
fn extract_tgz_lib(data: &[u8], dest: &Path) -> Result<()> {
    let gz = flate2::read::GzDecoder::new(data);
    let mut archive = tar::Archive::new(gz);

    for entry in archive.entries()? {
        let mut entry = entry?;
        let is_match = entry
            .path()?
            .to_string_lossy()
            .ends_with(ORT_LIB_NAME);

        if is_match {
            let target = dest.join(ORT_LIB_NAME);
            let mut out = std::fs::File::create(&target)?;
            std::io::copy(&mut entry, &mut out)?;
            tracing::debug!("Extracted {} → {}", ORT_LIB_NAME, target.display());
            return Ok(());
        }
    }

    Err(anyhow!(
        "{} not found in ONNX Runtime tgz archive",
        ORT_LIB_NAME
    ))
}

/// Initialize the ort crate with our cached ONNX Runtime library.
/// Must be called before any ort Session is created.
/// Idempotent — safe to call multiple times (only the first call initializes).
#[cfg(feature = "deepfilter")]
pub fn init_ort() -> Result<()> {
    use std::sync::OnceLock;
    static ORT_INIT: OnceLock<Result<(), String>> = OnceLock::new();

    let result = ORT_INIT.get_or_init(|| {
        let lib_path = ensure_onnxruntime().map_err(|e| e.to_string())?;
        ort::init_from(lib_path.to_string_lossy().to_string())
            .commit()
            .map_err(|e| e.to_string())?;
        tracing::info!("ONNX Runtime initialized");
        Ok(())
    });

    result.as_ref().map_err(|e| anyhow!("{e}")).copied()
}

#[cfg(not(feature = "deepfilter"))]
pub fn init_ort() -> Result<()> {
    Ok(())
}

#[cfg(not(feature = "deepfilter"))]
pub fn ensure_deepfilter_model() -> Result<PathBuf> {
    Err(anyhow!("DeepFilterNet support not compiled in. Rebuild with --features deepfilter"))
}

#[cfg(not(feature = "deepfilter"))]
pub fn is_ort_available() -> bool {
    false
}
