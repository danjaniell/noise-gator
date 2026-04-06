//! Download and cache management for DeepFilterNet ONNX model files.

use std::path::PathBuf;

use anyhow::{anyhow, Result};

const DEEPFILTER_MODEL_URL: &str =
    "https://github.com/Rikorose/DeepFilterNet/raw/main/models/DeepFilterNet3_onnx.tar.gz";

const DEEPFILTER_SHA256: &str = "";

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

pub fn ensure_deepfilter_model() -> Result<PathBuf> {
    let model_dir = deepfilter_model_dir();

    if is_deepfilter_available() {
        tracing::info!("DeepFilterNet model already cached at {}", model_dir.display());
        return Ok(model_dir);
    }

    tracing::info!("Downloading DeepFilterNet model from {DEEPFILTER_MODEL_URL}");

    let response = reqwest::blocking::get(DEEPFILTER_MODEL_URL)
        .map_err(|e| anyhow!("Failed to download DeepFilterNet model: {e}"))?;

    if !response.status().is_success() {
        return Err(anyhow!("Model download failed with status: {}", response.status()));
    }

    let bytes = response
        .bytes()
        .map_err(|e| anyhow!("Failed to read model download: {e}"))?;

    if !DEEPFILTER_SHA256.is_empty() {
        verify_sha256(&bytes, DEEPFILTER_SHA256)?;
    } else {
        tracing::warn!("No SHA256 hash pinned for model — skipping verification");
    }

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

#[cfg(not(feature = "deepfilter"))]
fn verify_sha256(_data: &[u8], _expected: &str) -> Result<()> {
    Ok(())
}

fn extract_tar_gz(data: &[u8], dest: &PathBuf) -> Result<()> {
    let gz = flate2::read::GzDecoder::new(data);
    let mut archive = tar::Archive::new(gz);

    for entry in archive.entries()? {
        let mut entry = entry?;
        let path = entry.path()?;
        if let Some(file_name) = path.file_name() {
            let target = dest.join(file_name);
            let mut file = std::fs::File::create(&target)?;
            std::io::copy(&mut entry, &mut file)?;
        }
    }
    Ok(())
}
