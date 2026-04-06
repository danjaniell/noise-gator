pub mod cli;

use std::path::PathBuf;
use std::sync::atomic::{AtomicBool, AtomicU32, Ordering};

use anyhow::Result;
use serde::{Deserialize, Serialize};

use crate::dsp::autogain::AutoGainSettings;
use crate::dsp::eq::EqSettings;
use crate::dsp::gate::GateSettings;

/// Available noise suppression engines.
#[derive(Debug, Default, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum DenoiseEngine {
    /// RNNoise via nnnoiseless — lightweight, always available, has neural VAD.
    #[default]
    RNNoise,
    /// DeepFilterNet via tract-onnx — higher quality, requires model download.
    DeepFilter,
}

impl std::fmt::Display for DenoiseEngine {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::RNNoise => write!(f, "RNNoise"),
            Self::DeepFilter => write!(f, "DeepFilter"),
        }
    }
}

/// Persisted configuration (TOML file).
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct Config {
    pub input_device: Option<String>,
    pub output_device: Option<String>,
    pub virtual_device: Option<String>,
    pub denoise_enabled: bool,
    pub hard_mode: bool,
    pub engine: DenoiseEngine,
    pub input_gain: f32,
    pub output_gain: f32,
    pub eq: EqSettings,
    pub gate: GateSettings,
    pub autogain: AutoGainSettings,
}

impl Default for Config {
    fn default() -> Self {
        Self {
            input_device: None,
            output_device: None,
            virtual_device: None,
            denoise_enabled: true,
            hard_mode: false,
            engine: DenoiseEngine::default(),
            input_gain: 1.0,
            output_gain: 1.0,
            eq: EqSettings::default(),
            gate: GateSettings::default(),
            autogain: AutoGainSettings::default(),
        }
    }
}

impl Config {
    /// Load config from the default path, or return defaults if missing.
    pub fn load() -> Self {
        match Self::load_from(Self::path()) {
            Ok(cfg) => cfg,
            Err(e) => {
                tracing::debug!("No config file or parse error, using defaults: {e}");
                Self::default()
            }
        }
    }

    /// Load config from a specific path.
    pub fn load_from_path(path: &str) -> Result<Self> {
        Self::load_from(PathBuf::from(path))
    }

    fn load_from(path: PathBuf) -> Result<Self> {
        let content = std::fs::read_to_string(&path)?;
        let cfg: Config = toml::from_str(&content)?;
        Ok(cfg)
    }

    /// Save config to disk.
    pub fn save(&self) -> Result<()> {
        let path = Self::path();
        if let Some(parent) = path.parent() {
            std::fs::create_dir_all(parent)?;
        }
        let content = toml::to_string_pretty(self)?;
        std::fs::write(&path, content)?;
        tracing::debug!("Config saved to {}", path.display());
        Ok(())
    }

    /// Config file path: `~/.config/noise-gator/config.toml` (or platform equivalent).
    pub fn path() -> PathBuf {
        dirs_path()
            .unwrap_or_else(|| PathBuf::from("."))
            .join("config.toml")
    }
}

/// Platform-appropriate config directory.
fn dirs_path() -> Option<PathBuf> {
    #[cfg(target_os = "windows")]
    {
        std::env::var("APPDATA")
            .ok()
            .map(|p| PathBuf::from(p).join("noise-gator"))
    }
    #[cfg(target_os = "macos")]
    {
        std::env::var("HOME")
            .ok()
            .map(|h| PathBuf::from(h).join("Library/Application Support/noise-gator"))
    }
    #[cfg(target_os = "linux")]
    {
        std::env::var("XDG_CONFIG_HOME")
            .ok()
            .or_else(|| std::env::var("HOME").ok().map(|h| format!("{h}/.config")))
            .map(|p| PathBuf::from(p).join("noise-gator"))
    }
}

/// Atomic runtime settings shared between the audio thread and tray UI.
/// These are the "live" values that the pipeline reads on every callback.
pub struct RuntimeSettings {
    pub denoise_enabled: AtomicBool,
    pub hard_mode: AtomicBool,
    pub input_gain: AtomicU32,
    pub output_gain: AtomicU32,
    pub eq_bass_db10: std::sync::atomic::AtomicI32,
    pub eq_mid_db10: std::sync::atomic::AtomicI32,
    pub eq_treble_db10: std::sync::atomic::AtomicI32,
    pub eq_enabled: AtomicBool,
    // Gate settings (f32 stored as AtomicU32 bit patterns)
    pub gate_threshold: AtomicU32,
    pub gate_attack_ms: AtomicU32,
    pub gate_release_ms: AtomicU32,
    pub gate_hold_ms: AtomicU32,
    pub gate_floor: AtomicU32,
    // Auto-gain settings
    pub autogain_enabled: AtomicBool,
    pub autogain_target_rms: AtomicU32,
    pub autogain_max_gain: AtomicU32,
    // Engine selection
    pub engine: std::sync::atomic::AtomicU8,
}

impl RuntimeSettings {
    pub fn from_config(cfg: &Config) -> Self {
        Self {
            denoise_enabled: AtomicBool::new(cfg.denoise_enabled),
            hard_mode: AtomicBool::new(cfg.hard_mode),
            input_gain: AtomicU32::new(cfg.input_gain.to_bits()),
            output_gain: AtomicU32::new(cfg.output_gain.to_bits()),
            eq_bass_db10: std::sync::atomic::AtomicI32::new((cfg.eq.bass_db * 10.0) as i32),
            eq_mid_db10: std::sync::atomic::AtomicI32::new((cfg.eq.mid_db * 10.0) as i32),
            eq_treble_db10: std::sync::atomic::AtomicI32::new((cfg.eq.treble_db * 10.0) as i32),
            eq_enabled: AtomicBool::new(cfg.eq.enabled),
            gate_threshold: AtomicU32::new(cfg.gate.threshold.to_bits()),
            gate_attack_ms: AtomicU32::new(cfg.gate.attack_ms.to_bits()),
            gate_release_ms: AtomicU32::new(cfg.gate.release_ms.to_bits()),
            gate_hold_ms: AtomicU32::new(cfg.gate.hold_ms.to_bits()),
            gate_floor: AtomicU32::new(cfg.gate.floor.to_bits()),
            autogain_enabled: AtomicBool::new(cfg.autogain.enabled),
            autogain_target_rms: AtomicU32::new(cfg.autogain.target_rms.to_bits()),
            autogain_max_gain: AtomicU32::new(cfg.autogain.max_gain.to_bits()),
            engine: std::sync::atomic::AtomicU8::new(cfg.engine as u8),
        }
    }

    pub fn load_engine(&self) -> DenoiseEngine {
        match self.engine.load(Ordering::Relaxed) {
            1 => DenoiseEngine::DeepFilter,
            _ => DenoiseEngine::RNNoise,
        }
    }

    /// Read current gate settings from atomics.
    pub fn load_gate_settings(&self) -> GateSettings {
        GateSettings {
            threshold: f32::from_bits(self.gate_threshold.load(Ordering::Relaxed)),
            attack_ms: f32::from_bits(self.gate_attack_ms.load(Ordering::Relaxed)),
            release_ms: f32::from_bits(self.gate_release_ms.load(Ordering::Relaxed)),
            hold_ms: f32::from_bits(self.gate_hold_ms.load(Ordering::Relaxed)),
            floor: f32::from_bits(self.gate_floor.load(Ordering::Relaxed)),
        }
    }

    /// Read current autogain settings from atomics.
    pub fn load_autogain_settings(&self) -> AutoGainSettings {
        let mut s = AutoGainSettings::default();
        s.enabled = self.autogain_enabled.load(Ordering::Relaxed);
        s.target_rms = f32::from_bits(self.autogain_target_rms.load(Ordering::Relaxed));
        s.max_gain = f32::from_bits(self.autogain_max_gain.load(Ordering::Relaxed));
        s
    }

    /// Read current EQ settings from atomics.
    pub fn load_eq_settings(&self) -> EqSettings {
        EqSettings {
            bass_db: self.eq_bass_db10.load(Ordering::Relaxed) as f32 / 10.0,
            mid_db: self.eq_mid_db10.load(Ordering::Relaxed) as f32 / 10.0,
            treble_db: self.eq_treble_db10.load(Ordering::Relaxed) as f32 / 10.0,
            enabled: self.eq_enabled.load(Ordering::Relaxed),
        }
    }

    /// Snapshot current runtime state back into a Config for saving.
    pub fn to_config(&self, base: &Config) -> Config {
        Config {
            input_device: base.input_device.clone(),
            output_device: base.output_device.clone(),
            virtual_device: base.virtual_device.clone(),
            denoise_enabled: self.denoise_enabled.load(Ordering::Relaxed),
            hard_mode: self.hard_mode.load(Ordering::Relaxed),
            engine: self.load_engine(),
            input_gain: f32::from_bits(self.input_gain.load(Ordering::Relaxed)),
            output_gain: f32::from_bits(self.output_gain.load(Ordering::Relaxed)),
            eq: self.load_eq_settings(),
            gate: self.load_gate_settings(),
            autogain: self.load_autogain_settings(),
        }
    }
}
