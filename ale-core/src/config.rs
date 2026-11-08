use crate::{AleError, Result};
use serde::{Deserialize, Serialize};
use std::path::{Path, PathBuf};

/// 云端API配置
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct CloudApiConfig {
    pub provider: String,
    pub api_key: String,
    pub api_url: String,
    pub model: String,
    pub max_tokens: usize,
    pub timeout: u32,
}

impl Default for CloudApiConfig {
    fn default() -> Self {
        Self {
            provider: "openai".to_string(),
            api_key: String::new(),
            api_url: "https://api.openai.com/v1".to_string(),
            model: "gpt-4o".to_string(),
            max_tokens: 1024,
            timeout: 30,
        }
    }
}

/// 模型配置
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct ModelsConfig {
    pub auto_download: bool,
    pub max_download_size: u64,
    pub preferred_quality: String,
    pub offline_mode: bool,
    pub models_dir: String,
}

impl Default for ModelsConfig {
    fn default() -> Self {
        Self {
            auto_download: true,
            max_download_size: 500 * 1024 * 1024, // 500MB
            preferred_quality: "balanced".to_string(),
            offline_mode: false,
            models_dir: "models".to_string(),
        }
    }
}

/// 推理配置
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct InferenceConfig {
    pub mode: String, // "local", "cloud", "adaptive"
    pub prefer_cloud: bool,
    pub timeout: u32,
    pub fallback_to_local: bool,
}

impl Default for InferenceConfig {
    fn default() -> Self {
        Self {
            mode: "adaptive".to_string(),
            prefer_cloud: true,
            timeout: 30,
            fallback_to_local: true,
        }
    }
}

/// 音频配置
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct AudioConfig {
    pub sample_rate: u32,
    pub channels: u16,
    pub buffer_size: u32,
    pub voice: String,
    pub speed: f32,
}

impl Default for AudioConfig {
    fn default() -> Self {
        Self {
            sample_rate: 16000,
            channels: 1,
            buffer_size: 4096,
            voice: "default".to_string(),
            speed: 1.0,
        }
    }
}

/// 界面配置
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct UiConfig {
    pub language: String,
    pub theme: String,
    pub font_size: u32,
    pub high_contrast: bool,
    pub screen_reader: bool,
    pub auto_speak: bool,
}

impl Default for UiConfig {
    fn default() -> Self {
        Self {
            language: "zh-CN".to_string(),
            theme: "system".to_string(),
            font_size: 16,
            high_contrast: false,
            screen_reader: true,
            auto_speak: true,
        }
    }
}

/// 应用配置
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
#[serde(default)]
pub struct AppConfig {
    pub cloud_api: CloudApiConfig,
    pub models: ModelsConfig,
    pub inference: InferenceConfig,
    pub audio: AudioConfig,
    pub ui: UiConfig,
}

/// 配置管理器
pub struct ConfigManager {
    config_path: PathBuf,
    config: AppConfig,
}

impl ConfigManager {
    pub fn new(config_path: &Path) -> Self {
        Self {
            config_path: config_path.to_path_buf(),
            config: AppConfig::default(),
        }
    }

    /// 加载配置
    pub fn load(&mut self) -> Result<()> {
        if !self.config_path.exists() {
            // 如果配置文件不存在，创建默认配置
            self.save()?;
            return Ok(());
        }

        let content = std::fs::read_to_string(&self.config_path)?;
        self.config = serde_json::from_str(&content)?;
        let original: serde_json::Value = serde_json::from_str(&content)?;
        let loaded = serde_json::to_value(&self.config)?;
        if original != loaded {
            self.save()?;
        }
        Ok(())
    }

    /// 保存配置
    pub fn save(&self) -> Result<()> {
        // 确保目录存在
        if let Some(parent) = self.config_path.parent() {
            std::fs::create_dir_all(parent)?;
        }

        let content = serde_json::to_string_pretty(&self.config)?;
        std::fs::write(&self.config_path, content)?;
        Ok(())
    }

    /// 获取配置
    pub fn config(&self) -> &AppConfig {
        &self.config
    }

    /// 更新配置
    pub fn update_config(&mut self, config: AppConfig) {
        self.config = config;
    }

    /// 更新云端API配置
    pub fn update_cloud_api(&mut self, config: CloudApiConfig) {
        self.config.cloud_api = config;
    }

    /// 更新模型配置
    pub fn update_models(&mut self, config: ModelsConfig) {
        self.config.models = config;
    }

    /// 更新推理配置
    pub fn update_inference(&mut self, config: InferenceConfig) {
        self.config.inference = config;
    }

    /// 更新音频配置
    pub fn update_audio(&mut self, config: AudioConfig) {
        self.config.audio = config;
    }

    /// 更新界面配置
    pub fn update_ui(&mut self, config: UiConfig) {
        self.config.ui = config;
    }

    /// 重置为默认配置
    pub fn reset_to_default(&mut self) {
        self.config = AppConfig::default();
    }

    /// 验证配置
    pub fn validate(&self) -> Result<()> {
        // 验证云端API配置
        if self.config.cloud_api.api_key.is_empty() {
            return Err(AleError::ConfigError("API key is required".to_string()));
        }

        // 验证模型配置
        if self.config.models.max_download_size == 0 {
            return Err(AleError::ConfigError(
                "Max download size must be greater than 0".to_string(),
            ));
        }

        // 验证推理配置
        let valid_modes = ["local", "cloud", "adaptive"];
        if !valid_modes.contains(&self.config.inference.mode.as_str()) {
            return Err(AleError::ConfigError(format!(
                "Invalid inference mode: {}. Must be one of: {:?}",
                self.config.inference.mode, valid_modes
            )));
        }

        Ok(())
    }

    /// 获取配置文件路径
    pub fn config_path(&self) -> &Path {
        &self.config_path
    }
}

/// 配置工厂
pub struct ConfigFactory;

impl ConfigFactory {
    /// 创建默认配置管理器
    pub fn create_default() -> ConfigManager {
        let config_dir = dirs::config_dir()
            .unwrap_or_else(|| PathBuf::from("."))
            .join("ale-my-eyes");

        let config_path = config_dir.join("config.json");
        ConfigManager::new(&config_path)
    }

    /// 创建指定路径的配置管理器
    pub fn create_with_path(config_path: &Path) -> ConfigManager {
        ConfigManager::new(config_path)
    }

    /// 创建测试配置
    pub fn create_test() -> ConfigManager {
        let config_path = PathBuf::from("/tmp/ale-my-eyes-test/config.json");
        ConfigManager::new(&config_path)
    }
}

/// 配置迁移器
pub struct ConfigMigrator;

impl ConfigMigrator {
    /// 迁移旧版本配置
    pub fn migrate(config_path: &Path) -> Result<()> {
        if !config_path.exists() {
            return Ok(());
        }

        let content = std::fs::read_to_string(config_path)?;
        let old_config: serde_json::Value = serde_json::from_str(&content)?;

        // 检查版本
        let version = old_config
            .get("version")
            .and_then(|v| v.as_str())
            .unwrap_or("1.0");

        match version {
            "1.0" => {
                // 从 1.0 迁移到 2.0
                Self::migrate_v1_to_v2(config_path, &old_config)?;
            }
            "2.0" => {
                // 已经是最新版本
            }
            _ => {
                return Err(AleError::ConfigError(format!(
                    "Unknown config version: {}",
                    version
                )));
            }
        }

        Ok(())
    }

    /// 从 v1.0 迁移到 v2.0
    fn migrate_v1_to_v2(config_path: &Path, old_config: &serde_json::Value) -> Result<()> {
        // 创建新的配置结构
        let mut new_config = AppConfig::default();

        // 迁移云端API配置
        if let Some(cloud_api) = old_config.get("cloud_api") {
            if let Some(provider) = cloud_api.get("provider").and_then(|v| v.as_str()) {
                new_config.cloud_api.provider = provider.to_string();
            }
            if let Some(api_key) = cloud_api.get("api_key").and_then(|v| v.as_str()) {
                new_config.cloud_api.api_key = api_key.to_string();
            }
        }

        // 迁移模型配置
        if let Some(models) = old_config.get("models") {
            if let Some(auto_download) = models.get("auto_download").and_then(|v| v.as_bool()) {
                new_config.models.auto_download = auto_download;
            }
        }

        // 保存新配置
        let content = serde_json::to_string_pretty(&new_config)?;
        std::fs::write(config_path, content)?;

        Ok(())
    }
}

/// 配置验证器
pub struct ConfigValidator;

impl ConfigValidator {
    /// 验证云端API配置
    pub fn validate_cloud_api(config: &CloudApiConfig) -> Result<()> {
        if config.api_key.is_empty() {
            return Err(AleError::ConfigError("API key is required".to_string()));
        }

        if config.max_tokens == 0 {
            return Err(AleError::ConfigError(
                "Max tokens must be greater than 0".to_string(),
            ));
        }

        if config.api_url.is_empty() {
            return Err(AleError::ConfigError("API URL is required".to_string()));
        }

        if !config.api_url.starts_with("http://") && !config.api_url.starts_with("https://") {
            return Err(AleError::ConfigError(
                "API URL must start with http:// or https://".to_string(),
            ));
        }

        if config.model.is_empty() {
            return Err(AleError::ConfigError("Model name is required".to_string()));
        }

        if config.timeout == 0 {
            return Err(AleError::ConfigError(
                "Timeout must be greater than 0".to_string(),
            ));
        }

        Ok(())
    }

    /// 验证模型配置
    pub fn validate_models(config: &ModelsConfig) -> Result<()> {
        if config.max_download_size == 0 {
            return Err(AleError::ConfigError(
                "Max download size must be greater than 0".to_string(),
            ));
        }

        let valid_qualities = ["low", "balanced", "high"];
        if !valid_qualities.contains(&config.preferred_quality.as_str()) {
            return Err(AleError::ConfigError(format!(
                "Invalid preferred quality: {}. Must be one of: {:?}",
                config.preferred_quality, valid_qualities
            )));
        }

        Ok(())
    }

    /// 验证推理配置
    pub fn validate_inference(config: &InferenceConfig) -> Result<()> {
        let valid_modes = ["local", "cloud", "adaptive"];
        if !valid_modes.contains(&config.mode.as_str()) {
            return Err(AleError::ConfigError(format!(
                "Invalid inference mode: {}. Must be one of: {:?}",
                config.mode, valid_modes
            )));
        }

        Ok(())
    }

    /// 验证完整配置
    pub fn validate_all(config: &AppConfig) -> Result<()> {
        if config.inference.mode == "local" {
            let mut cloud_api = config.cloud_api.clone();
            cloud_api.api_key = "local-mode".to_string();
            Self::validate_cloud_api(&cloud_api)?;
        } else {
            Self::validate_cloud_api(&config.cloud_api)?;
        }
        Self::validate_models(&config.models)?;
        Self::validate_inference(&config.inference)?;

        if config.ui.font_size == 0 {
            return Err(AleError::ConfigError(
                "Font size must be greater than 0".to_string(),
            ));
        }

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_default_config() {
        let config = AppConfig::default();
        assert_eq!(config.cloud_api.provider, "openai");
        assert_eq!(config.cloud_api.api_url, "https://api.openai.com/v1");
        assert_eq!(config.cloud_api.model, "gpt-4o");
        assert_eq!(config.cloud_api.max_tokens, 1024);
        assert_eq!(config.cloud_api.timeout, 30);
        assert_eq!(config.ui.language, "zh-CN");
        assert_eq!(config.ui.font_size, 16);
        assert!(!config.ui.high_contrast);
    }

    #[test]
    fn test_config_serialization_roundtrip() {
        let config = AppConfig::default();
        let json = serde_json::to_string_pretty(&config).unwrap();
        let restored: AppConfig = serde_json::from_str(&json).unwrap();
        assert_eq!(restored.cloud_api.provider, config.cloud_api.provider);
        assert_eq!(restored.cloud_api.api_url, config.cloud_api.api_url);
        assert_eq!(restored.ui.language, config.ui.language);
    }

    #[test]
    fn test_validate_cloud_api_missing_key() {
        let config = CloudApiConfig {
            api_key: String::new(),
            ..Default::default()
        };
        assert!(ConfigValidator::validate_cloud_api(&config).is_err());
    }

    #[test]
    fn test_validate_cloud_api_bad_url() {
        let config = CloudApiConfig {
            api_key: "sk-test".to_string(),
            api_url: "not-a-url".to_string(),
            ..Default::default()
        };
        assert!(ConfigValidator::validate_cloud_api(&config).is_err());
    }

    #[test]
    fn test_validate_cloud_api_valid() {
        let config = CloudApiConfig {
            api_key: "sk-test".to_string(),
            api_url: "https://api.openai.com/v1".to_string(),
            model: "gpt-4o".to_string(),
            ..Default::default()
        };
        assert!(ConfigValidator::validate_cloud_api(&config).is_ok());
    }

    #[test]
    fn test_validate_inference_invalid_mode() {
        let config = InferenceConfig {
            mode: "invalid".to_string(),
            ..Default::default()
        };
        assert!(ConfigValidator::validate_inference(&config).is_err());
    }

    #[test]
    fn test_validate_all_valid() {
        let config = AppConfig::default();
        let mut config = config;
        config.cloud_api.api_key = "sk-test".to_string();
        assert!(ConfigValidator::validate_all(&config).is_ok());
    }

    #[test]
    fn test_validate_all_local_mode_without_api_key() {
        let mut config = AppConfig::default();
        config.inference.mode = "local".to_string();
        config.cloud_api.api_key.clear();

        assert!(ConfigValidator::validate_all(&config).is_ok());
    }

    #[test]
    fn test_validate_all_local_mode_rejects_bad_api_url() {
        let mut config = AppConfig::default();
        config.inference.mode = "local".to_string();
        config.cloud_api.api_key.clear();
        config.cloud_api.api_url = "not-a-url".to_string();

        let error = ConfigValidator::validate_all(&config).unwrap_err();
        assert_eq!(
            error.to_string(),
            "Config error: API URL must start with http:// or https://"
        );
    }

    #[test]
    fn test_validate_all_local_mode_rejects_zero_timeout() {
        let mut config = AppConfig::default();
        config.inference.mode = "local".to_string();
        config.cloud_api.api_key.clear();
        config.cloud_api.timeout = 0;

        let error = ConfigValidator::validate_all(&config).unwrap_err();
        assert_eq!(
            error.to_string(),
            "Config error: Timeout must be greater than 0"
        );
    }

    #[test]
    fn test_validate_all_local_mode_rejects_zero_max_tokens() {
        let mut config = AppConfig::default();
        config.inference.mode = "local".to_string();
        config.cloud_api.api_key.clear();
        config.cloud_api.max_tokens = 0;

        let error = ConfigValidator::validate_all(&config).unwrap_err();
        assert_eq!(
            error.to_string(),
            "Config error: Max tokens must be greater than 0"
        );
    }

    #[test]
    fn test_validate_all_local_mode_rejects_empty_model() {
        let mut config = AppConfig::default();
        config.inference.mode = "local".to_string();
        config.cloud_api.api_key.clear();
        config.cloud_api.model.clear();

        let error = ConfigValidator::validate_all(&config).unwrap_err();
        assert_eq!(error.to_string(), "Config error: Model name is required");
    }

    #[test]
    fn test_config_manager_load_creates_default() {
        let path = std::path::PathBuf::from("/tmp/ale-my-eyes-test-unit/config.json");
        let _ = std::fs::remove_file(&path);
        let mut manager = ConfigManager::new(&path);
        manager.load().unwrap();
        assert_eq!(manager.config().cloud_api.provider, "openai");
        assert!(path.exists());
        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn test_config_manager_loads_legacy_ui_without_auto_speak() {
        let dir = std::path::PathBuf::from("/tmp/ale-my-eyes-test-legacy-ui");
        let path = dir.join("config.json");
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        std::fs::write(
            &path,
            r#"{
  "cloud_api": {
    "provider": "openai",
    "api_key": "sk-test",
    "api_url": "https://api.openai.com/v1",
    "model": "gpt-4o",
    "max_tokens": 1024,
    "timeout": 30
  },
  "models": {
    "auto_download": true,
    "max_download_size": 524288000,
    "preferred_quality": "balanced",
    "offline_mode": false,
    "models_dir": "models"
  },
  "inference": {
    "mode": "adaptive",
    "prefer_cloud": true,
    "timeout": 30,
    "fallback_to_local": true
  },
  "audio": {
    "sample_rate": 16000,
    "channels": 1,
    "buffer_size": 4096,
    "voice": "default",
    "speed": 1.0
  },
  "ui": {
    "language": "zh-CN",
    "theme": "system",
    "font_size": 16,
    "high_contrast": false,
    "screen_reader": true
  }
}"#,
        )
        .unwrap();

        let mut manager = ConfigManager::new(&path);
        manager.load().unwrap();

        assert!(manager.config().ui.auto_speak);
        let saved = std::fs::read_to_string(&path).unwrap();
        assert!(saved.contains("auto_speak"));
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn test_config_manager_load_does_not_rewrite_current_config() {
        let dir = std::path::PathBuf::from("/tmp/ale-my-eyes-test-no-rewrite");
        let path = dir.join("config.json");
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        let content = serde_json::to_string_pretty(&AppConfig::default()).unwrap();
        std::fs::write(&path, &content).unwrap();

        let before = std::fs::metadata(&path).unwrap().modified().unwrap();
        let mut manager = ConfigManager::new(&path);
        manager.load().unwrap();
        let after = std::fs::metadata(&path).unwrap().modified().unwrap();

        assert_eq!(before, after);
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn test_config_missing_sections_use_defaults() {
        let config: AppConfig =
            serde_json::from_str(r#"{"cloud_api":{"api_key":"sk-test"}}"#).unwrap();

        assert_eq!(config.cloud_api.api_key, "sk-test");
        assert_eq!(config.cloud_api.provider, "openai");
        assert_eq!(config.inference.mode, "adaptive");
        assert_eq!(config.audio.sample_rate, 16000);
        assert!(config.ui.auto_speak);
    }
}
