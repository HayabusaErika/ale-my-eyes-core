use crate::{AleError, Result};
use serde::{Deserialize, Serialize};
use std::path::{Path, PathBuf};
use std::sync::Arc;
use tokio::sync::Mutex;

/// 模型信息
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelInfo {
    pub id: String,
    pub name: String,
    pub description: String,
    pub size: u64, // 字节
    pub repo: String,
    pub filename: String,
    pub quantization: Option<String>,
    pub purpose: String,
    pub recommended_for: String,
}

/// 下载进度
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DownloadProgress {
    pub model_id: String,
    pub total_bytes: u64,
    pub downloaded_bytes: u64,
    pub progress: f32, // 0.0 - 1.0
    pub speed: f32,    // 字节/秒
    pub eta: u32,      // 预计剩余秒数
}

/// 进度回调函数类型
pub type ProgressCallback = Box<dyn Fn(DownloadProgress) + Send + Sync>;

/// 模型下载器
pub struct ModelDownloader {
    models_dir: PathBuf,
    progress_callback: Option<ProgressCallback>,
    client: reqwest::Client,
    known_models: Vec<ModelInfo>,
}

impl ModelDownloader {
    pub fn new(models_dir: &Path) -> Self {
        let client = reqwest::Client::builder()
            .timeout(std::time::Duration::from_secs(300))
            .build()
            .unwrap_or_else(|_| reqwest::Client::new());

        Self {
            models_dir: models_dir.to_path_buf(),
            progress_callback: None,
            client,
            known_models: Self::default_known_models(),
        }
    }

    /// 设置进度回调
    pub fn set_progress_callback(&mut self, callback: ProgressCallback) {
        self.progress_callback = Some(callback);
    }

    /// 默认的已知模型列表
    fn default_known_models() -> Vec<ModelInfo> {
        vec![
            ModelInfo {
                id: "whisper-tiny".to_string(),
                name: "Whisper Tiny".to_string(),
                description: "轻量级语音识别模型".to_string(),
                size: 75 * 1024 * 1024, // 75MB
                repo: "ggml-org/whisper.cpp".to_string(),
                filename: "ggml-tiny.bin".to_string(),
                quantization: Some("q4_0".to_string()),
                purpose: "基础语音识别".to_string(),
                recommended_for: "低性能设备".to_string(),
            },
            ModelInfo {
                id: "whisper-small".to_string(),
                name: "Whisper Small".to_string(),
                description: "中等质量语音识别模型".to_string(),
                size: 244 * 1024 * 1024, // 244MB
                repo: "ggml-org/whisper.cpp".to_string(),
                filename: "ggml-small.bin".to_string(),
                quantization: Some("q4_0".to_string()),
                purpose: "高质量语音识别".to_string(),
                recommended_for: "中端设备".to_string(),
            },
            ModelInfo {
                id: "whisper-large-v3".to_string(),
                name: "Whisper Large V3".to_string(),
                description: "最高质量语音识别模型".to_string(),
                size: 1500 * 1024 * 1024, // 1.5GB
                repo: "ggml-org/whisper.cpp".to_string(),
                filename: "ggml-large-v3.bin".to_string(),
                quantization: Some("q4_0".to_string()),
                purpose: "专业级语音识别".to_string(),
                recommended_for: "高端设备".to_string(),
            },
            ModelInfo {
                id: "piper-zh_CN".to_string(),
                name: "Piper 中文语音".to_string(),
                description: "轻量级中文语音合成".to_string(),
                size: 50 * 1024 * 1024, // 50MB
                repo: "rhasspy/piper".to_string(),
                filename: "zh_CN-huayan-medium.onnx".to_string(),
                quantization: None,
                purpose: "中文语音合成".to_string(),
                recommended_for: "所有设备".to_string(),
            },
            ModelInfo {
                id: "piper-en_US".to_string(),
                name: "Piper 英文语音".to_string(),
                description: "轻量级英文语音合成".to_string(),
                size: 50 * 1024 * 1024, // 50MB
                repo: "rhasspy/piper".to_string(),
                filename: "en_US-amy-medium.onnx".to_string(),
                quantization: None,
                purpose: "英文语音合成".to_string(),
                recommended_for: "所有设备".to_string(),
            },
        ]
    }

    /// 获取所有可用模型
    pub fn available_models(&self) -> &[ModelInfo] {
        &self.known_models
    }

    /// 根据ID获取模型信息
    pub fn get_model_info(&self, model_id: &str) -> Option<&ModelInfo> {
        self.known_models.iter().find(|m| m.id == model_id)
    }

    /// 检查模型是否已下载
    pub fn is_model_downloaded(&self, model_id: &str) -> bool {
        if let Some(model) = self.get_model_info(model_id) {
            let path = self.models_dir.join(&model.filename);
            path.exists()
        } else {
            false
        }
    }

    /// 获取模型文件路径
    pub fn get_model_path(&self, model_id: &str) -> Option<PathBuf> {
        if let Some(model) = self.get_model_info(model_id) {
            let path = self.models_dir.join(&model.filename);
            if path.exists() {
                Some(path)
            } else {
                None
            }
        } else {
            None
        }
    }

    /// 下载模型
    pub async fn download_model(&self, model_id: &str) -> Result<PathBuf> {
        let model = self
            .get_model_info(model_id)
            .ok_or_else(|| AleError::Other(anyhow::anyhow!("Unknown model: {}", model_id)))?
            .clone();

        // 检查是否已下载
        let target_path = self.models_dir.join(&model.filename);
        if target_path.exists() {
            return Ok(target_path);
        }

        // 确保目录存在
        std::fs::create_dir_all(&self.models_dir)?;

        // 构建下载URL
        let url = format!(
            "https://huggingface.co/{}/resolve/main/{}",
            model.repo, model.filename
        );

        // 开始下载
        let response = self
            .client
            .get(&url)
            .send()
            .await
            .map_err(|e| AleError::Other(anyhow::anyhow!("Download request failed: {}", e)))?;

        if !response.status().is_success() {
            return Err(AleError::Other(anyhow::anyhow!(
                "Download failed with status: {}",
                response.status()
            )));
        }

        // 获取文件大小
        let total_size = response.content_length().unwrap_or(model.size);

        // 创建临时文件
        let temp_path = target_path.with_extension("tmp");
        let mut file = std::fs::File::create(&temp_path)?;
        let temp_file_guard = TempFileGuard::new(temp_path.clone());

        // 下载并写入文件
        let mut downloaded: u64 = 0;
        let start_time = std::time::Instant::now();
        let mut stream = response.bytes_stream();

        use futures::StreamExt;
        use std::io::Write;

        while let Some(chunk) = stream.next().await {
            let chunk =
                chunk.map_err(|e| AleError::Other(anyhow::anyhow!("Download error: {}", e)))?;
            file.write_all(&chunk)?;

            downloaded += chunk.len() as u64;

            // 计算进度
            let progress = if total_size > 0 {
                downloaded as f32 / total_size as f32
            } else {
                0.0
            };
            let elapsed = start_time.elapsed().as_secs_f32();
            let speed = if elapsed > 0.0 {
                downloaded as f32 / elapsed
            } else {
                0.0
            };
            let remaining_bytes = total_size.saturating_sub(downloaded);
            let eta = if speed > 0.0 {
                (remaining_bytes as f32 / speed) as u32
            } else {
                0
            };

            // 调用进度回调
            if let Some(callback) = &self.progress_callback {
                callback(DownloadProgress {
                    model_id: model_id.to_string(),
                    total_bytes: total_size,
                    downloaded_bytes: downloaded,
                    progress,
                    speed,
                    eta,
                });
            }
        }

        // 重命名临时文件
        std::fs::rename(&temp_path, &target_path)?;
        temp_file_guard.disarm();

        Ok(target_path)
    }

    /// 删除模型
    pub fn delete_model(&self, model_id: &str) -> Result<()> {
        if let Some(model) = self.get_model_info(model_id) {
            let path = self.models_dir.join(&model.filename);
            if path.exists() {
                std::fs::remove_file(&path)?;
            }
        }
        Ok(())
    }

    /// 获取已下载模型列表
    pub fn downloaded_models(&self) -> Vec<&ModelInfo> {
        self.known_models
            .iter()
            .filter(|m| self.is_model_downloaded(&m.id))
            .collect()
    }

    /// 获取推荐模型（根据设备性能）
    pub fn recommended_models(
        &self,
        device_performance: &crate::inference::DevicePerformance,
    ) -> Vec<&ModelInfo> {
        match device_performance {
            crate::inference::DevicePerformance::Low => self
                .known_models
                .iter()
                .filter(|m| m.recommended_for == "低性能设备" || m.recommended_for == "所有设备")
                .collect(),
            crate::inference::DevicePerformance::Medium => self
                .known_models
                .iter()
                .filter(|m| m.recommended_for == "中端设备" || m.recommended_for == "所有设备")
                .collect(),
            crate::inference::DevicePerformance::High => self
                .known_models
                .iter()
                .filter(|m| m.recommended_for == "高端设备" || m.recommended_for == "所有设备")
                .collect(),
        }
    }

    /// 自动下载推荐模型
    pub async fn download_recommended_models(
        &self,
        device_performance: &crate::inference::DevicePerformance,
    ) -> Result<Vec<PathBuf>> {
        let recommended = self.recommended_models(device_performance);
        let mut paths = Vec::new();

        for model in recommended {
            if !self.is_model_downloaded(&model.id) {
                let path = self.download_model(&model.id).await?;
                paths.push(path);
            }
        }

        Ok(paths)
    }
}

struct TempFileGuard {
    path: PathBuf,
    keep: std::sync::atomic::AtomicBool,
}

impl TempFileGuard {
    fn new(path: PathBuf) -> Self {
        Self {
            path,
            keep: std::sync::atomic::AtomicBool::new(false),
        }
    }

    fn disarm(&self) {
        self.keep.store(true, std::sync::atomic::Ordering::Relaxed);
    }
}

impl Drop for TempFileGuard {
    fn drop(&mut self) {
        if !self.keep.load(std::sync::atomic::Ordering::Relaxed) {
            let _ = std::fs::remove_file(&self.path);
        }
    }
}

#[deprecated(note = "unused wrapper; use ModelDownloader/SmartModelManager directly")]
pub struct ModelDownloadManager {
    downloader: Arc<Mutex<ModelDownloader>>,
    max_concurrent_downloads: usize,
}

impl ModelDownloadManager {
    pub fn new(models_dir: &Path, max_concurrent: usize) -> Self {
        Self {
            downloader: Arc::new(Mutex::new(ModelDownloader::new(models_dir))),
            max_concurrent_downloads: max_concurrent,
        }
    }

    /// 批量下载模型
    pub async fn download_models(&self, model_ids: &[&str]) -> Result<Vec<PathBuf>> {
        let downloader = self.downloader.lock().await;
        let mut paths = Vec::new();

        for model_id in model_ids {
            let path = downloader.download_model(model_id).await?;
            paths.push(path);
        }

        Ok(paths)
    }

    /// 并发下载模型（限制并发数）
    pub async fn download_models_concurrent(&self, model_ids: &[&str]) -> Result<Vec<PathBuf>> {
        let downloader = self.downloader.clone();
        let mut handles = Vec::new();

        for chunk in model_ids.chunks(self.max_concurrent_downloads) {
            let downloader = downloader.clone();
            let chunk: Vec<String> = chunk.iter().map(|s| s.to_string()).collect();

            let handle = tokio::spawn(async move {
                let downloader = downloader.lock().await;
                let mut paths = Vec::new();

                for model_id in chunk {
                    let path = downloader.download_model(&model_id).await?;
                    paths.push(path);
                }

                Ok::<Vec<PathBuf>, AleError>(paths)
            });

            handles.push(handle);
        }

        let mut all_paths = Vec::new();
        for handle in handles {
            let paths = handle
                .await
                .map_err(|e| AleError::Other(anyhow::anyhow!("Task join error: {}", e)))??;
            all_paths.extend(paths);
        }

        Ok(all_paths)
    }
}
