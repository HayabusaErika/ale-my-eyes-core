use crate::{AleError, Result};
use async_trait::async_trait;
use std::path::Path;

/// 视觉语言模型trait
#[async_trait]
pub trait VisionModel: Send + Sync {
    /// 描述图像内容
    async fn describe_image(&self, image_data: &[u8]) -> Result<String>;

    fn model_info(&self) -> crate::RuntimeModelInfo;
}

/// 基于ONNX的VLM模型
pub struct OnnxVlm {
    model_path: std::path::PathBuf,
    session: Option<ort::Session>,
}

impl OnnxVlm {
    pub async fn new(model_path: &Path) -> Result<Self> {
        let model_path = model_path.to_path_buf();

        // 检查模型文件是否存在
        if !model_path.exists() {
            return Err(AleError::VlmError(format!(
                "Model file not found: {}",
                model_path.display()
            )));
        }

        Ok(Self {
            model_path,
            session: None,
        })
    }

    fn load_model(&mut self) -> Result<()> {
        if self.session.is_some() {
            return Ok(());
        }

        let session = ort::Session::builder()
            .map_err(|e| AleError::VlmError(format!("Failed to create session builder: {}", e)))?
            .commit_from_file(&self.model_path)
            .map_err(|e| AleError::VlmError(format!("Failed to load ONNX model: {}", e)))?;

        self.session = Some(session);
        Ok(())
    }
}

#[async_trait]
impl VisionModel for OnnxVlm {
    async fn describe_image(&self, image_data: &[u8]) -> Result<String> {
        // 这里需要实现实际的图像描述逻辑
        // 需要预处理图像，运行ONNX推理，后处理结果
        Err(AleError::VlmError("Not implemented yet".to_string()))
    }

    fn model_info(&self) -> crate::RuntimeModelInfo {
        crate::RuntimeModelInfo {
            name: "onnx-vlm".to_string(),
            version: "1.0".to_string(),
            device: "cpu".to_string(),
            loaded: self.session.is_some(),
        }
    }
}
