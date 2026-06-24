use crate::{AleError, Result};
use async_trait::async_trait;
use serde::{Deserialize, Serialize};
use std::path::Path;

/// LLM推理后端
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum LlmBackend {
    /// 本地推理（使用llama.cpp）
    Local,
    /// 远程API（OpenAI、Anthropic等）
    Remote,
    /// ONNX Runtime
    Onnx,
    /// Candle（Rust原生）
    Candle,
}

/// LLM配置
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LlmConfig {
    pub backend: LlmBackend,
    pub model_path: Option<String>,
    pub api_url: Option<String>,
    pub api_key: Option<String>,
    pub model_name: Option<String>,
    pub max_tokens: usize,
    pub temperature: f32,
    pub top_p: f32,
}

impl Default for LlmConfig {
    fn default() -> Self {
        Self {
            backend: LlmBackend::Local,
            model_path: None,
            api_url: None,
            api_key: None,
            model_name: None,
            max_tokens: 512,
            temperature: 0.7,
            top_p: 0.9,
        }
    }
}

/// LLM响应
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LlmResponse {
    pub text: String,
    pub tokens_used: usize,
    pub finish_reason: String,
}

/// LLM trait
#[async_trait]
pub trait LanguageModel: Send + Sync {
    /// 生成文本
    async fn generate(&self, prompt: &str) -> Result<LlmResponse>;

    /// 流式生成
    async fn generate_stream(&self, prompt: &str) -> Result<Box<dyn tokio::io::AsyncRead + Unpin>>;

    fn model_info(&self) -> crate::RuntimeModelInfo;
}

/// 本地LLM（基于llama.cpp）
pub struct LocalLlm {
    config: LlmConfig,
    model: Option<llama_cpp_rs::LlamaModel>,
}

impl LocalLlm {
    pub async fn new(config: LlmConfig) -> Result<Self> {
        Ok(Self {
            config,
            model: None,
        })
    }

    fn load_model(&mut self) -> Result<()> {
        if self.model.is_some() {
            return Ok(());
        }

        let model_path = self
            .config
            .model_path
            .as_ref()
            .ok_or(AleError::Other(anyhow::anyhow!("Model path not specified")))?;

        // 检查模型文件是否存在
        if !Path::new(model_path).exists() {
            return Err(AleError::Other(anyhow::anyhow!(
                "Model file not found: {}",
                model_path
            )));
        }

        // 加载模型
        let model = llama_cpp_rs::LlamaModel::load_from_file(
            model_path,
            &llama_cpp_rs::LlamaParams::default(),
        )
        .map_err(|e| AleError::Other(anyhow::anyhow!("Failed to load model: {}", e)))?;

        self.model = Some(model);
        Ok(())
    }
}

#[async_trait]
impl LanguageModel for LocalLlm {
    async fn generate(&self, prompt: &str) -> Result<LlmResponse> {
        // 这里需要实现实际的生成逻辑
        // 由于llama-cpp-rs API的限制，这里简化处理
        Err(AleError::Other(anyhow::anyhow!("Not implemented yet")))
    }

    async fn generate_stream(&self, prompt: &str) -> Result<Box<dyn tokio::io::AsyncRead + Unpin>> {
        // 流式生成
        Err(AleError::Other(anyhow::anyhow!(
            "Streaming not implemented yet"
        )))
    }

    fn model_info(&self) -> crate::RuntimeModelInfo {
        crate::RuntimeModelInfo {
            name: self.config.model_name.clone().unwrap_or_default(),
            version: "1.0".to_string(),
            device: "cpu".to_string(),
            loaded: self.model.is_some(),
        }
    }
}
pub struct RemoteLlm {
    config: LlmConfig,
    client: reqwest::Client,
}

impl RemoteLlm {
    pub async fn new(config: LlmConfig) -> Result<Self> {
        let client = reqwest::Client::new();
        Ok(Self { config, client })
    }

    fn response_text(response_body: &serde_json::Value) -> Result<String> {
        response_body["choices"][0]["message"]["content"]
            .as_str()
            .filter(|text| !text.trim().is_empty())
            .map(str::to_string)
            .ok_or_else(|| AleError::Other(anyhow::anyhow!("Missing LLM response content")))
    }
}

#[async_trait]
impl LanguageModel for RemoteLlm {
    async fn generate(&self, prompt: &str) -> Result<LlmResponse> {
        let api_url = self
            .config
            .api_url
            .as_ref()
            .ok_or(AleError::Other(anyhow::anyhow!("API URL not specified")))?;

        let api_key = self
            .config
            .api_key
            .as_ref()
            .ok_or(AleError::Other(anyhow::anyhow!("API key not specified")))?;

        let model_name = self
            .config
            .model_name
            .as_ref()
            .ok_or(AleError::Other(anyhow::anyhow!("Model name not specified")))?;

        // 构建请求体
        let request_body = serde_json::json!({
            "model": model_name,
            "messages": [
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            "max_tokens": self.config.max_tokens,
            "temperature": self.config.temperature,
            "top_p": self.config.top_p
        });

        // 发送请求
        let response = self
            .client
            .post(api_url)
            .header("Authorization", format!("Bearer {}", api_key))
            .header("Content-Type", "application/json")
            .json(&request_body)
            .send()
            .await
            .map_err(|e| AleError::Other(anyhow::anyhow!("API request failed: {}", e)))?;

        if !response.status().is_success() {
            let error_text = response.text().await.unwrap_or_default();
            return Err(AleError::Other(anyhow::anyhow!(
                "API request failed: {}",
                error_text
            )));
        }

        let response_body: serde_json::Value = response
            .json()
            .await
            .map_err(|e| AleError::Other(anyhow::anyhow!("Failed to parse response: {}", e)))?;

        // 提取响应文本
        let text = Self::response_text(&response_body)?;

        let tokens_used = response_body["usage"]["total_tokens"].as_u64().unwrap_or(0) as usize;

        Ok(LlmResponse {
            text,
            tokens_used,
            finish_reason: "stop".to_string(),
        })
    }

    async fn generate_stream(&self, prompt: &str) -> Result<Box<dyn tokio::io::AsyncRead + Unpin>> {
        // 流式生成
        Err(AleError::Other(anyhow::anyhow!(
            "Streaming not implemented yet"
        )))
    }

    fn model_info(&self) -> crate::RuntimeModelInfo {
        crate::RuntimeModelInfo {
            name: self.config.model_name.clone().unwrap_or_default(),
            version: "1.0".to_string(),
            device: "remote".to_string(),
            loaded: true,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_response_text_rejects_missing_content() {
        let response = serde_json::json!({"choices": [{"message": {}}]});
        assert!(RemoteLlm::response_text(&response).is_err());
    }

    #[test]
    fn test_response_text_rejects_empty_content() {
        let response = serde_json::json!({"choices": [{"message": {"content": "  "}}]});
        assert!(RemoteLlm::response_text(&response).is_err());
    }

    #[test]
    fn test_response_text_accepts_content() {
        let response = serde_json::json!({"choices": [{"message": {"content": "hello"}}]});
        assert_eq!(RemoteLlm::response_text(&response).unwrap(), "hello");
    }
}

/// LLM工厂
pub struct LlmFactory;

impl LlmFactory {
    pub async fn create(config: LlmConfig) -> Result<Box<dyn LanguageModel>> {
        match config.backend {
            LlmBackend::Local => {
                let llm = LocalLlm::new(config).await?;
                Ok(Box::new(llm))
            }
            LlmBackend::Remote => {
                let llm = RemoteLlm::new(config).await?;
                Ok(Box::new(llm))
            }
            _ => Err(AleError::Other(anyhow::anyhow!("Unsupported LLM backend"))),
        }
    }
}
