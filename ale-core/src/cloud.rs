use crate::{AleError, Result};
use async_trait::async_trait;
use base64::{engine::general_purpose, Engine as _};
use serde::{Deserialize, Serialize};
use std::time::Duration;

/// 云端API提供商
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum CloudProvider {
    OpenAI,
    Anthropic,
    Google,
    Azure,
    Custom(String),
}

/// 云端API配置
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CloudConfig {
    pub provider: CloudProvider,
    pub api_key: String,
    pub api_url: String,
    pub model: String,
    pub max_tokens: usize,
    pub timeout: Duration,
    pub retry_count: u32,
}

impl Default for CloudConfig {
    fn default() -> Self {
        Self {
            provider: CloudProvider::OpenAI,
            api_key: String::new(),
            api_url: "https://api.openai.com/v1".to_string(),
            model: "gpt-4o".to_string(),
            max_tokens: 1024,
            timeout: Duration::from_secs(30),
            retry_count: 3,
        }
    }
}

/// 云端API响应
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CloudResponse {
    pub content: String,
    pub tokens_used: usize,
    pub model: String,
    pub provider: CloudProvider,
}

/// 云端API trait
#[async_trait]
pub trait CloudApi: Send + Sync {
    /// 发送文本请求
    async fn chat(&self, messages: Vec<CloudMessage>) -> Result<CloudResponse>;

    /// 发送图像请求（描述模式）
    async fn vision(&self, image_data: &[u8], prompt: &str) -> Result<CloudResponse>;

    /// 发送图像请求（问答模式，支持 Function Calling）
    async fn vision_ask(
        &self,
        image_data: &[u8],
        question: &str,
        tools: Option<Vec<serde_json::Value>>,
    ) -> Result<VisionResponse>;

    /// 语音识别
    async fn transcribe(&self, audio_data: &[u8]) -> Result<CloudResponse>;

    /// 语音合成
    async fn synthesize(&self, text: &str) -> Result<Vec<u8>>;

    /// 检查连接状态
    async fn health_check(&self) -> Result<bool>;
}

/// 视觉问答响应（支持 Function Calling）
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VisionResponse {
    /// 文本回答
    pub content: String,
    /// 工具调用（如果有）
    pub tool_calls: Option<Vec<ToolCall>>,
    pub tokens_used: usize,
    pub model: String,
}

/// 工具调用
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ToolCall {
    pub id: String,
    pub function: FunctionCall,
}

/// 函数调用
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FunctionCall {
    pub name: String,
    pub arguments: String,
}

/// 云端消息
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CloudMessage {
    pub role: String,
    pub content: String,
}

/// OpenAI API 实现
pub struct OpenAIApi {
    config: CloudConfig,
    client: reqwest::Client,
}

impl OpenAIApi {
    pub fn new(config: CloudConfig) -> Self {
        let client = reqwest::Client::builder()
            .timeout(config.timeout)
            .build()
            .unwrap_or_else(|_| reqwest::Client::new());

        Self { config, client }
    }

    fn chat_content(response_body: &serde_json::Value) -> Result<String> {
        response_body["choices"][0]["message"]["content"]
            .as_str()
            .filter(|content| !content.trim().is_empty())
            .map(str::to_string)
            .ok_or_else(|| AleError::CloudApiError("Missing chat response content".to_string()))
    }

    fn transcription_text(response_body: &serde_json::Value) -> Result<String> {
        response_body["text"]
            .as_str()
            .filter(|text| !text.trim().is_empty())
            .map(str::to_string)
            .ok_or_else(|| AleError::CloudApiError("Missing transcription text".to_string()))
    }
}

#[async_trait]
impl CloudApi for OpenAIApi {
    async fn chat(&self, messages: Vec<CloudMessage>) -> Result<CloudResponse> {
        let url = format!("{}/chat/completions", self.config.api_url);

        let request_body = serde_json::json!({
            "model": self.config.model,
            "messages": messages,
            "max_tokens": self.config.max_tokens,
        });

        let response = self
            .client
            .post(&url)
            .header("Authorization", format!("Bearer {}", self.config.api_key))
            .header("Content-Type", "application/json")
            .json(&request_body)
            .send()
            .await
            .map_err(|e| AleError::CloudApiError(format!("Request failed: {}", e)))?;

        if !response.status().is_success() {
            let error_text = response.text().await.unwrap_or_default();
            return Err(AleError::CloudApiError(format!(
                "API error: {}",
                error_text
            )));
        }

        let response_body: serde_json::Value = response
            .json()
            .await
            .map_err(|e| AleError::CloudApiError(format!("Parse error: {}", e)))?;

        let content = Self::chat_content(&response_body)?;

        let tokens_used = response_body["usage"]["total_tokens"].as_u64().unwrap_or(0) as usize;

        Ok(CloudResponse {
            content,
            tokens_used,
            model: self.config.model.clone(),
            provider: self.config.provider.clone(),
        })
    }

    async fn vision(&self, image_data: &[u8], prompt: &str) -> Result<CloudResponse> {
        let url = format!("{}/chat/completions", self.config.api_url);

        // 将图像转换为base64
        let image_base64 = general_purpose::STANDARD.encode(image_data);

        let request_body = serde_json::json!({
            "model": self.config.model,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": prompt
                        },
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": format!("data:image/jpeg;base64,{}", image_base64)
                            }
                        }
                    ]
                }
            ],
            "max_tokens": self.config.max_tokens,
        });

        let response = self
            .client
            .post(&url)
            .header("Authorization", format!("Bearer {}", self.config.api_key))
            .header("Content-Type", "application/json")
            .json(&request_body)
            .send()
            .await
            .map_err(|e| AleError::CloudApiError(format!("Request failed: {}", e)))?;

        if !response.status().is_success() {
            let error_text = response.text().await.unwrap_or_default();
            return Err(AleError::CloudApiError(format!(
                "API error: {}",
                error_text
            )));
        }

        let response_body: serde_json::Value = response
            .json()
            .await
            .map_err(|e| AleError::CloudApiError(format!("Parse error: {}", e)))?;

        let content = Self::chat_content(&response_body)?;

        let tokens_used = response_body["usage"]["total_tokens"].as_u64().unwrap_or(0) as usize;

        Ok(CloudResponse {
            content,
            tokens_used,
            model: self.config.model.clone(),
            provider: self.config.provider.clone(),
        })
    }

    async fn transcribe(&self, audio_data: &[u8]) -> Result<CloudResponse> {
        let url = format!("{}/audio/transcriptions", self.config.api_url);

        // 创建multipart表单
        let form = reqwest::multipart::Form::new()
            .part(
                "file",
                reqwest::multipart::Part::bytes(audio_data.to_vec())
                    .file_name("audio.wav")
                    .mime_str("audio/wav")
                    .map_err(|e| AleError::CloudApiError(format!("Invalid MIME type: {e}")))?,
            )
            .text("model", "whisper-1");

        let response = self
            .client
            .post(&url)
            .header("Authorization", format!("Bearer {}", self.config.api_key))
            .multipart(form)
            .send()
            .await
            .map_err(|e| AleError::CloudApiError(format!("Request failed: {}", e)))?;

        if !response.status().is_success() {
            let error_text = response.text().await.unwrap_or_default();
            return Err(AleError::CloudApiError(format!(
                "API error: {}",
                error_text
            )));
        }

        let response_body: serde_json::Value = response
            .json()
            .await
            .map_err(|e| AleError::CloudApiError(format!("Parse error: {}", e)))?;

        let text = Self::transcription_text(&response_body)?;

        Ok(CloudResponse {
            content: text,
            tokens_used: 0,
            model: "whisper-1".to_string(),
            provider: CloudProvider::OpenAI,
        })
    }

    async fn vision_ask(
        &self,
        image_data: &[u8],
        question: &str,
        tools: Option<Vec<serde_json::Value>>,
    ) -> Result<VisionResponse> {
        let url = format!("{}/chat/completions", self.config.api_url);

        let image_base64 = general_purpose::STANDARD.encode(image_data);

        let mut request_body = serde_json::json!({
            "model": self.config.model,
            "messages": [
                {
                    "role": "system",
                    "content": "你是 Ale, My Eyes! 智能视觉辅助助手。用户会发送一张图片和一个问题，请根据图片内容回答问题。如果用户要求执行操作，请使用提供的工具函数。"
                },
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": question
                        },
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": format!("data:image/jpeg;base64,{}", image_base64)
                            }
                        }
                    ]
                }
            ],
            "max_tokens": self.config.max_tokens,
        });

        // 添加工具定义（如果有）
        if let Some(tools) = tools {
            request_body["tools"] = serde_json::json!(tools);
        }

        let response = self
            .client
            .post(&url)
            .header("Authorization", format!("Bearer {}", self.config.api_key))
            .header("Content-Type", "application/json")
            .json(&request_body)
            .send()
            .await
            .map_err(|e| AleError::CloudApiError(format!("Request failed: {}", e)))?;

        if !response.status().is_success() {
            let error_text = response.text().await.unwrap_or_default();
            return Err(AleError::CloudApiError(format!(
                "API error: {}",
                error_text
            )));
        }

        let response_body: serde_json::Value = response
            .json()
            .await
            .map_err(|e| AleError::CloudApiError(format!("Parse error: {}", e)))?;

        let message = &response_body["choices"][0]["message"];
        let content = message["content"].as_str().unwrap_or_default().to_string();
        let tool_calls = message["tool_calls"].as_array();
        if content.trim().is_empty() && tool_calls.map(|calls| calls.is_empty()).unwrap_or(true) {
            return Err(AleError::CloudApiError(
                "Missing vision response content or tool calls".to_string(),
            ));
        }

        let tool_calls = tool_calls.map(|calls| {
            calls
                .iter()
                .map(|tc| ToolCall {
                    id: tc["id"].as_str().unwrap_or_default().to_string(),
                    function: FunctionCall {
                        name: tc["function"]["name"]
                            .as_str()
                            .unwrap_or_default()
                            .to_string(),
                        arguments: tc["function"]["arguments"]
                            .as_str()
                            .unwrap_or_default()
                            .to_string(),
                    },
                })
                .collect()
        });

        let tokens_used = response_body["usage"]["total_tokens"].as_u64().unwrap_or(0) as usize;

        Ok(VisionResponse {
            content,
            tool_calls,
            tokens_used,
            model: self.config.model.clone(),
        })
    }

    async fn synthesize(&self, text: &str) -> Result<Vec<u8>> {
        let url = format!("{}/audio/speech", self.config.api_url);

        let request_body = serde_json::json!({
            "model": "tts-1",
            "input": text,
            "voice": "alloy",
            "response_format": "wav",
        });

        let response = self
            .client
            .post(&url)
            .header("Authorization", format!("Bearer {}", self.config.api_key))
            .header("Content-Type", "application/json")
            .json(&request_body)
            .send()
            .await
            .map_err(|e| AleError::CloudApiError(format!("Request failed: {}", e)))?;

        if !response.status().is_success() {
            let error_text = response.text().await.unwrap_or_default();
            return Err(AleError::CloudApiError(format!(
                "API error: {}",
                error_text
            )));
        }

        let audio_data = response
            .bytes()
            .await
            .map_err(|e| AleError::CloudApiError(format!("Failed to read audio: {}", e)))?;

        Ok(audio_data.to_vec())
    }

    async fn health_check(&self) -> Result<bool> {
        let url = format!("{}/models", self.config.api_url);

        let response = self
            .client
            .get(&url)
            .header("Authorization", format!("Bearer {}", self.config.api_key))
            .send()
            .await
            .map_err(|e| AleError::CloudApiError(format!("Health check failed: {}", e)))?;

        Ok(response.status().is_success())
    }
}

struct AnthropicApi {
    config: CloudConfig,
    client: reqwest::Client,
}

impl AnthropicApi {
    fn new(config: CloudConfig) -> Self {
        Self {
            client: client_for(&config),
            config,
        }
    }

    fn split_system_messages(
        messages: Vec<CloudMessage>,
    ) -> (Option<String>, Vec<serde_json::Value>) {
        let mut system = Vec::new();
        let mut chat_messages = Vec::new();

        for message in messages {
            if message.role == "system" {
                system.push(message.content);
            } else {
                let role = if message.role == "assistant" {
                    "assistant"
                } else {
                    "user"
                };
                chat_messages.push(serde_json::json!({
                    "role": role,
                    "content": message.content,
                }));
            }
        }

        let system = if system.is_empty() {
            None
        } else {
            Some(system.join("\n\n"))
        };
        (system, chat_messages)
    }

    fn content_text(response_body: &serde_json::Value) -> Result<String> {
        let text = response_body["content"]
            .as_array()
            .map_or_else(String::new, |content| {
                content
                    .iter()
                    .filter(|part| part["type"].as_str() == Some("text"))
                    .filter_map(|part| part["text"].as_str())
                    .collect::<Vec<_>>()
                    .join("\n")
            });

        if text.trim().is_empty() {
            Err(AleError::CloudApiError(
                "Missing Anthropic response content".to_string(),
            ))
        } else {
            Ok(text)
        }
    }

    fn tool_calls(response_body: &serde_json::Value) -> Option<Vec<ToolCall>> {
        let calls = response_body["content"]
            .as_array()?
            .iter()
            .filter(|part| part["type"].as_str() == Some("tool_use"))
            .map(|part| ToolCall {
                id: part["id"].as_str().unwrap_or_default().to_string(),
                function: FunctionCall {
                    name: part["name"].as_str().unwrap_or_default().to_string(),
                    arguments: part["input"].to_string(),
                },
            })
            .collect::<Vec<_>>();

        if calls.is_empty() {
            None
        } else {
            Some(calls)
        }
    }

    fn convert_tools(tools: Vec<serde_json::Value>) -> Vec<serde_json::Value> {
        tools
            .into_iter()
            .filter_map(|tool| {
                let function = tool.get("function")?;
                Some(serde_json::json!({
                    "name": function.get("name").cloned().unwrap_or_default(),
                    "description": function.get("description").cloned().unwrap_or_default(),
                    "input_schema": function.get("parameters").cloned().unwrap_or_else(|| serde_json::json!({"type":"object"})),
                }))
            })
            .collect()
    }
}

#[async_trait]
impl CloudApi for AnthropicApi {
    async fn chat(&self, messages: Vec<CloudMessage>) -> Result<CloudResponse> {
        let url = format!("{}/messages", self.config.api_url.trim_end_matches('/'));
        let (system, messages) = Self::split_system_messages(messages);
        let mut request_body = serde_json::json!({
            "model": self.config.model,
            "messages": messages,
            "max_tokens": self.config.max_tokens,
        });
        if let Some(system) = system {
            request_body["system"] = serde_json::json!(system);
        }

        let response_body = self
            .client
            .post(&url)
            .header("x-api-key", &self.config.api_key)
            .header("anthropic-version", "2023-06-01")
            .header("Content-Type", "application/json")
            .json(&request_body)
            .send()
            .await
            .map_err(|e| AleError::CloudApiError(format!("Request failed: {}", e)))?
            .error_for_status_with_body()
            .await?;

        let content = Self::content_text(&response_body)?;
        let tokens_used = response_body["usage"]["input_tokens"].as_u64().unwrap_or(0)
            + response_body["usage"]["output_tokens"]
                .as_u64()
                .unwrap_or(0);
        Ok(CloudResponse {
            content,
            tokens_used: tokens_used as usize,
            model: self.config.model.clone(),
            provider: self.config.provider.clone(),
        })
    }

    async fn vision(&self, image_data: &[u8], prompt: &str) -> Result<CloudResponse> {
        let response = self.vision_ask(image_data, prompt, None).await?;
        Ok(CloudResponse {
            content: response.content,
            tokens_used: response.tokens_used,
            model: response.model,
            provider: self.config.provider.clone(),
        })
    }

    async fn vision_ask(
        &self,
        image_data: &[u8],
        question: &str,
        tools: Option<Vec<serde_json::Value>>,
    ) -> Result<VisionResponse> {
        let url = format!("{}/messages", self.config.api_url.trim_end_matches('/'));
        let image_base64 = general_purpose::STANDARD.encode(image_data);
        let mut request_body = serde_json::json!({
            "model": self.config.model,
            "max_tokens": self.config.max_tokens,
            "system": "你是 Ale, My Eyes! 智能视觉辅助助手。用户会发送一张图片和一个问题，请根据图片内容回答问题。如果用户要求执行操作，请使用提供的工具函数。",
            "messages": [{
                "role": "user",
                "content": [
                    {"type": "text", "text": question},
                    {"type": "image", "source": {"type": "base64", "media_type": "image/jpeg", "data": image_base64}}
                ]
            }]
        });
        if let Some(tools) = tools {
            let tools = Self::convert_tools(tools);
            if !tools.is_empty() {
                request_body["tools"] = serde_json::json!(tools);
            }
        }

        let response_body = self
            .client
            .post(&url)
            .header("x-api-key", &self.config.api_key)
            .header("anthropic-version", "2023-06-01")
            .header("Content-Type", "application/json")
            .json(&request_body)
            .send()
            .await
            .map_err(|e| AleError::CloudApiError(format!("Request failed: {}", e)))?
            .error_for_status_with_body()
            .await?;

        let content = Self::content_text(&response_body).unwrap_or_default();
        let tool_calls = Self::tool_calls(&response_body);
        if content.trim().is_empty() && tool_calls.as_ref().map(|c| c.is_empty()).unwrap_or(true) {
            return Err(AleError::CloudApiError(
                "Missing Anthropic vision response content or tool calls".to_string(),
            ));
        }
        let tokens_used = response_body["usage"]["input_tokens"].as_u64().unwrap_or(0)
            + response_body["usage"]["output_tokens"]
                .as_u64()
                .unwrap_or(0);

        Ok(VisionResponse {
            content,
            tool_calls,
            tokens_used: tokens_used as usize,
            model: self.config.model.clone(),
        })
    }

    async fn transcribe(&self, _audio_data: &[u8]) -> Result<CloudResponse> {
        unsupported_provider_feature(&self.config.provider, "speech transcription")
    }

    async fn synthesize(&self, _text: &str) -> Result<Vec<u8>> {
        unsupported_provider_feature(&self.config.provider, "speech synthesis")
    }

    async fn health_check(&self) -> Result<bool> {
        let url = format!("{}/models", self.config.api_url.trim_end_matches('/'));
        let response = self
            .client
            .get(&url)
            .header("x-api-key", &self.config.api_key)
            .header("anthropic-version", "2023-06-01")
            .send()
            .await
            .map_err(|e| AleError::CloudApiError(format!("Health check failed: {}", e)))?;
        Ok(response.status().is_success())
    }
}

struct GoogleApi {
    config: CloudConfig,
    client: reqwest::Client,
}

impl GoogleApi {
    fn new(config: CloudConfig) -> Self {
        Self {
            client: client_for(&config),
            config,
        }
    }

    fn request(&self, method: reqwest::Method, url: String) -> reqwest::RequestBuilder {
        self.client
            .request(method, url)
            .header("x-goog-api-key", &self.config.api_key)
    }

    fn generate_url(&self) -> String {
        format!(
            "{}/models/{}:generateContent",
            self.config.api_url.trim_end_matches('/'),
            self.config.model
        )
    }

    fn convert_messages(messages: Vec<CloudMessage>) -> (Option<String>, Vec<serde_json::Value>) {
        let mut system = Vec::new();
        let mut contents = Vec::new();
        for message in messages {
            if message.role == "system" {
                system.push(message.content);
            } else {
                let role = if message.role == "assistant" {
                    "model"
                } else {
                    "user"
                };
                contents.push(serde_json::json!({
                    "role": role,
                    "parts": [{"text": message.content}],
                }));
            }
        }
        let system = if system.is_empty() {
            None
        } else {
            Some(system.join("\n\n"))
        };
        (system, contents)
    }

    fn content_text(response_body: &serde_json::Value) -> Result<String> {
        let text = response_body["candidates"][0]["content"]["parts"]
            .as_array()
            .map_or_else(String::new, |parts| {
                parts
                    .iter()
                    .filter_map(|part| part["text"].as_str())
                    .collect::<Vec<_>>()
                    .join("\n")
            });
        if text.trim().is_empty() {
            Err(AleError::CloudApiError(
                "Missing Google response content".to_string(),
            ))
        } else {
            Ok(text)
        }
    }

    fn tool_calls(response_body: &serde_json::Value) -> Option<Vec<ToolCall>> {
        let calls = response_body["candidates"][0]["content"]["parts"]
            .as_array()?
            .iter()
            .filter_map(|part| part.get("functionCall"))
            .map(|call| ToolCall {
                id: call["name"].as_str().unwrap_or_default().to_string(),
                function: FunctionCall {
                    name: call["name"].as_str().unwrap_or_default().to_string(),
                    arguments: call["args"].to_string(),
                },
            })
            .collect::<Vec<_>>();
        if calls.is_empty() {
            None
        } else {
            Some(calls)
        }
    }

    fn convert_tools(tools: Vec<serde_json::Value>) -> Vec<serde_json::Value> {
        let declarations = tools
            .into_iter()
            .filter_map(|tool| tool.get("function").cloned())
            .map(|function| {
                serde_json::json!({
                    "name": function.get("name").cloned().unwrap_or_default(),
                    "description": function.get("description").cloned().unwrap_or_default(),
                    "parameters": function.get("parameters").cloned().unwrap_or_else(|| serde_json::json!({"type":"object"})),
                })
            })
            .collect::<Vec<_>>();
        if declarations.is_empty() {
            Vec::new()
        } else {
            vec![serde_json::json!({"function_declarations": declarations})]
        }
    }
}

#[async_trait]
impl CloudApi for GoogleApi {
    async fn chat(&self, messages: Vec<CloudMessage>) -> Result<CloudResponse> {
        let (system, contents) = Self::convert_messages(messages);
        let mut request_body = serde_json::json!({
            "contents": contents,
            "generationConfig": {"maxOutputTokens": self.config.max_tokens},
        });
        if let Some(system) = system {
            request_body["systemInstruction"] = serde_json::json!({"parts": [{"text": system}]});
        }

        let response_body = self
            .request(reqwest::Method::POST, self.generate_url())
            .header("Content-Type", "application/json")
            .json(&request_body)
            .send()
            .await
            .map_err(|e| AleError::CloudApiError(format!("Request failed: {}", e)))?
            .error_for_status_with_body()
            .await?;
        let content = Self::content_text(&response_body)?;
        Ok(CloudResponse {
            content,
            tokens_used: google_tokens(&response_body),
            model: self.config.model.clone(),
            provider: self.config.provider.clone(),
        })
    }

    async fn vision(&self, image_data: &[u8], prompt: &str) -> Result<CloudResponse> {
        let response = self.vision_ask(image_data, prompt, None).await?;
        Ok(CloudResponse {
            content: response.content,
            tokens_used: response.tokens_used,
            model: response.model,
            provider: self.config.provider.clone(),
        })
    }

    async fn vision_ask(
        &self,
        image_data: &[u8],
        question: &str,
        tools: Option<Vec<serde_json::Value>>,
    ) -> Result<VisionResponse> {
        let image_base64 = general_purpose::STANDARD.encode(image_data);
        let mut request_body = serde_json::json!({
            "systemInstruction": {"parts": [{"text": "你是 Ale, My Eyes! 智能视觉辅助助手。用户会发送一张图片和一个问题，请根据图片内容回答问题。如果用户要求执行操作，请使用提供的工具函数。"}]},
            "contents": [{
                "role": "user",
                "parts": [
                    {"text": question},
                    {"inline_data": {"mime_type": "image/jpeg", "data": image_base64}}
                ]
            }],
            "generationConfig": {"maxOutputTokens": self.config.max_tokens},
        });
        if let Some(tools) = tools {
            let tools = Self::convert_tools(tools);
            if !tools.is_empty() {
                request_body["tools"] = serde_json::json!(tools);
            }
        }

        let response_body = self
            .request(reqwest::Method::POST, self.generate_url())
            .header("Content-Type", "application/json")
            .json(&request_body)
            .send()
            .await
            .map_err(|e| AleError::CloudApiError(format!("Request failed: {}", e)))?
            .error_for_status_with_body()
            .await?;
        let content = Self::content_text(&response_body).unwrap_or_default();
        let tool_calls = Self::tool_calls(&response_body);
        if content.trim().is_empty() && tool_calls.as_ref().map(|c| c.is_empty()).unwrap_or(true) {
            return Err(AleError::CloudApiError(
                "Missing Google vision response content or tool calls".to_string(),
            ));
        }
        Ok(VisionResponse {
            content,
            tool_calls,
            tokens_used: google_tokens(&response_body),
            model: self.config.model.clone(),
        })
    }

    async fn transcribe(&self, _audio_data: &[u8]) -> Result<CloudResponse> {
        unsupported_provider_feature(&self.config.provider, "speech transcription")
    }

    async fn synthesize(&self, _text: &str) -> Result<Vec<u8>> {
        unsupported_provider_feature(&self.config.provider, "speech synthesis")
    }

    async fn health_check(&self) -> Result<bool> {
        let url = format!(
            "{}/models/{}",
            self.config.api_url.trim_end_matches('/'),
            self.config.model
        );
        let response = self
            .request(reqwest::Method::GET, url)
            .send()
            .await
            .map_err(|e| AleError::CloudApiError(format!("Health check failed: {}", e)))?;
        Ok(response.status().is_success())
    }
}

struct AzureOpenAIApi {
    config: CloudConfig,
    client: reqwest::Client,
}

impl AzureOpenAIApi {
    fn new(config: CloudConfig) -> Self {
        Self {
            client: client_for(&config),
            config,
        }
    }

    fn chat_url(&self) -> String {
        let base = self.config.api_url.trim_end_matches('/');
        if base.contains("/chat/completions") {
            append_api_version(base)
        } else {
            append_api_version(&format!(
                "{}/openai/deployments/{}/chat/completions",
                base, self.config.model
            ))
        }
    }

    fn deployments_url(&self) -> String {
        let base = self.config.api_url.trim_end_matches('/');
        if let Some((resource_url, _)) = base.split_once("/openai/deployments/") {
            append_api_version(&format!("{}/openai/deployments", resource_url))
        } else {
            append_api_version(&format!("{}/openai/deployments", base))
        }
    }

    fn openai_compatible_response(
        &self,
        response_body: &serde_json::Value,
    ) -> Result<CloudResponse> {
        Ok(CloudResponse {
            content: OpenAIApi::chat_content(response_body)?,
            tokens_used: response_body["usage"]["total_tokens"].as_u64().unwrap_or(0) as usize,
            model: self.config.model.clone(),
            provider: self.config.provider.clone(),
        })
    }
}

#[async_trait]
impl CloudApi for AzureOpenAIApi {
    async fn chat(&self, messages: Vec<CloudMessage>) -> Result<CloudResponse> {
        let request_body = serde_json::json!({
            "messages": messages,
            "max_tokens": self.config.max_tokens,
        });
        let response_body = self
            .client
            .post(self.chat_url())
            .header("api-key", &self.config.api_key)
            .header("Content-Type", "application/json")
            .json(&request_body)
            .send()
            .await
            .map_err(|e| AleError::CloudApiError(format!("Request failed: {}", e)))?
            .error_for_status_with_body()
            .await?;
        self.openai_compatible_response(&response_body)
    }

    async fn vision(&self, image_data: &[u8], prompt: &str) -> Result<CloudResponse> {
        let response = self.vision_ask(image_data, prompt, None).await?;
        Ok(CloudResponse {
            content: response.content,
            tokens_used: response.tokens_used,
            model: response.model,
            provider: self.config.provider.clone(),
        })
    }

    async fn vision_ask(
        &self,
        image_data: &[u8],
        question: &str,
        tools: Option<Vec<serde_json::Value>>,
    ) -> Result<VisionResponse> {
        let image_base64 = general_purpose::STANDARD.encode(image_data);
        let mut request_body = serde_json::json!({
            "messages": [
                {"role": "system", "content": "你是 Ale, My Eyes! 智能视觉辅助助手。用户会发送一张图片和一个问题，请根据图片内容回答问题。如果用户要求执行操作，请使用提供的工具函数。"},
                {"role": "user", "content": [
                    {"type": "text", "text": question},
                    {"type": "image_url", "image_url": {"url": format!("data:image/jpeg;base64,{}", image_base64)}}
                ]}
            ],
            "max_tokens": self.config.max_tokens,
        });
        if let Some(tools) = tools {
            request_body["tools"] = serde_json::json!(tools);
        }
        let response_body = self
            .client
            .post(self.chat_url())
            .header("api-key", &self.config.api_key)
            .header("Content-Type", "application/json")
            .json(&request_body)
            .send()
            .await
            .map_err(|e| AleError::CloudApiError(format!("Request failed: {}", e)))?
            .error_for_status_with_body()
            .await?;

        let message = &response_body["choices"][0]["message"];
        let content = message["content"].as_str().unwrap_or_default().to_string();
        let tool_calls = parse_openai_tool_calls(message);
        if content.trim().is_empty() && tool_calls.as_ref().map(|c| c.is_empty()).unwrap_or(true) {
            return Err(AleError::CloudApiError(
                "Missing Azure vision response content or tool calls".to_string(),
            ));
        }

        Ok(VisionResponse {
            content,
            tool_calls,
            tokens_used: response_body["usage"]["total_tokens"].as_u64().unwrap_or(0) as usize,
            model: self.config.model.clone(),
        })
    }

    async fn transcribe(&self, _audio_data: &[u8]) -> Result<CloudResponse> {
        unsupported_provider_feature(&self.config.provider, "speech transcription")
    }

    async fn synthesize(&self, _text: &str) -> Result<Vec<u8>> {
        unsupported_provider_feature(&self.config.provider, "speech synthesis")
    }

    async fn health_check(&self) -> Result<bool> {
        let url = self.deployments_url();
        let response = self
            .client
            .get(url)
            .header("api-key", &self.config.api_key)
            .send()
            .await
            .map_err(|e| AleError::CloudApiError(format!("Health check failed: {}", e)))?;
        Ok(response.status().is_success())
    }
}

trait ResponseExt {
    async fn error_for_status_with_body(self) -> Result<serde_json::Value>;
}

impl ResponseExt for reqwest::Response {
    async fn error_for_status_with_body(self) -> Result<serde_json::Value> {
        if !self.status().is_success() {
            let error_text = self.text().await.unwrap_or_default();
            return Err(AleError::CloudApiError(format!(
                "API error: {}",
                error_text
            )));
        }
        self.json()
            .await
            .map_err(|e| AleError::CloudApiError(format!("Parse error: {}", e)))
    }
}

fn client_for(config: &CloudConfig) -> reqwest::Client {
    reqwest::Client::builder()
        .timeout(config.timeout)
        .build()
        .unwrap_or_else(|_| reqwest::Client::new())
}

fn unsupported_provider_feature<T>(provider: &CloudProvider, feature: &str) -> Result<T> {
    Err(AleError::CloudApiError(format!(
        "Provider {:?} does not support {} in this app yet",
        provider, feature
    )))
}

fn google_tokens(response_body: &serde_json::Value) -> usize {
    response_body["usageMetadata"]["totalTokenCount"]
        .as_u64()
        .unwrap_or(0) as usize
}

fn append_api_version(url: &str) -> String {
    if url.contains("api-version=") {
        url.to_string()
    } else if url.contains('?') {
        format!("{}&api-version=2024-02-15-preview", url)
    } else {
        format!("{}?api-version=2024-02-15-preview", url)
    }
}

fn parse_openai_tool_calls(message: &serde_json::Value) -> Option<Vec<ToolCall>> {
    let calls = message["tool_calls"]
        .as_array()?
        .iter()
        .map(|tc| ToolCall {
            id: tc["id"].as_str().unwrap_or_default().to_string(),
            function: FunctionCall {
                name: tc["function"]["name"]
                    .as_str()
                    .unwrap_or_default()
                    .to_string(),
                arguments: tc["function"]["arguments"]
                    .as_str()
                    .unwrap_or_default()
                    .to_string(),
            },
        })
        .collect::<Vec<_>>();
    if calls.is_empty() {
        None
    } else {
        Some(calls)
    }
}

/// 云端API工厂
pub struct CloudApiFactory;

impl CloudApiFactory {
    pub fn create(config: CloudConfig) -> Box<dyn CloudApi> {
        match config.provider.clone() {
            CloudProvider::OpenAI => Box::new(OpenAIApi::new(config)),
            CloudProvider::Anthropic => Box::new(AnthropicApi::new(config)),
            CloudProvider::Google => Box::new(GoogleApi::new(config)),
            CloudProvider::Azure => Box::new(AzureOpenAIApi::new(config)),
            CloudProvider::Custom(_) => Box::new(OpenAIApi::new(config)),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_cloud_config_default() {
        let config = CloudConfig::default();
        assert_eq!(config.api_url, "https://api.openai.com/v1");
        assert_eq!(config.model, "gpt-4o");
        assert_eq!(config.max_tokens, 1024);
        assert_eq!(config.timeout, Duration::from_secs(30));
    }

    #[test]
    fn test_cloud_provider_serialization() {
        let provider = CloudProvider::OpenAI;
        let json = serde_json::to_string(&provider).unwrap();
        let restored: CloudProvider = serde_json::from_str(&json).unwrap();
        assert!(matches!(restored, CloudProvider::OpenAI));
    }

    #[test]
    fn test_cloud_api_factory_creates_openai() {
        let config = CloudConfig {
            provider: CloudProvider::OpenAI,
            api_key: "test".to_string(),
            ..Default::default()
        };
        let _api = CloudApiFactory::create(config);
    }

    #[test]
    fn test_cloud_api_factory_custom_provider() {
        let config = CloudConfig {
            provider: CloudProvider::Custom("test".to_string()),
            api_key: "test".to_string(),
            ..Default::default()
        };
        let _api = CloudApiFactory::create(config);
    }

    #[test]
    fn test_append_api_version() {
        assert_eq!(
            append_api_version("https://example.openai.azure.com/openai/deployments/gpt/chat/completions"),
            "https://example.openai.azure.com/openai/deployments/gpt/chat/completions?api-version=2024-02-15-preview"
        );
        assert_eq!(
            append_api_version("https://example.openai.azure.com/path?foo=bar"),
            "https://example.openai.azure.com/path?foo=bar&api-version=2024-02-15-preview"
        );
        assert_eq!(
            append_api_version("https://example.openai.azure.com/path?api-version=custom"),
            "https://example.openai.azure.com/path?api-version=custom"
        );
    }

    #[test]
    fn test_azure_deployments_url_accepts_chat_url() {
        let api = AzureOpenAIApi::new(CloudConfig {
            provider: CloudProvider::Azure,
            api_url: "https://example.openai.azure.com/openai/deployments/gpt/chat/completions"
                .to_string(),
            model: "gpt".to_string(),
            ..Default::default()
        });

        assert_eq!(
            api.deployments_url(),
            "https://example.openai.azure.com/openai/deployments?api-version=2024-02-15-preview"
        );
    }

    #[test]
    fn test_parse_openai_tool_calls() {
        let message = serde_json::json!({
            "tool_calls": [{
                "id": "call-1",
                "function": {
                    "name": "execute_action_plan",
                    "arguments": "{\"actions\":[]}"
                }
            }]
        });

        let calls = parse_openai_tool_calls(&message).unwrap();
        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0].id, "call-1");
        assert_eq!(calls[0].function.name, "execute_action_plan");
    }

    #[test]
    fn test_google_content_text_accepts_parts() {
        let response = serde_json::json!({
            "candidates": [{"content": {"parts": [{"text": "hello"}, {"text": "world"}]}}]
        });

        assert_eq!(GoogleApi::content_text(&response).unwrap(), "hello\nworld");
    }

    #[test]
    fn test_google_chat_request_does_not_put_api_key_in_url() {
        let api = GoogleApi::new(CloudConfig {
            provider: CloudProvider::Google,
            api_key: "AIzaSyDUMMY_TEST_KEY_1234567890".to_string(),
            api_url: "https://generativelanguage.googleapis.com/v1beta".to_string(),
            model: "gemini-1.5-pro".to_string(),
            ..Default::default()
        });

        let request = api
            .request(reqwest::Method::POST, api.generate_url())
            .header("Content-Type", "application/json")
            .json(&serde_json::json!({
                "contents": [],
                "generationConfig": {"maxOutputTokens": api.config.max_tokens},
            }))
            .build()
            .unwrap();

        assert!(
            !request
                .url()
                .as_str()
                .contains("AIzaSyDUMMY_TEST_KEY_1234567890"),
            "Google request URL leaked the API key"
        );
    }

    #[test]
    fn test_google_chat_request_sets_google_api_key_header() {
        let api = GoogleApi::new(CloudConfig {
            provider: CloudProvider::Google,
            api_key: "AIzaSyDUMMY_TEST_KEY_1234567890".to_string(),
            api_url: "https://generativelanguage.googleapis.com/v1beta".to_string(),
            model: "gemini-1.5-pro".to_string(),
            ..Default::default()
        });

        let request = api
            .request(reqwest::Method::POST, api.generate_url())
            .header("Content-Type", "application/json")
            .json(&serde_json::json!({
                "contents": [],
                "generationConfig": {"maxOutputTokens": api.config.max_tokens},
            }))
            .build()
            .unwrap();

        assert!(
            request.headers().get("x-goog-api-key").is_some(),
            "Google request is missing the x-goog-api-key header"
        );
    }

    #[tokio::test]
    async fn google_authentication_stays_in_headers_for_all_requests() {
        use tokio::io::{AsyncReadExt, AsyncWriteExt};
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let address = listener.local_addr().unwrap();
        let server = tokio::spawn(async move {
            for method in ["GET", "POST", "POST"] {
                let (mut socket, _) = listener.accept().await.unwrap();
                let mut bytes = Vec::new();
                let mut chunk = [0; 4096];
                loop {
                    let n = socket.read(&mut chunk).await.unwrap();
                    assert!(n > 0);
                    bytes.extend_from_slice(&chunk[..n]);
                    if bytes.windows(4).any(|w| w == b"\r\n\r\n") {
                        break;
                    }
                }
                let request = String::from_utf8_lossy(&bytes);
                let first_line = request.lines().next().unwrap();
                assert!(first_line.starts_with(method));
                assert!(!first_line.contains("review-test-key"));
                assert!(!first_line.contains("key="));
                assert!(request
                    .to_lowercase()
                    .contains("x-goog-api-key: review-test-key\r\n"));
                let body = r#"{"candidates":[{"content":{"parts":[{"text":"ok"}]}}]}"#;
                let response = format!(
                    "HTTP/1.1 200 OK\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{}",
                    body.len(),
                    body
                );
                socket.write_all(response.as_bytes()).await.unwrap();
            }
        });
        let api = GoogleApi::new(CloudConfig {
            provider: CloudProvider::Google,
            api_url: format!("http://{address}"),
            api_key: "review-test-key".into(),
            model: "gemini".into(),
            ..Default::default()
        });
        tokio::time::timeout(std::time::Duration::from_secs(10), async {
            assert!(api.health_check().await.unwrap());
            assert_eq!(api.chat(Vec::new()).await.unwrap().content, "ok");
            assert_eq!(
                api.vision(b"test-image", "describe").await.unwrap().content,
                "ok"
            );
            server.await.unwrap();
        })
        .await
        .unwrap();
    }

    #[test]
    fn test_anthropic_content_text_accepts_text_blocks() {
        let response = serde_json::json!({
            "content": [
                {"type": "text", "text": "hello"},
                {"type": "tool_use", "name": "ignored"}
            ]
        });

        assert_eq!(AnthropicApi::content_text(&response).unwrap(), "hello");
    }

    #[test]
    fn test_chat_content_rejects_missing_content() {
        let response = serde_json::json!({"choices": [{"message": {}}]});
        assert!(OpenAIApi::chat_content(&response).is_err());
    }

    #[test]
    fn test_chat_content_rejects_empty_content() {
        let response = serde_json::json!({"choices": [{"message": {"content": "  "}}]});
        assert!(OpenAIApi::chat_content(&response).is_err());
    }

    #[test]
    fn test_chat_content_accepts_content() {
        let response = serde_json::json!({"choices": [{"message": {"content": "hello"}}]});
        assert_eq!(OpenAIApi::chat_content(&response).unwrap(), "hello");
    }

    #[test]
    fn test_transcription_text_rejects_missing_text() {
        let response = serde_json::json!({});
        assert!(OpenAIApi::transcription_text(&response).is_err());
    }
}
