use crate::{AleError, Result};
use async_trait::async_trait;

/// 语音合成trait
#[async_trait]
pub trait TextToSpeech: Send + Sync {
    /// 合成语音
    async fn synthesize(&self, text: &str) -> Result<Vec<u8>>;

    /// 流式合成语音
    async fn synthesize_stream(&self, text: &str) -> Result<Box<dyn tokio::io::AsyncRead + Unpin>>;

    /// 获取可用语音列表
    fn available_voices(&self) -> Vec<String>;

    fn model_info(&self) -> crate::RuntimeModelInfo;
}

/// 系统TTS引擎
pub struct SystemTts {
    voice: Option<String>,
    tts_engine: Option<tts::Tts>,
}

impl SystemTts {
    pub async fn new(voice: Option<&str>) -> Result<Self> {
        let voice = voice.map(|v| v.to_string());

        Ok(Self {
            voice,
            tts_engine: None,
        })
    }

    fn init_engine(&mut self) -> Result<()> {
        if self.tts_engine.is_some() {
            return Ok(());
        }

        let mut tts = tts::Tts::default()
            .map_err(|e| AleError::TtsError(format!("Failed to initialize TTS: {}", e)))?;

        // 设置语音
        if let Some(voice_name) = &self.voice {
            let voices = tts
                .voices()
                .map_err(|e| AleError::TtsError(format!("Failed to get voices: {}", e)))?;

            for voice in voices {
                if voice.name() == *voice_name {
                    tts.set_voice(&voice)
                        .map_err(|e| AleError::TtsError(format!("Failed to set voice: {}", e)))?;
                    break;
                }
            }
        }

        self.tts_engine = Some(tts);
        Ok(())
    }
}

#[async_trait]
impl TextToSpeech for SystemTts {
    async fn synthesize(&self, text: &str) -> Result<Vec<u8>> {
        // 这里需要实现实际的语音合成逻辑
        // 由于tts crate的API限制，这里简化处理
        Err(AleError::TtsError("Not implemented yet".to_string()))
    }

    async fn synthesize_stream(&self, text: &str) -> Result<Box<dyn tokio::io::AsyncRead + Unpin>> {
        // 流式合成
        Err(AleError::TtsError(
            "Streaming not implemented yet".to_string(),
        ))
    }

    fn available_voices(&self) -> Vec<String> {
        // 返回可用语音列表
        vec![
            "default".to_string(),
            "male".to_string(),
            "female".to_string(),
        ]
    }

    fn model_info(&self) -> crate::RuntimeModelInfo {
        crate::RuntimeModelInfo {
            name: "system-tts".to_string(),
            version: "1.0".to_string(),
            device: "cpu".to_string(),
            loaded: self.tts_engine.is_some(),
        }
    }
}
