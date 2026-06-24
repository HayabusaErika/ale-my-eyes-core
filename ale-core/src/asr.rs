use crate::{AleError, Result};
use async_trait::async_trait;
use std::path::Path;
use whisper_rs::{FullParams, SamplingStrategy, WhisperContext, WhisperContextParameters};

const WHISPER_SAMPLE_RATE: u32 = 16000;

/// Speech recognition trait
#[async_trait]
pub trait SpeechRecognizer: Send + Sync {
    async fn transcribe(&self, audio_data: &[u8]) -> Result<String>;
    fn supported_languages(&self) -> Vec<String>;
    fn model_info(&self) -> crate::RuntimeModelInfo;
}

/// Whisper speech recognizer backed by whisper-rs (whisper.cpp FFI)
pub struct WhisperRecognizer {
    model_path: std::path::PathBuf,
    ctx: Option<WhisperContext>,
    language: Option<String>,
    n_threads: i32,
}

impl WhisperRecognizer {
    pub async fn new(model_path: &Path) -> Result<Self> {
        if !model_path.exists() {
            return Err(AleError::AsrError(format!(
                "Model file not found: {}",
                model_path.display()
            )));
        }

        Ok(Self {
            model_path: model_path.to_path_buf(),
            ctx: None,
            language: None,
            n_threads: num_cpus().min(8) as i32,
        })
    }

    pub fn with_language(mut self, lang: Option<String>) -> Self {
        self.language = lang;
        self
    }

    pub fn with_threads(mut self, n: i32) -> Self {
        self.n_threads = n;
        self
    }

    pub fn load_model(&mut self) -> Result<()> {
        if self.ctx.is_some() {
            return Ok(());
        }

        let ctx = WhisperContext::new_with_params(
            self.model_path
                .to_str()
                .ok_or_else(|| AleError::AsrError("Model path is not valid UTF-8".to_string()))?,
            WhisperContextParameters::default(),
        )
        .map_err(|e| AleError::AsrError(format!("Failed to load whisper model: {e}")))?;

        self.ctx = Some(ctx);
        Ok(())
    }

    pub fn is_loaded(&self) -> bool {
        self.ctx.is_some()
    }

    fn run_inference(&self, ctx: &WhisperContext, samples: &[f32]) -> Result<String> {
        let mut state = ctx
            .create_state()
            .map_err(|e| AleError::AsrError(format!("Failed to create whisper state: {e}")))?;

        let mut params = FullParams::new(SamplingStrategy::Greedy { best_of: 1 });
        params.set_n_threads(self.n_threads);
        params.set_print_special(false);
        params.set_print_progress(false);
        params.set_print_realtime(false);
        params.set_print_timestamps(false);
        params.set_suppress_blank(true);
        params.set_suppress_nst(true);

        if let Some(ref lang) = self.language {
            params.set_language(Some(lang));
        } else {
            params.set_language(Some("auto"));
        }

        state
            .full(params, samples)
            .map_err(|e| AleError::AsrError(format!("Whisper inference failed: {e}")))?;

        let num_segments = state.full_n_segments();

        let mut text = String::new();
        for i in 0..num_segments {
            if let Some(segment) = state.get_segment(i) {
                if let Ok(segment_text) = segment.to_str() {
                    text.push_str(segment_text);
                }
            }
        }

        Ok(text.trim().to_string())
    }
}

#[async_trait]
impl SpeechRecognizer for WhisperRecognizer {
    async fn transcribe(&self, audio_data: &[u8]) -> Result<String> {
        let samples = parse_audio_to_f32_mono(audio_data, WHISPER_SAMPLE_RATE)?;

        if samples.is_empty() {
            return Err(AleError::AsrError("No audio samples found".to_string()));
        }

        let ctx = self.ctx.as_ref().ok_or_else(|| {
            AleError::AsrError("Whisper model not loaded, call load_model() first".to_string())
        })?;

        self.run_inference(ctx, &samples)
    }

    fn supported_languages(&self) -> Vec<String> {
        vec![
            "auto".into(),
            "en".into(),
            "zh".into(),
            "ja".into(),
            "ko".into(),
            "fr".into(),
            "de".into(),
            "es".into(),
            "ru".into(),
            "pt".into(),
            "it".into(),
            "nl".into(),
            "pl".into(),
            "ar".into(),
            "tr".into(),
            "vi".into(),
            "th".into(),
        ]
    }

    fn model_info(&self) -> crate::RuntimeModelInfo {
        crate::RuntimeModelInfo {
            name: "whisper".to_string(),
            version: format!("cpp-{}", whisper_rs::WHISPER_CPP_VERSION),
            device: "cpu".to_string(),
            loaded: self.ctx.is_some(),
        }
    }
}

// ── Audio parsing ──────────────────────────────────────────────

/// Parse raw audio bytes (WAV or raw PCM) into f32 mono samples at the target sample rate.
pub fn parse_audio_to_f32_mono(audio_data: &[u8], target_rate: u32) -> Result<Vec<f32>> {
    if audio_data.len() < 44 {
        return Err(AleError::AsrError("Audio data too short".to_string()));
    }

    // Check for WAV header
    if &audio_data[0..4] == b"RIFF" && &audio_data[8..12] == b"WAVE" {
        parse_wav(audio_data, target_rate)
    } else {
        // Assume raw 16-bit PCM mono at target rate
        Ok(pcm16_to_f32(audio_data))
    }
}

fn parse_wav(data: &[u8], target_rate: u32) -> Result<Vec<f32>> {
    let num_channels = u16::from_le_bytes([data[22], data[23]]) as u32;
    let sample_rate = u32::from_le_bytes([data[24], data[25], data[26], data[27]]);
    let bits_per_sample = u16::from_le_bytes([data[34], data[35]]) as u32;

    // Find data chunk
    let mut offset = 36;
    while offset + 8 <= data.len() {
        let chunk_id = &data[offset..offset + 4];
        let chunk_size = u32::from_le_bytes([
            data[offset + 4],
            data[offset + 5],
            data[offset + 6],
            data[offset + 7],
        ]) as usize;

        if chunk_id == b"data" {
            let audio_start = offset + 8;
            let audio_end = audio_start + chunk_size.min(data.len() - audio_start);
            let raw_audio = &data[audio_start..audio_end];

            let samples = match bits_per_sample {
                16 => pcm16_to_f32(raw_audio),
                32 => pcm32f_to_f32(raw_audio),
                _ => {
                    return Err(AleError::AsrError(format!(
                        "Unsupported WAV bit depth: {bits_per_sample}"
                    )))
                }
            };

            let mono = if num_channels >= 2 {
                stereo_to_mono(&samples)
            } else {
                samples
            };

            if sample_rate != target_rate {
                return Ok(resample(&mono, sample_rate, target_rate));
            }
            return Ok(mono);
        }

        offset += 8 + chunk_size;
        if chunk_size % 2 != 0 {
            offset += 1;
        }
    }

    Err(AleError::AsrError("WAV data chunk not found".to_string()))
}

fn pcm16_to_f32(data: &[u8]) -> Vec<f32> {
    data.chunks_exact(2)
        .map(|chunk| {
            let sample = i16::from_le_bytes([chunk[0], chunk[1]]);
            sample as f32 / 32768.0
        })
        .collect()
}

fn pcm32f_to_f32(data: &[u8]) -> Vec<f32> {
    data.chunks_exact(4)
        .map(|chunk| f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]))
        .collect()
}

fn stereo_to_mono(samples: &[f32]) -> Vec<f32> {
    samples
        .chunks_exact(2)
        .map(|pair| (pair[0] + pair[1]) / 2.0)
        .collect()
}

fn resample(samples: &[f32], from_rate: u32, to_rate: u32) -> Vec<f32> {
    if from_rate == to_rate {
        return samples.to_vec();
    }

    let ratio = from_rate as f64 / to_rate as f64;
    let output_len = (samples.len() as f64 / ratio) as usize;
    let mut output = Vec::with_capacity(output_len);

    for i in 0..output_len {
        let src_idx = i as f64 * ratio;
        let idx = src_idx as usize;
        let frac = src_idx - idx as f64;

        let sample = if idx + 1 < samples.len() {
            samples[idx] as f64 * (1.0 - frac) + samples[idx + 1] as f64 * frac
        } else if idx < samples.len() {
            samples[idx] as f64
        } else {
            0.0
        };

        output.push(sample as f32);
    }

    output
}

fn num_cpus() -> usize {
    std::thread::available_parallelism()
        .map(|n| n.get())
        .unwrap_or(4)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_stereo_to_mono() {
        let stereo = vec![0.5, -0.5, 1.0, -1.0];
        let mono = stereo_to_mono(&stereo);
        assert_eq!(mono.len(), 2);
        assert!((mono[0] - 0.0).abs() < 1e-6);
        assert!((mono[1] - 0.0).abs() < 1e-6);
    }

    #[test]
    fn test_pcm16_to_f32() {
        let bytes = (i16::MAX as i16).to_le_bytes();
        let samples = pcm16_to_f32(&bytes);
        assert!((samples[0] - 32767.0 / 32768.0).abs() < 1e-6);
    }

    #[test]
    fn test_resample_same_rate() {
        let samples = vec![1.0, 2.0, 3.0];
        let result = resample(&samples, 16000, 16000);
        assert_eq!(result, samples);
    }

    #[test]
    fn test_resample_downsample() {
        let samples: Vec<f32> = (0..48000).map(|i| (i as f32).sin()).collect();
        let result = resample(&samples, 48000, 16000);
        assert_eq!(result.len(), 16000);
    }
}
