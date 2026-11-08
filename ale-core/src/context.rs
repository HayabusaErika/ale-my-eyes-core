use crate::cloud::CloudMessage;
use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use std::collections::{HashSet, VecDeque};
use uuid::Uuid;

/// 上下文条目角色
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum Role {
    User,
    Assistant,
    System,
}

/// 上下文条目
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ContextEntry {
    pub role: Role,
    pub content: String,
    pub image_ref: Option<String>,
    pub token_count: usize,
}

/// 帧摘要（存储 AI 生成的描述，而非原始图像）
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FrameSummary {
    pub description: String,
    pub key_elements: Vec<String>,
    pub source: FrameSource,
}

/// 帧来源
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum FrameSource {
    Camera,
    Screen,
}

/// 长期记忆条目
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MemoryEntry {
    #[serde(default = "MemoryEntry::new_id")]
    pub id: String,
    pub content: String,
    pub importance: f32,
    pub source: String,
    #[serde(default = "Utc::now")]
    pub created_at: DateTime<Utc>,
    #[serde(default)]
    pub last_used_at: Option<DateTime<Utc>>,
    #[serde(default)]
    pub tags: Vec<String>,
}

impl MemoryEntry {
    pub fn new(content: String, importance: f32, source: String) -> Self {
        Self {
            id: Self::new_id(),
            content,
            importance,
            source,
            created_at: Utc::now(),
            last_used_at: None,
            tags: Vec::new(),
        }
    }

    fn new_id() -> String {
        Uuid::new_v4().to_string()
    }
}

pub struct ContextManager {
    conversation: VecDeque<ContextEntry>,
    visual_memory: VecDeque<FrameSummary>,
    max_visual: usize,
    long_term_memory: Vec<MemoryEntry>,
    max_long_term: usize,
    conversation_summary: Option<String>,
    current_tokens: usize,
    max_tokens: usize,
    session_tokens_used: usize,
    system_prompt: String,
}

impl ContextManager {
    pub fn new(max_tokens: usize) -> Self {
        Self {
            conversation: VecDeque::new(),
            visual_memory: VecDeque::new(),
            max_visual: 5,
            long_term_memory: Vec::new(),
            max_long_term: 100,
            conversation_summary: None,
            current_tokens: 0,
            max_tokens,
            session_tokens_used: 0,
            system_prompt: Self::default_system_prompt(),
        }
    }

    pub fn with_system_prompt(mut self, prompt: String) -> Self {
        self.system_prompt = prompt;
        self
    }

    fn default_system_prompt() -> String {
        "你是 Ale, My Eyes! 智能视觉辅助助手。你可以看到用户的摄像头画面或电脑屏幕，帮助用户理解视觉内容并执行操作。请用简洁自然的中文回答。".to_string()
    }

    /// 添加用户消息
    pub fn add_user_message(&mut self, content: String) {
        let token_count = self.estimate_tokens(&content);
        self.conversation.push_back(ContextEntry {
            role: Role::User,
            content,
            image_ref: None,
            token_count,
        });
        self.current_tokens += token_count;
        self.maybe_compact();
    }

    /// 添加助手消息
    pub fn add_assistant_message(&mut self, content: String) {
        let token_count = self.estimate_tokens(&content);
        self.conversation.push_back(ContextEntry {
            role: Role::Assistant,
            content,
            image_ref: None,
            token_count,
        });
        self.current_tokens += token_count;
        self.maybe_compact();
    }

    /// 添加带图像引用的用户消息
    pub fn add_user_message_with_image(&mut self, content: String, image_ref: String) {
        let token_count = self.estimate_tokens(&content) + 100; // 图像描述的 token 估算
        self.conversation.push_back(ContextEntry {
            role: Role::User,
            content,
            image_ref: Some(image_ref),
            token_count,
        });
        self.current_tokens += token_count;
        self.maybe_compact();
    }

    /// 添加视觉帧摘要
    pub fn add_frame_summary(&mut self, summary: FrameSummary) {
        // 检查是否与最近帧重复
        if let Some(last) = self.visual_memory.back() {
            if last.description == summary.description {
                return;
            }
        }

        self.visual_memory.push_back(summary);
        if self.visual_memory.len() > self.max_visual {
            self.visual_memory.pop_front();
        }
    }

    pub fn add_memory(&mut self, entry: MemoryEntry) {
        let is_duplicate = self.long_term_memory.iter().any(|m| {
            m.content == entry.content || jaccard_similarity(&m.content, &entry.content) > 0.7
        });
        if is_duplicate {
            return;
        }
        self.long_term_memory.push(entry);
        if self.long_term_memory.len() > self.max_long_term {
            let min_idx = self
                .long_term_memory
                .iter()
                .enumerate()
                .min_by(|a, b| {
                    a.1.importance
                        .partial_cmp(&b.1.importance)
                        .unwrap_or(std::cmp::Ordering::Equal)
                })
                .map(|(i, _)| i);
            if let Some(idx) = min_idx {
                self.long_term_memory.swap_remove(idx);
            }
        }
    }

    /// 批量替换长期记忆，通常用于加载持久化记忆。
    pub fn replace_memories(&mut self, memories: Vec<MemoryEntry>) {
        self.long_term_memory = memories;
    }

    /// 获取所有长期记忆。
    pub fn memories(&self) -> &[MemoryEntry] {
        &self.long_term_memory
    }

    pub fn relevant_memories(&self, query: &str, limit: usize) -> Vec<&MemoryEntry> {
        let query_terms = extract_terms(query);
        let mut scored: Vec<(&MemoryEntry, f32)> = self
            .long_term_memory
            .iter()
            .map(|memory| {
                let content_lower = memory.content.to_lowercase();
                let tag_lower = memory.tags.join(" ").to_lowercase();
                let source_lower = memory.source.to_lowercase();

                let term_score = query_terms
                    .iter()
                    .filter(|term| {
                        contains_ci(&content_lower, term)
                            || contains_ci(&tag_lower, term)
                            || contains_ci(&source_lower, term)
                    })
                    .count() as f32;
                let tag_score = memory.tags.len().min(5) as f32 * 0.05;
                let used_score = if memory.last_used_at.is_some() {
                    0.1
                } else {
                    0.0
                };

                (
                    memory,
                    term_score * 2.0 + memory.importance + tag_score + used_score,
                )
            })
            .collect();

        scored.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        scored
            .into_iter()
            .filter(|(_, score)| *score > 0.0)
            .take(limit)
            .map(|(memory, _)| memory)
            .collect()
    }

    pub fn build_messages(
        &self,
        current_image_description: Option<&str>,
        current_question: &str,
    ) -> Vec<CloudMessage> {
        let mut messages = Vec::with_capacity(16);
        let mut total_tokens = 0usize;
        let budget = self.max_tokens;

        let memories = self.relevant_memories(current_question, 8);
        let system = if memories.is_empty() {
            self.system_prompt.clone()
        } else {
            let mut s = String::with_capacity(
                self.system_prompt.len()
                    + 32
                    + memories.iter().map(|m| m.content.len() + 4).sum::<usize>(),
            );
            s.push_str(&self.system_prompt);
            s.push_str("\n\n用户相关记忆：\n");
            for mem in memories {
                s.push_str("- ");
                s.push_str(&mem.content);
                s.push('\n');
            }
            s
        };
        total_tokens += self.estimate_tokens(&system);
        messages.push(CloudMessage {
            role: "system".to_string(),
            content: system,
        });

        if !self.visual_memory.is_empty() {
            let last_two: Vec<_> = self.visual_memory.iter().rev().take(2).collect();
            let visual_len: usize = last_two
                .iter()
                .map(|f| {
                    f.description.len()
                        + f.key_elements.iter().map(|k| k.len() + 2).sum::<usize>()
                        + 32
                })
                .sum();
            let mut visual_context = String::with_capacity(32 + visual_len);
            visual_context.push_str("最近的画面变化：\n");
            for (i, frame) in last_two.iter().rev().enumerate() {
                let source_label = match frame.source {
                    FrameSource::Camera => "相机",
                    FrameSource::Screen => "屏幕",
                };
                visual_context.push_str(&format!(
                    "{}. [{}] {}\n",
                    i + 1,
                    source_label,
                    frame.description
                ));
                if !frame.key_elements.is_empty() {
                    visual_context.push_str("   关键元素: ");
                    visual_context.push_str(&frame.key_elements.join(", "));
                    visual_context.push('\n');
                }
            }
            total_tokens += self.estimate_tokens(&visual_context);
            messages.push(CloudMessage {
                role: "system".to_string(),
                content: visual_context,
            });
        }

        if let Some(ref summary) = self.conversation_summary {
            let mut content = String::with_capacity(20 + summary.len());
            content.push_str("之前的对话摘要：");
            content.push_str(summary);
            total_tokens += self.estimate_tokens(&content);
            messages.push(CloudMessage {
                role: "system".to_string(),
                content,
            });
        }

        let recent_start = self.conversation.len().saturating_sub(10);
        for entry in self.conversation.iter().skip(recent_start) {
            let entry_tokens = entry.token_count;
            if total_tokens + entry_tokens > budget * 90 / 100 {
                break;
            }
            let role = match entry.role {
                Role::User => "user",
                Role::Assistant => "assistant",
                Role::System => "system",
            };
            total_tokens += entry_tokens;
            messages.push(CloudMessage {
                role: role.to_string(),
                content: entry.content.clone(),
            });
        }

        let user_content = if let Some(img_desc) = current_image_description {
            format!("[当前画面：{}]\n\n{}", img_desc, current_question)
        } else {
            current_question.to_string()
        };
        messages.push(CloudMessage {
            role: "user".to_string(),
            content: user_content,
        });

        messages
    }

    /// 格式化视觉记忆（用于调试）
    pub fn format_visual_memory(&self) -> String {
        self.visual_memory
            .iter()
            .map(|f| format!("[{:?}] {}", f.source, f.description))
            .collect::<Vec<_>>()
            .join("\n")
    }

    fn maybe_compact(&mut self) {
        if self.current_tokens < self.max_tokens * 70 / 100 {
            return;
        }

        let keep = 6usize;
        let compress_count = self.conversation.len().saturating_sub(keep);
        if compress_count < 3 {
            return;
        }

        let summary_budget = self.max_tokens * 15 / 100;

        let mut new_parts = Vec::new();
        for _ in 0..compress_count {
            if let Some(entry) = self.conversation.pop_front() {
                self.current_tokens = self.current_tokens.saturating_sub(entry.token_count);
                if entry.role == Role::User || entry.role == Role::Assistant {
                    let role_label = match entry.role {
                        Role::User => "用户",
                        Role::Assistant => "助手",
                        _ => "系统",
                    };
                    let excerpt = if entry.token_count > 60 {
                        let max_chars = if new_parts.len() < 3 { 150 } else { 60 };
                        format!("{}...", truncate_chars(&entry.content, max_chars))
                    } else {
                        entry.content
                    };
                    new_parts.push(format!("[{}]: {}", role_label, excerpt));
                }
            }
        }

        let new_section = new_parts.join("\n");
        let new_tokens = self.estimate_tokens(&new_section);

        let combined = if let Some(existing) = &self.conversation_summary {
            let existing_tokens = self.estimate_tokens(existing);
            if existing_tokens + new_tokens <= summary_budget {
                format!("{}\n{}", existing, new_section)
            } else {
                let overflow = (existing_tokens + new_tokens).saturating_sub(summary_budget);
                let keep_from_existing = existing_tokens.saturating_sub(overflow);
                let keep_chars = keep_from_existing * 2;
                if keep_chars > 100 {
                    let trimmed = tail_chars(existing, keep_chars);
                    format!("...{}\n{}", trimmed, new_section)
                } else {
                    new_section
                }
            }
        } else {
            new_section
        };

        let combined_tokens = self.estimate_tokens(&combined);
        self.conversation_summary = Some(if combined_tokens > summary_budget {
            let target_chars = summary_budget * 2;
            format!("...{}", tail_chars(&combined, target_chars))
        } else {
            combined
        });
    }

    /// 累加 API 调用消耗的 token
    pub fn add_tokens(&mut self, tokens: usize) {
        self.session_tokens_used += tokens;
    }

    /// 获取会话累计 token 消耗
    pub fn session_tokens(&self) -> usize {
        self.session_tokens_used
    }

    /// 简单的 token 估算（1 中文字符 ≈ 2 tokens，1 英文单词 ≈ 1 token）
    fn estimate_tokens(&self, text: &str) -> usize {
        let (cjk_count, other_count) = text.chars().fold((0usize, 0usize), |(cjk, other), c| {
            if is_cjk(c) {
                (cjk + 1, other)
            } else {
                (cjk, other + 1)
            }
        });
        cjk_count * 2 + other_count / 4
    }

    /// 清空上下文
    pub fn clear(&mut self) {
        self.conversation.clear();
        self.visual_memory.clear();
        self.conversation_summary = None;
        self.current_tokens = 0;
        self.session_tokens_used = 0;
    }

    /// 获取当前对话轮数
    pub fn conversation_turns(&self) -> usize {
        self.conversation.len()
    }

    /// 获取当前 token 估算
    pub fn estimated_tokens(&self) -> usize {
        self.current_tokens
    }
}

fn extract_terms(text: &str) -> HashSet<String> {
    let mut terms = text
        .split(|c: char| !c.is_alphanumeric() && !is_cjk(c))
        .map(|term| term.trim().to_lowercase())
        .filter(|term| term.chars().count() >= 2)
        .collect::<HashSet<_>>();

    let cjk_chars = text.chars().filter(|c| is_cjk(*c)).collect::<Vec<_>>();
    for window in cjk_chars.windows(2) {
        terms.insert(window.iter().collect());
    }

    terms
}

fn is_cjk(c: char) -> bool {
    matches!(
        c as u32,
        0x4E00..=0x9FFF | 0x3400..=0x4DBF | 0x3040..=0x30FF | 0xAC00..=0xD7AF
    )
}

fn truncate_chars(text: &str, max_chars: usize) -> String {
    text.chars().take(max_chars).collect()
}

fn tail_chars(text: &str, max_chars: usize) -> String {
    let char_count = text.chars().count();
    text.chars()
        .skip(char_count.saturating_sub(max_chars))
        .collect()
}

fn contains_ci(haystack: &str, needle: &str) -> bool {
    haystack.contains(needle)
}

fn jaccard_similarity(a: &str, b: &str) -> f32 {
    let set_a: HashSet<char> = a.chars().collect();
    let set_b: HashSet<char> = b.chars().collect();
    let intersection = set_a.intersection(&set_b).count() as f32;
    let union = set_a.union(&set_b).count() as f32;
    if union == 0.0 {
        0.0
    } else {
        intersection / union
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_context_manager_basic() {
        let mut ctx = ContextManager::new(4000);
        ctx.add_user_message("你好".to_string());
        ctx.add_assistant_message("你好！有什么可以帮助你的？".to_string());

        assert_eq!(ctx.conversation_turns(), 2);
        assert!(ctx.estimated_tokens() > 0);
    }

    #[test]
    fn test_build_messages() {
        let mut ctx = ContextManager::new(4000);
        ctx.add_user_message("这是什么？".to_string());
        ctx.add_assistant_message("这是一只猫。".to_string());

        let messages = ctx.build_messages(Some("一只橘色的猫"), "它在做什么？");
        assert!(messages.len() >= 3); // system + history + current
        assert!(messages.last().unwrap().content.contains("橘色的猫"));
    }

    #[test]
    fn test_visual_memory() {
        let mut ctx = ContextManager::new(4000);
        ctx.add_frame_summary(FrameSummary {
            description: "桌面环境".to_string(),
            key_elements: vec!["任务栏".to_string(), "浏览器窗口".to_string()],
            source: FrameSource::Screen,
        });

        assert_eq!(ctx.visual_memory.len(), 1);

        // 重复帧不添加
        ctx.add_frame_summary(FrameSummary {
            description: "桌面环境".to_string(),
            key_elements: vec!["任务栏".to_string()],
            source: FrameSource::Screen,
        });
        assert_eq!(ctx.visual_memory.len(), 1);
    }

    #[test]
    fn test_compact() {
        let mut ctx = ContextManager::new(100); // 很小的 token 预算
        for i in 0..20 {
            ctx.add_user_message(format!("消息 {}", i));
            ctx.add_assistant_message(format!("回复 {}", i));
        }

        // 应该触发压缩
        assert!(ctx.conversation_summary.is_some());
        assert!(ctx.conversation_turns() < 40);
    }

    #[test]
    fn test_compact_handles_multibyte_boundaries() {
        let mut ctx = ContextManager::new(100);
        let long_chinese = "这是一个包含很多中文字符的长消息".repeat(20);

        for _ in 0..12 {
            ctx.add_user_message(long_chinese.clone());
            ctx.add_assistant_message(long_chinese.clone());
        }

        assert!(ctx.conversation_summary.is_some());
        assert!(ctx
            .conversation_summary
            .as_ref()
            .unwrap()
            .is_char_boundary(0));
    }

    #[test]
    fn test_token_estimate_counts_chars_not_bytes() {
        let ctx = ContextManager::new(4000);
        assert_eq!(ctx.estimate_tokens("你好世界"), 8);
        assert_eq!(ctx.estimate_tokens("abcdefgh"), 2);
    }

    #[test]
    fn test_memory() {
        let mut ctx = ContextManager::new(4000);
        ctx.add_memory(MemoryEntry {
            id: "memory-1".to_string(),
            content: "用户喜欢简洁的回答".to_string(),
            importance: 0.8,
            source: "对话".to_string(),
            created_at: Utc::now(),
            last_used_at: None,
            tags: vec!["偏好".to_string()],
        });

        // 重复记忆不添加
        ctx.add_memory(MemoryEntry {
            id: "memory-2".to_string(),
            content: "用户喜欢简洁的回答".to_string(),
            importance: 0.9,
            source: "对话".to_string(),
            created_at: Utc::now(),
            last_used_at: None,
            tags: vec![],
        });
        assert_eq!(ctx.long_term_memory.len(), 1);
    }

    #[test]
    fn test_clear() {
        let mut ctx = ContextManager::new(4000);
        ctx.add_user_message("test".to_string());
        ctx.add_frame_summary(FrameSummary {
            description: "test".to_string(),
            key_elements: vec![],
            source: FrameSource::Camera,
        });

        ctx.clear();
        assert_eq!(ctx.conversation_turns(), 0);
        assert_eq!(ctx.visual_memory.len(), 0);
    }

    #[test]
    fn test_memory_similarity_dedup() {
        let mut ctx = ContextManager::new(4000);
        ctx.add_memory(MemoryEntry {
            id: "m1".to_string(),
            content: "用户喜欢简洁的回答".to_string(),
            importance: 0.8,
            source: "对话".to_string(),
            created_at: Utc::now(),
            last_used_at: None,
            tags: vec![],
        });

        ctx.add_memory(MemoryEntry {
            id: "m2".to_string(),
            content: "用户喜欢简明的回答".to_string(),
            importance: 0.9,
            source: "对话".to_string(),
            created_at: Utc::now(),
            last_used_at: None,
            tags: vec![],
        });
        assert_eq!(ctx.long_term_memory.len(), 1);

        ctx.add_memory(MemoryEntry {
            id: "m3".to_string(),
            content: "今天天气很好适合出门".to_string(),
            importance: 0.5,
            source: "对话".to_string(),
            created_at: Utc::now(),
            last_used_at: None,
            tags: vec![],
        });
        assert_eq!(ctx.long_term_memory.len(), 2);
    }

    #[test]
    fn test_token_budget_enforcement() {
        let mut ctx = ContextManager::new(200);
        for i in 0..20 {
            ctx.add_user_message(format!("这是一条测试消息 {}", i));
            ctx.add_assistant_message(format!("这是回复 {}", i));
        }

        let messages = ctx.build_messages(None, "你好");
        let total_tokens: usize = messages.iter().map(|m| m.content.len() / 4).sum();
        assert!(total_tokens < 400);
    }
}
