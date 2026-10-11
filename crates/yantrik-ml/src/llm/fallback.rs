//! Fallback LLM — wraps a primary backend with an automatic fallback.
//!
//! When the primary backend fails (network error, timeout, CLI crash),
//! the fallback backend is lazily initialized and used instead.
//! This enables offline intelligence via a local GGUF model (e.g. Qwen3.5-0.8B)
//! while keeping the powerful primary backend (Claude CLI, API) for normal use.
//!
//! Supports two fallback modes:
//! - `api` — fallback to a local llama-server (or any OpenAI-compatible endpoint)
//! - `llamacpp` — fallback via embedded llama.cpp (requires `llamacpp` feature)

use std::path::PathBuf;
use std::sync::{Arc, Mutex};

use anyhow::Result;
use tracing;

use crate::traits::LLMBackend;
use crate::types::{ChatMessage, GenerationConfig, LLMResponse};

/// Configuration for the fallback LLM backend.
#[derive(Debug, Clone)]
pub enum FallbackConfig {
    /// Use an API endpoint (e.g. local llama-server) as fallback.
    Api {
        base_url: String,
        model: String,
    },
    /// Use embedded llama.cpp (requires `llamacpp` feature at compile time).
    LlamaCpp {
        model_path: PathBuf,
        n_gpu_layers: u32,
        context_size: u32,
    },
}

/// An LLM backend that wraps a primary + fallback.
///
/// On every call, tries the primary backend first. If it fails,
/// lazily initializes and uses the fallback.
pub struct FallbackLLM {
    primary: Arc<dyn LLMBackend>,
    fallback: Mutex<Option<Arc<dyn LLMBackend>>>,
    fallback_config: Option<FallbackConfig>,
    /// Track consecutive primary failures to avoid slow retries
    primary_failures: Mutex<u32>,
}

impl FallbackLLM {
    /// Create a new FallbackLLM.
    ///
    /// The fallback is NOT initialized until the primary fails.
    /// If `fallback_config` is None, no fallback is available — primary errors pass through.
    pub fn new(
        primary: Arc<dyn LLMBackend>,
        fallback_config: Option<FallbackConfig>,
    ) -> Self {
        Self {
            primary,
            fallback: Mutex::new(None),
            fallback_config,
            primary_failures: Mutex::new(0),
        }
    }

    /// Create with an already-initialized fallback backend (for testing or pre-warming).
    pub fn with_fallback(
        primary: Arc<dyn LLMBackend>,
        fallback: Arc<dyn LLMBackend>,
    ) -> Self {
        Self {
            primary,
            fallback: Mutex::new(Some(fallback)),
            fallback_config: None,
            primary_failures: Mutex::new(0),
        }
    }

    /// Get or initialize the fallback backend.
    fn get_fallback(&self) -> Result<Arc<dyn LLMBackend>> {
        let mut guard = self.fallback.lock().map_err(|e| anyhow::anyhow!("lock: {e}"))?;

        if let Some(ref fb) = *guard {
            return Ok(fb.clone());
        }

        let config = self.fallback_config.as_ref()
            .ok_or_else(|| anyhow::anyhow!("no fallback configured"))?;

        let fb: Arc<dyn LLMBackend> = match config {
            FallbackConfig::Api { base_url, model } => {
                tracing::info!(
                    base_url = %base_url,
                    model = %model,
                    "Initializing fallback LLM (API)"
                );
                #[cfg(feature = "api-llm")]
                {
                    Arc::new(crate::llm::ApiLLM::new(
                        base_url.clone(),
                        None, // no API key for local server
                        model,
                    ))
                }
                #[cfg(not(feature = "api-llm"))]
                {
                    anyhow::bail!("fallback API requires 'api-llm' feature at compile time")
                }
            }
            FallbackConfig::LlamaCpp { model_path, n_gpu_layers, context_size } => {
                if !model_path.exists() {
                    anyhow::bail!(
                        "fallback model not found: {}",
                        model_path.display()
                    );
                }
                tracing::info!(
                    model = %model_path.display(),
                    gpu_layers = n_gpu_layers,
                    ctx = context_size,
                    "Initializing fallback LLM (llama.cpp)"
                );
                #[cfg(feature = "llamacpp")]
                {
                    Arc::new(crate::llm::LlamaCppLLM::from_gguf(
                        model_path,
                        *n_gpu_layers,
                        *context_size,
                    )?)
                }
                #[cfg(not(feature = "llamacpp"))]
                {
                    anyhow::bail!("fallback requires 'llamacpp' feature at compile time")
                }
            }
        };

        *guard = Some(fb.clone());
        Ok(fb)
    }

    /// Record a primary success — reset failure counter.
    fn primary_success(&self) {
        if let Ok(mut count) = self.primary_failures.lock() {
            if *count > 0 {
                tracing::info!(prev_failures = *count, "Primary LLM recovered");
                *count = 0;
            }
        }
    }

    /// Record a primary failure — increment counter.
    fn primary_failure(&self) {
        if let Ok(mut count) = self.primary_failures.lock() {
            *count += 1;
        }
    }

    /// Check if we should skip the primary (too many consecutive failures).
    /// After 3 consecutive failures, go straight to fallback for 10 calls,
    /// then retry primary once.
    fn should_skip_primary(&self) -> bool {
        if let Ok(count) = self.primary_failures.lock() {
            *count >= 3 && *count % 10 != 0
        } else {
            false
        }
    }

    /// Slim down messages for the tiny fallback model.
    /// Keeps system prompt (truncated), last user message, and at most 2 history turns.
    fn slim_messages(messages: &[ChatMessage]) -> Vec<ChatMessage> {
        let mut slim = Vec::with_capacity(4);

        // System message: truncate to ~600 chars
        if let Some(sys) = messages.first() {
            if sys.role == "system" {
                let content = if sys.content.len() > 600 {
                    // Find a safe truncation point
                    let mut end = 600;
                    while end > 0 && !sys.content.is_char_boundary(end) {
                        end -= 1;
                    }
                    format!("{}\nBe concise. Answer directly.", &sys.content[..end])
                } else {
                    sys.content.clone()
                };
                slim.push(ChatMessage::system(content));
            }
        }

        // Last user message (always keep)
        if let Some(last) = messages.last() {
            if last.role == "user" {
                // Keep at most 4 preceding messages, never the system prompt pushed above.
                // The floor is 1 only when messages[0] IS that prompt: a lone user message has
                // nothing before it, and `1..0` panicked.
                let first_kept = usize::from(messages[0].role == "system");
                let history_start = messages.len().saturating_sub(5).max(first_kept);
                for msg in &messages[history_start..messages.len() - 1] {
                    slim.push(msg.clone());
                }
                slim.push(last.clone());
            }
        }

        slim
    }

    /// Reduced generation config for fallback: fewer tokens, lower temp.
    fn slim_config(config: &GenerationConfig) -> GenerationConfig {
        GenerationConfig {
            max_tokens: config.max_tokens.min(512),
            temperature: config.temperature.min(0.5),
            top_p: config.top_p,
            ..Default::default()
        }
    }
}

impl LLMBackend for FallbackLLM {
    fn chat(
        &self,
        messages: &[ChatMessage],
        config: &GenerationConfig,
        tools: Option<&[serde_json::Value]>,
    ) -> Result<LLMResponse> {
        // A cancelled call starts nothing, on either backend.
        crate::call_cancel::check()?;
        // If primary has been failing consistently, skip to fallback
        if self.should_skip_primary() {
            if let Ok(fb) = self.get_fallback() {
                tracing::debug!("Skipping primary (consecutive failures), using fallback");
                return fb.chat(messages, config, tools);
            }
        }

        match self.primary.chat(messages, config, tools) {
            Ok(resp) => {
                self.primary_success();
                Ok(resp)
            }
            // A cancel is not a failure: no count against the primary, no fallback request.
            Err(e) if crate::call_cancel::is_cancel(&e) => Err(e),
            Err(primary_err) => {
                self.primary_failure();
                tracing::warn!(
                    error = %primary_err,
                    primary = self.primary.backend_name(),
                    "Primary LLM failed, trying fallback"
                );

                match self.get_fallback() {
                    Ok(fb) => {
                        let slim = Self::slim_messages(messages);
                        let slim_cfg = Self::slim_config(config);
                        fb.chat(&slim, &slim_cfg, None)
                    }
                    Err(fb_err) => {
                        tracing::error!(
                            fallback_error = %fb_err,
                            "Fallback LLM also failed"
                        );
                        Err(primary_err)
                    }
                }
            }
        }
    }

    fn chat_streaming(
        &self,
        messages: &[ChatMessage],
        config: &GenerationConfig,
        tools: Option<&[serde_json::Value]>,
        on_token: &mut dyn FnMut(&str),
    ) -> Result<LLMResponse> {
        // A cancelled call starts nothing, on either backend.
        crate::call_cancel::check()?;
        if self.should_skip_primary() {
            if let Ok(fb) = self.get_fallback() {
                tracing::debug!("Skipping primary (consecutive failures), using fallback streaming");
                return fb.chat_streaming(messages, config, tools, on_token);
            }
        }

        match self.primary.chat_streaming(messages, config, tools, on_token) {
            Ok(resp) => {
                self.primary_success();
                Ok(resp)
            }
            // A cancel is not a failure: no count against the primary, and no fallback stream
            // appending a second answer to the tokens already sent.
            Err(e) if crate::call_cancel::is_cancel(&e) => Err(e),
            Err(primary_err) => {
                self.primary_failure();
                tracing::warn!(
                    error = %primary_err,
                    primary = self.primary.backend_name(),
                    "Primary LLM streaming failed, trying fallback"
                );

                match self.get_fallback() {
                    Ok(fb) => {
                        let slim = Self::slim_messages(messages);
                        let slim_cfg = Self::slim_config(config);
                        fb.chat_streaming(&slim, &slim_cfg, None, on_token)
                    }
                    Err(_) => Err(primary_err),
                }
            }
        }
    }

    fn count_tokens(&self, text: &str) -> Result<usize> {
        self.primary.count_tokens(text)
            .or_else(|_| {
                self.get_fallback()
                    .and_then(|fb| fb.count_tokens(text))
            })
    }

    fn backend_name(&self) -> &str {
        self.primary.backend_name()
    }

    fn is_degraded(&self) -> bool {
        if let Ok(count) = self.primary_failures.lock() {
            *count > 0
        } else {
            false
        }
    }

    fn model_id(&self) -> &str {
        self.primary.model_id()
    }
}

/// Yantrik Mind E.VOICE2 (review on yantrik-companion#2): a cancelled call is not a primary
/// failure. It starts no fallback request, appends nothing more to the stream, and never pushes
/// the wrapper into skipping a good primary.
#[cfg(test)]
mod cancel_tests {
    use super::*;
    use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};

    /// Emits one token, then reads the cancel flag the way a stream loop does (setting it first
    /// when `cancel_mid_stream`), or fails outright when `fail`.
    struct Counting {
        calls: AtomicUsize,
        cancel_mid_stream: Option<Arc<AtomicBool>>,
        fail: bool,
    }

    impl Counting {
        fn new(cancel_mid_stream: Option<Arc<AtomicBool>>, fail: bool) -> Arc<Self> {
            Arc::new(Self { calls: AtomicUsize::new(0), cancel_mid_stream, fail })
        }
        fn run(&self, on_token: &mut dyn FnMut(&str)) -> Result<LLMResponse> {
            self.calls.fetch_add(1, Ordering::SeqCst);
            on_token("tok ");
            if let Some(f) = &self.cancel_mid_stream {
                f.store(true, Ordering::SeqCst);
            }
            crate::call_cancel::check()?;
            if self.fail {
                anyhow::bail!("connection reset");
            }
            Ok(LLMResponse { text: "tok".into(), ..Default::default() })
        }
    }

    impl LLMBackend for Counting {
        fn chat(&self, _: &[ChatMessage], _: &GenerationConfig, _: Option<&[serde_json::Value]>) -> Result<LLMResponse> {
            self.run(&mut |_| {})
        }
        fn chat_streaming(&self, _: &[ChatMessage], _: &GenerationConfig, _: Option<&[serde_json::Value]>, on: &mut dyn FnMut(&str)) -> Result<LLMResponse> {
            self.run(on)
        }
        fn count_tokens(&self, t: &str) -> Result<usize> {
            Ok(t.len())
        }
        fn backend_name(&self) -> &str {
            "counting"
        }
    }

    fn msgs() -> Vec<ChatMessage> {
        vec![ChatMessage::system("be brief"), ChatMessage::user("hi")]
    }

    #[test]
    fn a_cancel_mid_stream_starts_no_fallback_and_counts_no_failure() {
        let flag = Arc::new(AtomicBool::new(false));
        let _g = crate::call_cancel::install_cancel(Some(flag.clone()));
        let primary = Counting::new(Some(flag.clone()), false);
        let fallback = Counting::new(None, false);
        let llm = FallbackLLM::with_fallback(primary.clone(), fallback.clone());
        let mut tokens = 0;
        let e = llm.chat_streaming(&msgs(), &GenerationConfig::default(), None, &mut |_| tokens += 1).unwrap_err();
        assert!(crate::call_cancel::is_cancel(&e), "{e:#}");
        assert_eq!(fallback.calls.load(Ordering::SeqCst), 0, "a fallback request was sent after the cancel");
        assert_eq!(tokens, 1, "only the primary's token reached the stream");
        assert!(!llm.is_degraded(), "a cancel was counted as a primary failure");
        // More calls while the flag is set start nothing, on either backend.
        for _ in 0..3 {
            assert!(llm.chat(&msgs(), &GenerationConfig::default(), None).is_err());
        }
        assert_eq!(primary.calls.load(Ordering::SeqCst), 1, "a call started after the flag was set");
        assert_eq!(fallback.calls.load(Ordering::SeqCst), 0);
    }

    /// The Err arm itself, with the guard at the top of the call out of the way: the primary is
    /// cancelled DURING the call, and a later call with a clear flag still goes to the primary.
    #[test]
    fn three_cancels_never_make_it_skip_the_primary() {
        let primary = Counting::new(None, false);
        let fallback = Counting::new(None, false);
        let llm = FallbackLLM::with_fallback(primary.clone(), fallback.clone());
        for _ in 0..3 {
            let flag = Arc::new(AtomicBool::new(false));
            let _g = crate::call_cancel::install_cancel(Some(flag.clone()));
            let mid = Counting::new(Some(flag), false);
            let one = FallbackLLM::with_fallback(mid, fallback.clone());
            assert!(one.chat(&msgs(), &GenerationConfig::default(), None).is_err());
            assert!(!one.is_degraded(), "a cancel was counted as a failure");
        }
        assert_eq!(fallback.calls.load(Ordering::SeqCst), 0, "a cancel fell back");
        llm.chat(&msgs(), &GenerationConfig::default(), None).expect("the primary answers");
        assert_eq!(primary.calls.load(Ordering::SeqCst), 1);
    }

    /// The same wrapper, cancelled mid-stream three times: the count never moves, so the fourth
    /// call (flag clear) still goes to the primary.
    #[test]
    fn three_cancels_on_one_wrapper_never_skip_the_primary() {
        let flag = Arc::new(AtomicBool::new(false));
        let primary = Counting::new(Some(flag.clone()), false);
        let fallback = Counting::new(None, false);
        let llm = FallbackLLM::with_fallback(primary.clone(), fallback.clone());
        for _ in 0..3 {
            flag.store(false, Ordering::SeqCst);
            let _g = crate::call_cancel::install_cancel(Some(flag.clone()));
            let e = llm.chat_streaming(&msgs(), &GenerationConfig::default(), None, &mut |_| {}).unwrap_err();
            assert!(crate::call_cancel::is_cancel(&e), "{e:#}");
            assert!(!llm.is_degraded());
        }
        let _none = crate::call_cancel::install_cancel(None);
        llm.chat(&msgs(), &GenerationConfig::default(), None).expect("the primary answers");
        assert_eq!(primary.calls.load(Ordering::SeqCst), 4, "the primary was skipped after cancels");
        assert_eq!(fallback.calls.load(Ordering::SeqCst), 0);
    }

    /// Real failures still count: after three, the primary is skipped.
    #[test]
    fn three_real_failures_still_skip_the_primary() {
        let primary = Counting::new(None, true);
        let fallback = Counting::new(None, false);
        let llm = FallbackLLM::with_fallback(primary.clone(), fallback.clone());
        for _ in 0..4 {
            llm.chat(&msgs(), &GenerationConfig::default(), None).expect("the fallback answers");
        }
        assert_eq!(primary.calls.load(Ordering::SeqCst), 3, "the fourth call skipped the primary");
        assert_eq!(fallback.calls.load(Ordering::SeqCst), 4);
    }

    #[test]
    fn a_real_primary_failure_still_falls_back() {
        let primary = Counting::new(None, true);
        let fallback = Counting::new(None, false);
        let llm = FallbackLLM::with_fallback(primary.clone(), fallback.clone());
        llm.chat_streaming(&msgs(), &GenerationConfig::default(), None, &mut |_| {}).expect("the fallback answers");
        assert_eq!(fallback.calls.load(Ordering::SeqCst), 1);
        assert!(llm.is_degraded(), "a failure is still counted");
    }
}

/// `slim_messages` must not panic on any conversation (review on yantrik-companion#2: a lone
/// user message sliced `1..0`).
#[cfg(test)]
mod slim_tests {
    use super::*;

    fn tool_result(id: &str, content: &str) -> ChatMessage {
        ChatMessage { role: "tool".into(), content: content.into(), tool_calls: None, tool_call_id: Some(id.into()), name: Some("recall".into()) }
    }

    #[test]
    fn empty_gives_empty() {
        assert!(FallbackLLM::slim_messages(&[]).is_empty());
    }

    #[test]
    fn a_lone_user_message_is_kept() {
        let s = FallbackLLM::slim_messages(&[ChatMessage::user("hi")]);
        assert_eq!(s.len(), 1);
        assert_eq!((s[0].role.as_str(), s[0].content.as_str()), ("user", "hi"));
    }

    #[test]
    fn user_turns_with_no_system_prompt_keep_the_first_turn() {
        let m = [ChatMessage::user("a"), ChatMessage::assistant("b"), ChatMessage::user("c")];
        let s = FallbackLLM::slim_messages(&m);
        assert_eq!(s.iter().map(|m| m.content.as_str()).collect::<Vec<_>>(), ["a", "b", "c"]);
    }

    #[test]
    fn a_single_system_message_is_kept_alone() {
        let s = FallbackLLM::slim_messages(&[ChatMessage::system("be brief")]);
        assert_eq!(s.len(), 1);
        assert_eq!(s[0].role, "system");
    }

    #[test]
    fn a_system_prompt_and_a_user_message() {
        let s = FallbackLLM::slim_messages(&[ChatMessage::system("sys"), ChatMessage::user("hi")]);
        assert_eq!(s.iter().map(|m| m.role.as_str()).collect::<Vec<_>>(), ["system", "user"]);
    }

    #[test]
    fn a_long_history_keeps_four_messages_before_the_last() {
        let mut m = vec![ChatMessage::system("sys")];
        for i in 0..10 {
            m.push(ChatMessage::user(format!("u{i}")));
            m.push(ChatMessage::assistant(format!("a{i}")));
        }
        m.push(ChatMessage::user("last"));
        let s = FallbackLLM::slim_messages(&m);
        assert_eq!(s.len(), 6, "system + 4 history + last");
        assert_eq!(s[0].role, "system");
        assert_eq!(s[5].content, "last");
    }

    /// Tool messages never panic. Ending on a tool result (mid tool round) leaves only the system
    /// prompt -- known and pre-existing; the yantrik-os copy anchors on the last user turn instead.
    #[test]
    fn tool_messages_do_not_panic() {
        let s = FallbackLLM::slim_messages(&[tool_result("c1", "r")]);
        assert!(s.is_empty());
        let m = [ChatMessage::system("sys"), ChatMessage::user("find food"), tool_result("c1", "r")];
        let s = FallbackLLM::slim_messages(&m);
        assert_eq!(s.len(), 1);
        let m = [ChatMessage::system("sys"), tool_result("c1", "r"), ChatMessage::user("and?")];
        assert_eq!(FallbackLLM::slim_messages(&m).len(), 3);
        let m = [tool_result("c1", "r"), ChatMessage::user("and?")];
        assert_eq!(FallbackLLM::slim_messages(&m).len(), 2);
    }

    /// Byte 600 falls inside a multi-byte character at every offset of a 2-, 3- and 4-byte char.
    #[test]
    fn non_ascii_at_the_cut_point_does_not_panic() {
        for ch in ["é", "ह", "日", "🙂"] {
            for pad in 0..4 {
                let content = format!("{}{}", "a".repeat(597 + pad), ch.repeat(20));
                let s = FallbackLLM::slim_messages(&[ChatMessage::system(content.clone()), ChatMessage::user("hi")]);
                let kept = s[0].content.strip_suffix("\nBe concise. Answer directly.").expect("truncated");
                assert!(kept.len() <= 600 && content.starts_with(kept), "{ch} pad {pad}");
            }
        }
    }
}
