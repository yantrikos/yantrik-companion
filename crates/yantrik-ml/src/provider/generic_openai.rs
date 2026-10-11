//! GenericOpenAIBackend — a single reusable backend for all OpenAI-compatible providers.
//!
//! Covers: Ollama, OpenAI, DeepSeek, OpenRouter, Groq, Together, Fireworks,
//! Mistral, HuggingFace, xAI, and any custom OpenAI-compatible endpoint.
//!
//! Refactored from the original `api.rs` into a configurable, provider-agnostic backend.

use std::collections::HashMap;
use std::io::{BufRead, BufReader};

use anyhow::{Context, Result};

use crate::chat_template;
use crate::traits::LLMBackend;
use crate::types::{ApiToolCall, ApiToolCallFunction, ChatMessage, GenerationConfig, LLMResponse, ToolCall};

/// Provider-specific presets that modify request behavior.
#[derive(Debug, Clone)]
pub struct ProviderPresets {
    /// Whether to disable thinking mode (e.g. `"think": false` for Qwen models on Ollama).
    pub disable_thinking: bool,
    /// Custom context window size for Ollama options.
    pub context_window: Option<u32>,
    /// Extra headers to include in every request.
    pub extra_headers: Vec<(String, String)>,
    /// Whether this is an Ollama endpoint (uses /api/chat native endpoint).
    pub is_ollama: bool,
}

impl Default for ProviderPresets {
    fn default() -> Self {
        Self {
            disable_thinking: false,
            context_window: None,
            extra_headers: Vec::new(),
            is_ollama: false,
        }
    }
}

impl ProviderPresets {
    /// Presets for Ollama local inference.
    pub fn ollama() -> Self {
        Self {
            disable_thinking: true,
            context_window: Some(32768),
            is_ollama: true,
            ..Default::default()
        }
    }

    /// Presets for OpenRouter (needs HTTP-Referer and X-Title).
    pub fn openrouter() -> Self {
        Self {
            extra_headers: vec![
                ("HTTP-Referer".to_string(), "https://yantrikos.com".to_string()),
                ("X-Title".to_string(), "Yantrik OS".to_string()),
            ],
            ..Default::default()
        }
    }
}

/// A generic OpenAI-compatible LLM backend.
///
/// Supports both standard OpenAI `/v1/chat/completions` and Ollama native
/// `/api/chat` endpoints. Configurable base URL, auth, and provider presets.
pub struct GenericOpenAIBackend {
    /// Base URL (e.g. "https://api.openai.com/v1" or "http://localhost:11434/v1").
    base_url: String,
    /// API key (None for local/unauthenticated endpoints).
    api_key: Option<String>,
    /// Model name (e.g. "gpt-4o", "qwen3.5:27b-nothink").
    model: String,
    /// Auth header style: "bearer" sends `Authorization: Bearer <key>` when a key is set,
    /// "none" never sends one.
    auth_style: String,
    /// Provider-specific behavior presets.
    presets: ProviderPresets,
}

impl GenericOpenAIBackend {
    /// Create a new GenericOpenAIBackend.
    ///
    /// # Arguments
    /// * `base_url` — API base URL (e.g. "https://api.openai.com/v1")
    /// * `api_key` — Optional API key
    /// * `model` — Model name/identifier
    /// * `auth_style` — "bearer" or "none"
    /// * `presets` — Provider-specific behavior presets
    pub fn new(
        base_url: impl Into<String>,
        api_key: Option<String>,
        model: impl Into<String>,
        auth_style: impl Into<String>,
        presets: ProviderPresets,
    ) -> Self {
        let base_url: String = base_url.into();
        let mut presets = presets;
        // Auto-detect Ollama if URL contains :11434
        if base_url.contains(":11434") && !presets.is_ollama {
            presets.is_ollama = true;
            presets.disable_thinking = true;
            if presets.context_window.is_none() {
                presets.context_window = Some(32768);
            }
        }
        Self {
            base_url,
            api_key,
            model: model.into(),
            auth_style: auth_style.into(),
            presets,
        }
    }

    /// Convenience constructor for a known provider type.
    pub fn for_provider(
        provider_type: &str,
        base_url: impl Into<String>,
        api_key: Option<String>,
        model: impl Into<String>,
    ) -> Self {
        let presets = match provider_type {
            "ollama" => ProviderPresets::ollama(),
            "openrouter" => ProviderPresets::openrouter(),
            _ => ProviderPresets::default(),
        };
        // Every provider sends the key it was given for this address, and none without one. Ollama
        // itself takes no key, but one behind an authenticating proxy (a gateway, a sealed gate)
        // needs it, and a key is only ever set for the address it belongs to.
        Self::new(base_url, api_key, model, "bearer", presets)
    }

    /// Per-workload thinking control (builder). `true` lets the model reason (higher quality,
    /// slower); `false` suppresses the reasoning preamble (faster/cheaper). On qwen3.6 MoE this
    /// is a BINARY accuracy trade — the maintainer confirmed reasoning_effort low/medium/high are
    /// accepted but don't scale, so on/off per workload is the only real lever. Prefer `true` on
    /// planning / grounded-reasoning paths; reserve `false` for clearly-trivial latency-critical
    /// turns. Applied post-construction, so it also overrides the :11434 auto-detect default.
    pub fn with_thinking(mut self, enabled: bool) -> Self {
        self.presets.disable_thinking = !enabled;
        self
    }

    /// Serialize a ChatMessage to JSON.
    fn serialize_message(m: &ChatMessage, ollama_compat: bool) -> serde_json::Value {
        let mut msg = serde_json::json!({ "role": m.role });

        if m.role == "assistant" && m.content.is_empty() && m.tool_calls.is_some() {
            msg["content"] = serde_json::Value::Null;
        } else {
            msg["content"] = serde_json::json!(m.content);
        }

        if let Some(ref calls) = m.tool_calls {
            if ollama_compat {
                let fixed: Vec<serde_json::Value> = calls.iter().map(|tc| {
                    let args = serde_json::from_str::<serde_json::Value>(&tc.function.arguments)
                        .unwrap_or(serde_json::json!({}));
                    serde_json::json!({
                        "id": tc.id,
                        "function": { "name": tc.function.name, "arguments": args }
                    })
                }).collect();
                msg["tool_calls"] = serde_json::json!(fixed);
            } else {
                msg["tool_calls"] = serde_json::json!(calls);
            }
        }
        if let Some(ref id) = m.tool_call_id {
            msg["tool_call_id"] = serde_json::json!(id);
        }
        if let Some(ref name) = m.name {
            msg["name"] = serde_json::json!(name);
        }

        msg
    }

    /// Parse native tool_calls from the response JSON.
    fn parse_api_tool_calls(message: &serde_json::Value) -> Vec<ApiToolCall> {
        let Some(calls) = message["tool_calls"].as_array() else {
            return Vec::new();
        };

        calls
            .iter()
            .filter_map(|tc| {
                let id = tc["id"].as_str()?.to_string();
                let call_type = tc["type"].as_str().unwrap_or("function").to_string();
                let name = tc["function"]["name"].as_str()?.to_string();
                let arguments = match &tc["function"]["arguments"] {
                    v if v.is_string() => v.as_str().unwrap().to_string(),
                    v if v.is_object() || v.is_array() => serde_json::to_string(v).unwrap_or_else(|_| "{}".to_string()),
                    _ => "{}".to_string(),
                };
                Some(ApiToolCall {
                    id,
                    call_type,
                    function: ApiToolCallFunction { name, arguments },
                })
            })
            .collect()
    }

    fn build_agent(&self) -> ureq::Agent {
        ureq::Agent::new_with_config(
            ureq::config::Config::builder()
                .timeout_global(Some(crate::call_timeout::call_timeout_for(&self.model)))
                .build()
        )
    }

    /// The agent for a chat request: an error status comes back as a response, so its body -- the
    /// server's own reason -- can be read (`status_error`). Only the two chat request paths use it.
    fn build_chat_agent(&self) -> ureq::Agent {
        ureq::Agent::new_with_config(
            ureq::config::Config::builder()
                .timeout_global(Some(crate::call_timeout::call_timeout_for(&self.model)))
                .http_status_as_error(false)
                .build()
        )
    }

    /// A non-2xx answer as an error that keeps what the server said (Yantrik Mind E.QUOTA1). It
    /// begins `http status: N` exactly as ureq's own error did, so every matcher of that text keeps
    /// working; then ` (retry after X)` when the server sent Retry-After, and `: <its words>`. A 429
    /// that means "your weekly allowance is spent" and one that means "slow down" were the same
    /// three words before, and the caller waited a minute on each.
    fn status_error(resp: ureq::http::Response<ureq::Body>) -> anyhow::Error {
        use std::io::Read as _;
        let code = resp.status().as_u16();
        let retry = resp
            .headers()
            .get("retry-after")
            .and_then(|v| v.to_str().ok())
            .map(|v| v.trim().chars().take(40).collect::<String>())
            .filter(|v| !v.is_empty());
        let mut raw = Vec::new();
        let _ = resp.into_body().into_reader().take(4096).read_to_end(&mut raw);
        let said = Self::server_said(&String::from_utf8_lossy(&raw));
        let mut msg = format!("http status: {code}");
        if let Some(r) = retry {
            msg.push_str(&format!(" (retry after {r})"));
        }
        if !said.is_empty() {
            msg.push_str(": ");
            msg.push_str(&said);
        }
        anyhow::anyhow!(msg)
    }

    /// What an error body says, as one line of at most 300 characters: `error.message`, `error`, or
    /// `message` from JSON, else the text itself.
    pub(crate) fn server_said(body: &str) -> String {
        let v: Option<serde_json::Value> = serde_json::from_str(body.trim()).ok();
        let picked = v.as_ref().and_then(|v| {
            v["error"]["message"].as_str().or_else(|| v["error"].as_str()).or_else(|| v["message"].as_str()).map(str::to_string)
        });
        let text = picked.unwrap_or_else(|| body.to_string());
        let one: String = text.split_whitespace().collect::<Vec<_>>().join(" ");
        one.chars().take(300).collect()
    }

    /// Build auth and extra headers as a vec of (key, value) pairs.
    fn auth_headers(&self) -> Vec<(String, String)> {
        let mut headers = Vec::new();
        if let Some(key) = self.api_key.as_deref().filter(|k| !k.trim().is_empty()) {
            if self.auth_style != "none" {
                headers.push(("Authorization".to_string(), format!("Bearer {key}")));
            }
        }
        for (k, v) in &self.presets.extra_headers {
            headers.push((k.clone(), v.clone()));
        }
        headers
    }

    // ── Ollama native API (/api/chat) ──

    fn ollama_base_url(&self) -> String {
        let base = self.base_url.trim_end_matches('/');
        base.trim_end_matches("/v1").to_string()
    }

    fn build_ollama_body(
        &self,
        messages: &[ChatMessage],
        config: &GenerationConfig,
        tools: Option<&[serde_json::Value]>,
        stream: bool,
    ) -> serde_json::Value {
        let msgs: Vec<serde_json::Value> = messages
            .iter()
            .map(|m| Self::serialize_message(m, true))
            .collect();

        let mut options = serde_json::json!({
            "temperature": config.temperature,
            "num_predict": config.max_tokens,
        });

        if let Some(ctx) = self.presets.context_window {
            options["num_ctx"] = serde_json::json!(ctx);
        }
        if let Some(p) = config.top_p {
            options["top_p"] = serde_json::json!(p);
        }
        if config.repeat_penalty != 1.0 {
            options["repeat_penalty"] = serde_json::json!(config.repeat_penalty);
        }
        if !config.stop.is_empty() {
            options["stop"] = serde_json::json!(config.stop);
        }

        let mut body = serde_json::json!({
            "model": self.model,
            "messages": msgs,
            "stream": stream,
            "options": options,
        });

        // Per-call think override (config.think) wins over the provider preset — Ollama honors
        // `think` in the request body, which is what enables dual-mode from a single model tag:
        // OFF for tool-selection dispatch, ON for reasoning/compose. None => preset default.
        match crate::think_policy::think_setting_for_model(&self.model, config.think) {
            Some(setting) => body["think"] = setting.native_json(),
            None if self.presets.disable_thinking => body["think"] = serde_json::json!(false),
            None => {}
        }

        if let Some(tools) = tools {
            if !tools.is_empty() {
                body["tools"] = serde_json::json!(tools);
            }
        }

        body
    }

    fn send_ollama_request(&self, body: &serde_json::Value) -> Result<ureq::Body> {
        let url = format!("{}/api/chat", self.ollama_base_url());
        let body_str = serde_json::to_string(body)?;

        let tool_count = body.get("tools").and_then(|t| t.as_array()).map(|a| a.len()).unwrap_or(0);
        tracing::debug!(body_bytes = body_str.len(), tools = tool_count, url = %url, "GenericOpenAI Ollama request");

        let agent = self.build_chat_agent();
        let mut req = agent.post(&url).header("Content-Type", "application/json");
        for (k, v) in self.auth_headers() {
            req = req.header(&k, &v);
        }

        let resp = req
            .send(body_str.as_bytes())
            .map_err(|e| {
                tracing::error!(error = %e, "Ollama API request failed");
                e
            })
            .map_err(|e| crate::call_timeout::describe_send_error(&self.model, e))
            .with_context(|| {
                format!("Ollama API request to {} failed", crate::call_timeout::shown_endpoint(&url))
            })?;
        if !resp.status().is_success() {
            return Err(Self::status_error(resp).context(format!(
                "Ollama API request to {} failed",
                crate::call_timeout::shown_endpoint(&url)
            )));
        }

        Ok(resp.into_body())
    }

    fn ollama_chat(
        &self,
        messages: &[ChatMessage],
        config: &GenerationConfig,
        tools: Option<&[serde_json::Value]>,
    ) -> Result<LLMResponse> {
        let body = self.build_ollama_body(messages, config, tools, false);
        let mut resp_body = self.send_ollama_request(&body)?;

        let json: serde_json::Value = resp_body.read_json()?;
        let message = &json["message"];
        let text = message["content"].as_str().unwrap_or("").to_string();
        // Reasoning comes back as its OWN field, not as <think> tags in the content — so it is
        // invisible to any tag-splitting the caller does, and dropping it here is silent. This is
        // the backend the mind's local lane actually uses, so this line is what decides whether the
        // cockpit's reasoning fold has anything to show.
        let thinking = message["thinking"]
            .as_str()
            .or_else(|| message["reasoning"].as_str())
            .unwrap_or("")
            .trim()
            .to_string();

        let eval_count = json["eval_count"].as_u64().unwrap_or(0) as usize;
        let prompt_eval_count = json["prompt_eval_count"].as_u64().unwrap_or(0) as usize;

        let api_tool_calls = Self::parse_api_tool_calls(message);
        let tool_calls = if !api_tool_calls.is_empty() {
            api_tool_calls.iter().filter_map(ToolCall::from_api).collect()
        } else {
            chat_template::parse_tool_calls(&text)
        };

        Ok(LLMResponse {
            thinking,
            text,
            prompt_tokens: prompt_eval_count,
            completion_tokens: eval_count,
            tool_calls,
            api_tool_calls,
            stop_reason: if json["done"].as_bool() == Some(true) { "stop".to_string() } else { "length".to_string() },
        })
    }

    fn ollama_chat_streaming(
        &self,
        messages: &[ChatMessage],
        config: &GenerationConfig,
        tools: Option<&[serde_json::Value]>,
        on_token: &mut dyn FnMut(&str),
    ) -> Result<LLMResponse> {
        let body = self.build_ollama_body(messages, config, tools, true);
        let resp_body = self.send_ollama_request(&body)?;

        Self::parse_ollama_stream(BufReader::new(resp_body.into_reader()), on_token)
    }

    /// The Ollama stream loop as a function over any reader, so a transcript can drive it.
    pub(crate) fn parse_ollama_stream<R: std::io::BufRead>(
        reader: R,
        on_token: &mut dyn FnMut(&str),
    ) -> Result<LLMResponse> {
        let mut full_text = String::new();
        // E.VOICE2 (Yantrik Mind): the reasoning arrives in its own `thinking` field, as on the
        // plain path (`ollama_chat`), which keeps it; the stream used to drop it.
        let mut thinking = String::new();
        let mut stop_reason = "stop".to_string();
        let mut eval_count = 0usize;
        let mut prompt_eval_count = 0usize;
        let mut api_tool_calls = Vec::new();

        for line_result in reader.lines() {
            // E.VOICE2: a cancelled request stops reading here; dropping the body closes it.
            crate::call_cancel::check()?;
            let line: String = line_result.context("reading Ollama stream line")?;
            if line.trim().is_empty() { continue; }

            let chunk: serde_json::Value = match serde_json::from_str(&line) {
                Ok(v) => v,
                Err(_) => continue,
            };

            if let Some(content) = chunk["message"]["content"].as_str() {
                if !content.is_empty() {
                    on_token(content);
                    full_text.push_str(content);
                }
            }
            if let Some(t) = chunk["message"]["thinking"].as_str() {
                thinking.push_str(t);
            }

            let tc = Self::parse_api_tool_calls(&chunk["message"]);
            if !tc.is_empty() { api_tool_calls = tc; }

            if chunk["done"].as_bool() == Some(true) {
                eval_count = chunk["eval_count"].as_u64().unwrap_or(0) as usize;
                // E.USAGE1: the final chunk carries the prompt count too, and it was discarded.
                prompt_eval_count = chunk["prompt_eval_count"].as_u64().unwrap_or(0) as usize;
                if chunk["done_reason"].as_str() == Some("length") {
                    stop_reason = "length".to_string();
                }
            }
        }

        let tool_calls = if !api_tool_calls.is_empty() {
            api_tool_calls.iter().filter_map(ToolCall::from_api).collect()
        } else {
            chat_template::parse_tool_calls(&full_text)
        };

        Ok(LLMResponse {
            thinking: thinking.trim().to_string(),
            text: full_text,
            prompt_tokens: prompt_eval_count,
            completion_tokens: eval_count,
            tool_calls,
            api_tool_calls,
            stop_reason,
        })
    }

    // ── OpenAI-compatible API (/v1/chat/completions) ──

    fn build_openai_body(
        &self,
        messages: &[ChatMessage],
        config: &GenerationConfig,
        tools: Option<&[serde_json::Value]>,
        stream: bool,
    ) -> serde_json::Value {
        let msgs: Vec<serde_json::Value> = messages
            .iter()
            .map(|m| Self::serialize_message(m, false))
            .collect();

        let mut body = serde_json::json!({
            "model": self.model,
            "messages": msgs,
            "max_tokens": config.max_tokens,
            "temperature": config.temperature,
            "stream": stream,
        });

        // E.USAGE1: an OpenAI-compatible server sends NO usage record on a stream unless it is
        // asked. Without this every streamed generation was accounted as zero tokens -- in the
        // mind's own spend log, in the benchmark's receipt, and to the rate learner that sizes a
        // clamped authoring budget. Measured against the local model on 2026-09-06: streaming
        // without the option returns no usage, with it returns the same counts the non-streaming
        // call reports. `YM_STREAM_USAGE=0` withdraws it for a provider that rejects the field,
        // without a rebuild.
        if stream && std::env::var("YM_STREAM_USAGE").map(|v| v.trim() != "0").unwrap_or(true) {
            body["stream_options"] = serde_json::json!({ "include_usage": true });
        }

        if let Some(p) = config.top_p {
            body["top_p"] = serde_json::json!(p);
        }
        if !config.stop.is_empty() {
            body["stop"] = serde_json::json!(config.stop);
        }
        if config.repeat_penalty != 1.0 {
            body["frequency_penalty"] = serde_json::json!((config.repeat_penalty - 1.0).clamp(-2.0, 2.0));
        }

        // Per-call think override wins over the preset (see build_ollama_body). On the OpenAI-compat
        // path Ollama ignores `think`, so also emit reasoning_effort:"none" when forcing OFF.
        match crate::think_policy::think_setting_for_model(&self.model, config.think) {
            Some(setting) => {
                body["think"] = setting.native_json();
                if let Some(effort) = setting.reasoning_effort() {
                    body["reasoning_effort"] = serde_json::json!(effort);
                }
            }
            None if self.presets.disable_thinking => { body["think"] = serde_json::json!(false); body["reasoning_effort"] = serde_json::json!("none"); }
            None => {}
        }

        if let Some(tools) = tools {
            if !tools.is_empty() {
                body["tools"] = serde_json::json!(tools);
            }
        }

        body
    }

    fn openai_endpoint_url(&self) -> String {
        format!("{}/chat/completions", self.base_url.trim_end_matches('/'))
    }

    fn send_openai_request(&self, body: &serde_json::Value) -> Result<ureq::Body> {
        let url = self.openai_endpoint_url();
        let body_str = serde_json::to_string(body)?;

        let agent = self.build_chat_agent();
        let mut req = agent.post(&url).header("Content-Type", "application/json");
        for (k, v) in self.auth_headers() {
            req = req.header(&k, &v);
        }

        let resp = req
            .send(body_str.as_bytes())
            .map_err(|e| crate::call_timeout::describe_send_error(&self.model, e))
            .with_context(|| {
                format!(
                    "OpenAI-compatible API request to {} failed",
                    crate::call_timeout::shown_endpoint(&url)
                )
            })?;
        if !resp.status().is_success() {
            return Err(Self::status_error(resp).context(format!(
                "OpenAI-compatible API request to {} failed",
                crate::call_timeout::shown_endpoint(&url)
            )));
        }

        Ok(resp.into_body())
    }

    fn openai_chat(
        &self,
        messages: &[ChatMessage],
        config: &GenerationConfig,
        tools: Option<&[serde_json::Value]>,
    ) -> Result<LLMResponse> {
        let body = self.build_openai_body(messages, config, tools, false);
        let mut resp_body = self.send_openai_request(&body)?;

        let json: serde_json::Value = resp_body.read_json()?;
        let message = &json["choices"][0]["message"];

        let text = message["content"].as_str().unwrap_or("").to_string();
        let prompt_tokens = json["usage"]["prompt_tokens"].as_u64().unwrap_or(0) as usize;
        let completion_tokens = json["usage"]["completion_tokens"].as_u64().unwrap_or(0) as usize;
        let stop_reason = json["choices"][0]["finish_reason"].as_str().unwrap_or("stop").to_string();

        let api_tool_calls = Self::parse_api_tool_calls(message);
        let tool_calls = if !api_tool_calls.is_empty() {
            api_tool_calls.iter().filter_map(ToolCall::from_api).collect()
        } else {
            chat_template::parse_tool_calls(&text)
        };

        Ok(LLMResponse {
            thinking: String::new(),
            text,
            prompt_tokens,
            completion_tokens,
            tool_calls,
            api_tool_calls,
            stop_reason,
        })
    }

    fn openai_chat_streaming(
        &self,
        messages: &[ChatMessage],
        config: &GenerationConfig,
        tools: Option<&[serde_json::Value]>,
        on_token: &mut dyn FnMut(&str),
    ) -> Result<LLMResponse> {
        let body = self.build_openai_body(messages, config, tools, true);
        let resp_body = self.send_openai_request(&body)?;

        Self::parse_openai_stream(BufReader::new(resp_body.into_reader()), on_token)
    }

    /// The SSE loop as a function over any reader, so a RECORDED transcript can drive it.
    ///
    /// E.USAGE1: it reported `prompt_tokens: 0, completion_tokens: 0` for every streamed
    /// generation, which is why four readings could compare the mind's model REQUESTS but not its
    /// tokens, and why the rate learner never saw a sample from this path.
    pub(crate) fn parse_openai_stream<R: std::io::BufRead>(
        reader: R,
        on_token: &mut dyn FnMut(&str),
    ) -> Result<LLMResponse> {
        let mut full_text = String::new();
        let mut prompt_tokens = 0usize;
        let mut completion_tokens = 0usize;
        let mut stop_reason = "stop".to_string();
        let mut tc_ids: HashMap<usize, String> = HashMap::new();
        let mut tc_names: HashMap<usize, String> = HashMap::new();
        let mut tc_args: HashMap<usize, String> = HashMap::new();

        for line_result in reader.lines() {
            // E.VOICE2: a cancelled request stops reading here; dropping the body closes it.
            crate::call_cancel::check()?;
            let line: String = line_result.context("reading SSE line")?;
            let data = match line.strip_prefix("data: ") {
                Some(d) => d,
                None => continue,
            };
            if data == "[DONE]" { break; }

            let chunk: serde_json::Value = match serde_json::from_str(data) {
                Ok(v) => v,
                Err(_) => continue,
            };

            // E.USAGE1: the usage record arrives in its OWN final chunk, whose `choices` array is
            // EMPTY -- so it must be read here, before anything reaches for `choices[0]`.
            if chunk["usage"].is_object() {
                if let Some(n) = chunk["usage"]["prompt_tokens"].as_u64() {
                    prompt_tokens = n as usize;
                }
                if let Some(n) = chunk["usage"]["completion_tokens"].as_u64() {
                    completion_tokens = n as usize;
                }
            }

            let delta = &chunk["choices"][0]["delta"];

            if let Some(content) = delta["content"].as_str() {
                on_token(content);
                full_text.push_str(content);
            }

            if let Some(tool_calls) = delta["tool_calls"].as_array() {
                for tc_delta in tool_calls {
                    let idx = tc_delta["index"].as_u64().unwrap_or(0) as usize;
                    if let Some(id) = tc_delta["id"].as_str() {
                        tc_ids.insert(idx, id.to_string());
                    }
                    if let Some(name) = tc_delta["function"]["name"].as_str() {
                        tc_names.entry(idx).or_default().push_str(name);
                    }
                    if let Some(args) = tc_delta["function"]["arguments"].as_str() {
                        tc_args.entry(idx).or_default().push_str(args);
                    }
                }
            }

            if let Some(reason) = chunk["choices"][0]["finish_reason"].as_str() {
                stop_reason = reason.to_string();
            }
        }

        let mut api_tool_calls = Vec::new();
        let max_idx = tc_names.keys().copied().max().unwrap_or(0);
        for idx in 0..=max_idx {
            if let Some(name) = tc_names.get(&idx) {
                let id = tc_ids.get(&idx).cloned().unwrap_or_else(|| format!("call_{idx}"));
                let arguments = tc_args.get(&idx).cloned().unwrap_or_else(|| "{}".to_string());
                api_tool_calls.push(ApiToolCall {
                    id,
                    call_type: "function".to_string(),
                    function: ApiToolCallFunction {
                        name: name.clone(),
                        arguments,
                    },
                });
            }
        }

        let tool_calls = if !api_tool_calls.is_empty() {
            api_tool_calls.iter().filter_map(ToolCall::from_api).collect()
        } else {
            chat_template::parse_tool_calls(&full_text)
        };

        Ok(LLMResponse {
            thinking: String::new(),
            text: full_text,
            prompt_tokens,
            completion_tokens,
            tool_calls,
            api_tool_calls,
            stop_reason,
        })
    }

    /// Probe the provider's health by sending a minimal request.
    /// Returns latency in milliseconds on success.
    pub fn probe_health(&self) -> Result<u64> {
        let start = std::time::Instant::now();

        if self.presets.is_ollama {
            // Ollama: just check /api/tags
            let url = format!("{}/api/tags", self.ollama_base_url());
            let agent = self.build_agent();
            let _resp = agent.get(&url)
                .call()
                .context("Ollama health probe failed")?;
        } else {
            // OpenAI-compatible: send a tiny models list request
            let url = format!("{}/models", self.base_url.trim_end_matches('/'));
            let agent = self.build_agent();
            let mut req = agent.get(&url);
            for (k, v) in self.auth_headers() {
                req = req.header(&k, &v);
            }
            let _resp = req.call().context("OpenAI health probe failed")?;
        }

        Ok(start.elapsed().as_millis() as u64)
    }
}

impl LLMBackend for GenericOpenAIBackend {
    fn chat(
        &self,
        messages: &[ChatMessage],
        config: &GenerationConfig,
        tools: Option<&[serde_json::Value]>,
    ) -> Result<LLMResponse> {
        if self.presets.is_ollama {
            self.ollama_chat(messages, config, tools)
        } else {
            self.openai_chat(messages, config, tools)
        }
    }

    fn chat_streaming(
        &self,
        messages: &[ChatMessage],
        config: &GenerationConfig,
        tools: Option<&[serde_json::Value]>,
        on_token: &mut dyn FnMut(&str),
    ) -> Result<LLMResponse> {
        if self.presets.is_ollama {
            self.ollama_chat_streaming(messages, config, tools, on_token)
        } else {
            self.openai_chat_streaming(messages, config, tools, on_token)
        }
    }

    fn count_tokens(&self, text: &str) -> Result<usize> {
        // Approximate: 1 token ≈ 4 chars
        Ok(text.len() / 4)
    }

    fn backend_name(&self) -> &str {
        "api"
    }

    fn model_id(&self) -> &str {
        &self.model
    }
}

#[cfg(test)]
mod think_body_tests {
    use super::*;
    use crate::types::{ChatMessage, GenerationConfig};

    /// The tests below mutate one process-wide env var; they must not interleave.
    static ENV_LOCK: std::sync::Mutex<()> = std::sync::Mutex::new(());
    fn env_guard() -> std::sync::MutexGuard<'static, ()> {
        ENV_LOCK.lock().unwrap_or_else(|e| e.into_inner())
    }

    fn bodies(cfg_think: Option<bool>) -> (serde_json::Value, serde_json::Value) {
        let b = GenericOpenAIBackend::for_provider("ollama", "http://127.0.0.1:11434", None, "qwen3.8:27b-q4_K_M");
        let msgs = vec![ChatMessage::user("hi")];
        let cfg = GenerationConfig { think: cfg_think, ..GenerationConfig::default() };
        (b.build_ollama_body(&msgs, &cfg, None, false), b.build_openai_body(&msgs, &cfg, None, false))
    }

    /// E.THINKLVL1: a level is written as the measured wire words on both paths.
    #[test]
    fn a_per_model_level_reaches_both_bodies_as_the_measured_wire_words() {
        let _env = env_guard();
        std::env::set_var("YM_THINK_MODELS", "qwen3.8=low");
        let (native, openai) = bodies(Some(false));
        std::env::remove_var("YM_THINK_MODELS");
        assert_eq!(native["think"], serde_json::json!("low"), "{native}");
        assert_eq!(openai["reasoning_effort"], serde_json::json!("low"), "{openai}");
        assert_eq!(openai["think"], serde_json::json!("low"), "{openai}");
    }

    /// E.THINKLVL1: the profile's measured default applies when nothing else decides.
    #[test]
    fn the_measured_family_default_is_low_when_nothing_else_decides() {
        let _env = env_guard();
        std::env::remove_var("YM_THINK_MODELS");
        let (native, _) = bodies(None);
        assert_eq!(native["think"], serde_json::json!("low"), "qwen3.8 profile default: {native}");
    }

    /// E.THINKOBS1: the request body is the last code before the wire, so it is the observable.
    #[test]
    fn a_per_model_entry_reaches_both_request_bodies_and_beats_the_caller() {
        let _env = env_guard();
        std::env::set_var("YM_THINK_MODELS", "qwen3.8=off");
        let (native, openai) = bodies(Some(true));
        std::env::remove_var("YM_THINK_MODELS");
        assert_eq!(native["think"], serde_json::json!(false), "{native}");
        assert_eq!(openai["think"], serde_json::json!(false), "{openai}");
        assert_eq!(openai["reasoning_effort"], serde_json::json!("none"), "{openai}");
        let (native2, _) = bodies(Some(true));
        assert_eq!(native2["think"], serde_json::json!(true), "without the entry the caller's request stands: {native2}");
    }
}

/// E.MSG3: a refused connection names the address it was refused at, and the cause.
#[cfg(test)]
mod endpoint_in_error_tests {
    use super::*;
    use crate::traits::LLMBackend;
    use crate::types::{ChatMessage, GenerationConfig};

    #[test]
    fn a_refused_connection_names_where_it_went() {
        // A port that was just free: bound, read, released.
        let port = std::net::TcpListener::bind("127.0.0.1:0").unwrap().local_addr().unwrap().port();
        let be = GenericOpenAIBackend::for_provider(
            "openrouter",
            format!("http://127.0.0.1:{port}/v1"),
            Some("k".into()),
            "m",
        );
        let e = be
            .chat(&[ChatMessage::user("hi")], &GenerationConfig::default(), None)
            .expect_err("nothing is listening");
        let text = format!("{e:#}");
        assert!(text.contains(&format!("127.0.0.1:{port}")), "the address is missing: {text}");
        assert!(text.split(" failed").nth(1).is_some_and(|c| !c.trim().is_empty()), "the cause is missing: {text}");
    }
}

#[cfg(test)]
mod usage_tests {
    use super::*;
    use crate::types::{ChatMessage, GenerationConfig};

    fn backend() -> GenericOpenAIBackend {
        GenericOpenAIBackend::for_provider(
            "ollama",
            "http://127.0.0.1:11434",
            None,
            "gpt-oss-backup:20b",
        )
    }

    /// The transcript the local model actually sent on 2026-09-06 when asked for usage: the record
    /// arrives in its own final chunk, whose `choices` array is EMPTY. Four readings could compare
    /// the mind's model requests but not its tokens because this was thrown away.
    #[test]
    fn the_recorded_local_stream_reports_the_counts_the_server_sent() {
        let sse = std::fs::read(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/fixtures/stream/oss20_usage.sse"
        ))
        .expect("fixture present");
        let mut seen = String::new();
        let r = GenericOpenAIBackend::parse_openai_stream(
            std::io::BufReader::new(&sse[..]),
            &mut |t| seen.push_str(t),
        )
        .expect("the transcript parses");
        assert_eq!(r.prompt_tokens, 76, "the prompt count the server reported");
        assert_eq!(r.completion_tokens, 40, "the completion count the server reported");
        assert_eq!(r.stop_reason, "length", "the finish reason still comes from its own chunk");
    }

    /// A server that sends no usage record is unchanged: zero, and nothing invented.
    #[test]
    fn a_stream_without_a_usage_record_parses_and_claims_nothing() {
        let sse = "data: {\"choices\":[{\"index\":0,\"delta\":{\"content\":\"hi\"}}]}\n\
                   data: {\"choices\":[{\"index\":0,\"delta\":{},\"finish_reason\":\"stop\"}]}\n\
                   data: [DONE]\n";
        let mut seen = String::new();
        let r = GenericOpenAIBackend::parse_openai_stream(
            std::io::BufReader::new(sse.as_bytes()),
            &mut |t| seen.push_str(t),
        )
        .expect("parses");
        assert_eq!(seen, "hi", "the text still streams");
        assert_eq!((r.prompt_tokens, r.completion_tokens), (0, 0));
        assert_eq!(r.stop_reason, "stop");
    }

    /// The ask itself. Measured against the local model: streaming WITHOUT this option returns no
    /// usage at all, so the counts above can only exist because the body requested them.
    #[test]
    fn a_streaming_body_asks_for_the_usage_record_and_a_blocking_one_does_not() {
        let b = backend();
        let msgs = vec![ChatMessage::user("hi")];
        let cfg = GenerationConfig::default();
        let streaming = b.build_openai_body(&msgs, &cfg, None, true);
        assert_eq!(
            streaming["stream_options"]["include_usage"],
            serde_json::json!(true),
            "a stream must ask, or the server sends nothing"
        );
        let blocking = b.build_openai_body(&msgs, &cfg, None, false);
        assert!(
            blocking.get("stream_options").is_none(),
            "a blocking call already gets usage and must not carry a streaming-only field"
        );
    }
}

#[cfg(test)]
mod auth_header_tests {
    use super::GenericOpenAIBackend;

    fn authorization(b: &GenericOpenAIBackend) -> Option<String> {
        b.auth_headers().into_iter().find(|(k, _)| k == "Authorization").map(|(_, v)| v)
    }

    #[test]
    fn an_ollama_behind_a_gate_is_sent_the_key_it_was_given() {
        let gated = GenericOpenAIBackend::for_provider("ollama", "http://10.99.0.1:8443", Some("k".into()), "m");
        assert_eq!(authorization(&gated).as_deref(), Some("Bearer k"));
    }

    #[test]
    fn no_key_or_a_blank_one_sends_no_header() {
        for key in [None, Some(String::new()), Some("  ".to_string())] {
            for provider in ["ollama", "openai", "nvidia-nim"] {
                let b = GenericOpenAIBackend::for_provider(provider, "http://localhost:11434/v1", key.clone(), "m");
                assert_eq!(authorization(&b), None, "{provider} {key:?}");
            }
        }
    }

    #[test]
    fn auth_style_none_never_sends_a_key() {
        let b = GenericOpenAIBackend::new("http://x/v1", Some("k".into()), "m", "none", Default::default());
        assert_eq!(authorization(&b), None);
    }

    /// The request head an Ollama backend with `key` actually puts on the wire, read by a
    /// listener standing in for the server.
    fn request_head_sent(key: Option<&str>) -> String {
        use crate::traits::LLMBackend;
        use crate::types::{ChatMessage, GenerationConfig};
        use std::io::{Read, Write};

        let listener = std::net::TcpListener::bind("127.0.0.1:0").unwrap();
        let port = listener.local_addr().unwrap().port();
        let server = std::thread::spawn(move || {
            // A deadline on every wait, so a client that never connects or stalls fails the test
            // instead of hanging it.
            let limit = std::time::Duration::from_secs(10);
            let deadline = std::time::Instant::now() + limit;
            listener.set_nonblocking(true).unwrap();
            let mut conn = loop {
                match listener.accept() {
                    Ok((conn, _)) => break conn,
                    Err(e) if e.kind() == std::io::ErrorKind::WouldBlock && std::time::Instant::now() < deadline => {
                        std::thread::sleep(std::time::Duration::from_millis(10))
                    }
                    Err(e) => panic!("no request reached the listener within {limit:?}: {e}"),
                }
            };
            conn.set_nonblocking(false).unwrap();
            conn.set_read_timeout(Some(limit)).unwrap();
            let mut got = Vec::new();
            let mut buf = [0u8; 4096];
            let head_end = loop {
                let n = conn.read(&mut buf).unwrap();
                assert!(n > 0, "the client hung up before sending a request");
                got.extend_from_slice(&buf[..n]);
                if let Some(i) = got.windows(4).position(|w| w == b"\r\n\r\n") {
                    break i;
                }
            };
            let head = String::from_utf8_lossy(&got[..head_end]).to_string();
            let body_len = head
                .lines()
                .find_map(|l| l.to_ascii_lowercase().strip_prefix("content-length:").map(|v| v.trim().parse::<usize>().unwrap()))
                .unwrap_or(0);
            while got.len() < head_end + 4 + body_len {
                let n = conn.read(&mut buf).unwrap();
                if n == 0 {
                    break;
                }
                got.extend_from_slice(&buf[..n]);
            }
            let reply = r#"{"model":"m","message":{"role":"assistant","content":"ok"},"done":true,"done_reason":"stop","prompt_eval_count":1,"eval_count":1}"#;
            write!(
                conn,
                "HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{reply}",
                reply.len()
            )
            .unwrap();
            head
        });
        let be = GenericOpenAIBackend::for_provider("ollama", format!("http://127.0.0.1:{port}"), key.map(String::from), "m");
        // Only the request matters here; the reply is whatever the server stand-in could say.
        let _ = be.chat(&[ChatMessage::user("hi")], &GenerationConfig::default(), None);
        server.join().unwrap()
    }

    #[test]
    fn on_the_wire_a_plain_ollama_sees_no_authorization_and_a_gated_one_sees_its_key() {
        for key in [None, Some(""), Some("  ")] {
            let head = request_head_sent(key).to_ascii_lowercase();
            assert!(head.starts_with("post /api/chat "), "{head}");
            assert!(!head.contains("\nauthorization:"), "{key:?} sent an Authorization header:\n{head}");
        }
        let head = request_head_sent(Some("gate-key"));
        assert!(
            head.lines().any(|l| l.eq_ignore_ascii_case("authorization: Bearer gate-key")),
            "the gated request carries its key:\n{head}"
        );
    }
}

/// E.VOICE2 (Yantrik Mind): a cancelled stream stops between lines, and an unset or absent flag
/// changes nothing.
#[cfg(test)]
mod cancel_tests {
    use super::*;
    use std::sync::atomic::{AtomicBool, Ordering};
    use std::sync::Arc;

    /// A server that does not stop writing: one content chunk per line, ten thousand lines (a
    /// bound, so a missing check fails the test instead of hanging it).
    struct Endless(usize);

    impl std::io::Read for Endless {
        fn read(&mut self, buf: &mut [u8]) -> std::io::Result<usize> {
            if self.0 >= 10_000 {
                return Ok(0);
            }
            self.0 += 1;
            let line = b"data: {\"choices\":[{\"index\":0,\"delta\":{\"content\":\"w \"}}]}\n";
            let n = line.len().min(buf.len());
            buf[..n].copy_from_slice(&line[..n]);
            Ok(n)
        }
    }

    #[test]
    fn a_cancelled_stream_stops_within_one_line() {
        let flag = Arc::new(AtomicBool::new(false));
        let _g = crate::call_cancel::install_cancel(Some(flag.clone()));
        let mut tokens = 0usize;
        let r = GenericOpenAIBackend::parse_openai_stream(std::io::BufReader::new(Endless(0)), &mut |_| {
            tokens += 1;
            if tokens == 3 {
                flag.store(true, Ordering::SeqCst);
            }
        });
        let e = r.expect_err("the stream must end by being cancelled, not by running out");
        assert!(format!("{e:#}").contains("cancelled"), "{e:#}");
        assert_eq!(tokens, 3, "no line was read after the flag was set");
    }

    /// The streamed Ollama path returns what the plain one does: the reasoning from its own field,
    /// the words, and a native tool call.
    #[test]
    fn the_ollama_stream_keeps_its_thinking_and_its_tool_call() {
        let lines = [
            r#"{"message":{"role":"assistant","content":"","thinking":"The person wants "}}"#,
            r#"{"message":{"role":"assistant","content":"","thinking":"dinner ideas."}}"#,
            r#"{"message":{"role":"assistant","content":"Let me look."}}"#,
            r#"{"message":{"role":"assistant","content":"","tool_calls":[{"id":"call_1","function":{"name":"recall","arguments":{"query":"food"}}}]}}"#,
            r#"{"message":{"role":"assistant","content":""},"done":true,"eval_count":9,"prompt_eval_count":40}"#,
        ];
        let body = lines.join("
");
        let mut seen = String::new();
        let r = GenericOpenAIBackend::parse_ollama_stream(std::io::BufReader::new(body.as_bytes()), &mut |t| seen.push_str(t))
            .expect("parses");
        assert_eq!(r.thinking, "The person wants dinner ideas.");
        assert_eq!((seen.as_str(), r.text.as_str()), ("Let me look.", "Let me look."));
        assert_eq!(r.tool_calls.len(), 1, "{:?}", r.tool_calls);
        assert_eq!(r.tool_calls[0].name, "recall");
        assert_eq!((r.prompt_tokens, r.completion_tokens), (40, 9));
    }

    #[test]
    fn an_installed_flag_that_is_never_set_changes_nothing() {
        let _g = crate::call_cancel::install_cancel(Some(Arc::new(AtomicBool::new(false))));
        let sse = "data: {\"choices\":[{\"index\":0,\"delta\":{\"content\":\"hi \"}}]}\n\
                   data: {\"choices\":[{\"index\":0,\"delta\":{\"content\":\"there\"}}]}\n\
                   data: [DONE]\n";
        let mut seen = String::new();
        let r = GenericOpenAIBackend::parse_openai_stream(std::io::BufReader::new(sse.as_bytes()), &mut |t| seen.push_str(t))
            .expect("parses");
        assert_eq!((seen.as_str(), r.text.as_str()), ("hi there", "hi there"));
    }
}

/// Yantrik Mind E.QUOTA1: an error status keeps what the server said, behind the same prefix.
#[cfg(test)]
mod status_body_tests {
    use super::*;
    use crate::types::{ChatMessage, GenerationConfig};
    use std::io::{BufRead, BufReader, Read, Write};

    /// What Ollama Cloud answered on 2026-10-11 once the account's weekly allowance was spent.
    const OLLAMA_WEEKLY: &str = r#"{"error":{"message":"You reached your Pro weekly limit. Max is $100/month for $300 of usage, with no 5-hour or weekly caps: https://ollama.com/settings/billing (ref: 144bb863-fdc4-4827-83f0-b72167486392)","type":"api_error","param":null,"code":null}}"#;

    /// One answer from a local server: `status`, extra header lines, `body`.
    fn serve_once(status: &'static str, headers: &'static str, body: String) -> String {
        let listener = std::net::TcpListener::bind("127.0.0.1:0").unwrap();
        let port = listener.local_addr().unwrap().port();
        std::thread::spawn(move || {
            let (mut stream, _) = listener.accept().unwrap();
            let mut reader = BufReader::new(stream.try_clone().unwrap());
            let mut len = 0usize;
            loop {
                let mut line = String::new();
                reader.read_line(&mut line).unwrap();
                if let Some(v) = line.to_ascii_lowercase().strip_prefix("content-length:") {
                    len = v.trim().parse().unwrap();
                }
                if line.trim().is_empty() {
                    break;
                }
            }
            let mut sink = vec![0u8; len];
            let _ = reader.read_exact(&mut sink);
            let _ = write!(stream, "HTTP/1.1 {status}\r\n{headers}Content-Length: {}\r\nConnection: close\r\n\r\n{body}", body.len());
        });
        format!("http://127.0.0.1:{port}/v1")
    }

    fn chat_err(base: String, ollama: bool) -> String {
        let b = if ollama {
            GenericOpenAIBackend::for_provider("ollama", base.trim_end_matches("/v1"), None, "m")
        } else {
            GenericOpenAIBackend::for_provider("openai", base, Some("k".into()), "m")
        };
        format!("{:#}", b.chat(&[ChatMessage::user("hi")], &GenerationConfig::default(), None).unwrap_err())
    }

    #[test]
    fn a_429_keeps_the_prefix_and_says_why() {
        let e = chat_err(serve_once("429 Too Many Requests", "Content-Type: application/json\r\n", OLLAMA_WEEKLY.into()), false);
        assert!(e.contains("http status: 429: You reached your Pro weekly limit."), "{e}");
        assert!(e.starts_with("OpenAI-compatible API request to http://127.0.0.1:"), "the context is kept: {e}");
        let e = chat_err(serve_once("429 Too Many Requests", "", OLLAMA_WEEKLY.into()), true);
        assert!(e.contains("http status: 429: You reached your Pro weekly limit."), "the native Ollama path too: {e}");
    }

    #[test]
    fn a_retry_after_is_kept_and_an_html_page_is_one_bounded_line() {
        let e = chat_err(serve_once("429 Too Many Requests", "Retry-After: 120\r\n", r#"{"error":"slow down"}"#.into()), false);
        assert!(e.contains("http status: 429 (retry after 120): slow down"), "{e}");
        let page = format!("<html>\n<body>\n{}</body></html>", "Bad gateway. ".repeat(80));
        let e = chat_err(serve_once("502 Bad Gateway", "Content-Type: text/html\r\n", page), false);
        let said = e.split("http status: 502: ").nth(1).expect("the prefix");
        assert!(!said.contains('\n') && said.chars().count() <= 300, "{said}");
    }

    #[test]
    fn a_success_is_unchanged() {
        let base = serve_once("200 OK", "Content-Type: application/json\r\n", r#"{"choices":[{"message":{"content":"hello"},"finish_reason":"stop"}]}"#.into());
        let b = GenericOpenAIBackend::for_provider("openai", base, Some("k".into()), "m");
        assert_eq!(b.chat(&[ChatMessage::user("hi")], &GenerationConfig::default(), None).unwrap().text, "hello");
    }
}
