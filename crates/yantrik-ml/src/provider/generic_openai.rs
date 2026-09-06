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
    /// Auth header style: "bearer" sends `Authorization: Bearer <key>`,
    /// "none" sends no auth header.
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
        let auth_style = match provider_type {
            "ollama" => "none",
            _ => "bearer",
        };
        Self::new(base_url, api_key, model, auth_style, presets)
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

    /// Build auth and extra headers as a vec of (key, value) pairs.
    fn auth_headers(&self) -> Vec<(String, String)> {
        let mut headers = Vec::new();
        if let Some(ref key) = self.api_key {
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

        let agent = self.build_agent();
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
            .context("Ollama API request failed")?;

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

        let reader = BufReader::new(resp_body.into_reader());
        let mut full_text = String::new();
        let mut stop_reason = "stop".to_string();
        let mut eval_count = 0usize;
        let mut prompt_eval_count = 0usize;
        let mut api_tool_calls = Vec::new();

        for line_result in reader.lines() {
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
            thinking: String::new(),
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

        let agent = self.build_agent();
        let mut req = agent.post(&url).header("Content-Type", "application/json");
        for (k, v) in self.auth_headers() {
            req = req.header(&k, &v);
        }

        let resp = req
            .send(body_str.as_bytes())
            .map_err(|e| crate::call_timeout::describe_send_error(&self.model, e))
            .context("OpenAI-compatible API request failed")?;

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
