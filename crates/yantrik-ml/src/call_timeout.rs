//! The per-call timeout every HTTP client in this crate uses. One knob, one place, so a local
//! lane that takes 312 s to author a project (measured) is a configuration, not a code change.
//!
//! Precedence (E.PROFILE1): `YM_LLM_TIMEOUT_MODELS` ("qwen3.8=600,gpt-oss=120", matched by
//! substring of the model name) → `YM_LLM_TIMEOUT_S` (global) → the model's
//! `ModelCapabilityProfile::call_timeout_s` (measured per tier) → 300 s.

const DEFAULT_SECS: u64 = 300;
const MAX_SECS: u64 = 86_400;

fn parse_secs(v: &str) -> Option<u64> {
    v.trim().parse::<u64>().ok().filter(|s| *s > 0 && *s <= MAX_SECS)
}

/// Pure precedence, so it is testable without env races.
pub fn resolve_timeout_secs(
    model: &str,
    per_model_csv: Option<&str>,
    global: Option<&str>,
    profile_secs: Option<u64>,
) -> u64 {
    per_model_csv
        .and_then(|csv| crate::model_overrides::lookup(csv, model))
        .and_then(parse_secs)
        .or_else(|| global.and_then(parse_secs))
        .or(profile_secs)
        .unwrap_or(DEFAULT_SECS)
}

/// The timeout for a call to `model`: the per-model env, the global env, the profile, 300 s.
pub fn call_timeout_for(model: &str) -> std::time::Duration {
    let profile = crate::capability::ModelCapabilityProfile::from_model_name(model).call_timeout_s;
    std::time::Duration::from_secs(resolve_timeout_secs(
        model,
        std::env::var("YM_LLM_TIMEOUT_MODELS").ok().as_deref(),
        std::env::var("YM_LLM_TIMEOUT_S").ok().as_deref(),
        profile,
    ))
}

/// The model-less timeout: `YM_LLM_TIMEOUT_S`, else 300 s. For callers that have no model name.
pub fn call_timeout() -> std::time::Duration {
    std::time::Duration::from_secs(resolve_timeout_secs(
        "",
        None,
        std::env::var("YM_LLM_TIMEOUT_S").ok().as_deref(),
        None,
    ))
}

/// The sentence a client writes when ITS call timed out: the seconds it actually waited and the
/// model it waited for (E.MSG2). One source of truth, at the site that knows both — the lane
/// above relays it instead of guessing from a model-less knob (which said "300 s" for a 5 s
/// per-model timeout on 2026-09-04).
pub fn timeout_sentence(model: &str, secs: u64) -> String {
    format!("timed out after {secs} s waiting for {model} (YM_LLM_TIMEOUT_MODELS / YM_LLM_TIMEOUT_S)")
}

/// Turn a transport error into the error the caller sees: a ureq timeout becomes the sentence
/// above; anything else passes through unchanged.
#[cfg(feature = "api-llm")]
pub fn describe_send_error(model: &str, e: ureq::Error) -> anyhow::Error {
    match e {
        ureq::Error::Timeout(_) => {
            anyhow::anyhow!("{}", timeout_sentence(model, call_timeout_for(model).as_secs()))
        }
        other => anyhow::Error::from(other),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_timeout_sentence_names_the_seconds_the_model_and_both_knobs() {
        let t = timeout_sentence("qwen3.8:27b-q4_K_M", 5);
        assert!(t.starts_with("timed out after 5 s"), "{t}");
        assert!(t.contains("qwen3.8:27b-q4_K_M") && t.contains("YM_LLM_TIMEOUT_MODELS") && t.contains("YM_LLM_TIMEOUT_S"), "{t}");
    }

    #[cfg(feature = "api-llm")]
    #[test]
    fn a_ureq_timeout_becomes_the_sentence_and_other_errors_pass_through() {
        let e = describe_send_error("qwen3.8:27b", ureq::Error::Timeout(ureq::Timeout::Global));
        let text = format!("{e:#}");
        assert!(text.contains("timed out after ") && text.contains("qwen3.8:27b"), "{text}");
        let other = describe_send_error("m", ureq::Error::HostNotFound);
        assert!(!format!("{other:#}").contains("timed out after"), "{other:#}");
    }

    #[test]
    fn precedence_is_per_model_then_global_then_profile_then_default() {
        let m = "qwen3.8:27b-q4_K_M";
        assert_eq!(resolve_timeout_secs(m, Some("qwen3.8=900"), Some("120"), Some(600)), 900);
        assert_eq!(resolve_timeout_secs(m, Some("gpt-oss=900"), Some("120"), Some(600)), 120);
        assert_eq!(resolve_timeout_secs(m, None, None, Some(600)), 600);
        assert_eq!(resolve_timeout_secs(m, None, None, None), 300);
    }

    #[test]
    fn a_bad_value_at_any_level_falls_through_to_the_next() {
        let m = "qwen3.8:27b";
        assert_eq!(resolve_timeout_secs(m, Some("qwen3.8=lots"), Some("120"), None), 120);
        assert_eq!(resolve_timeout_secs(m, Some("qwen3.8=0"), Some("0"), Some(600)), 600);
        assert_eq!(resolve_timeout_secs(m, Some("qwen3.8=99999999"), Some("-1"), None), 300);
    }

    #[test]
    fn the_large_tier_profile_earns_a_longer_timeout_and_a_medium_one_does_not() {
        use crate::capability::ModelCapabilityProfile as P;
        assert_eq!(P::from_model_name("qwen3.8:27b-q4_K_M").call_timeout_s, Some(600), "312 s was measured");
        // The tier boundary is the crate's (>= 14B is Large), so a 20B model earns it too.
        assert_eq!(P::from_model_name("gpt-oss-backup:20b").call_timeout_s, Some(600));
        assert_eq!(P::from_model_name("qwen3.5:9b").call_timeout_s, None, "Medium: unmeasured, default");
        assert_eq!(P::from_model_name("gemma4:e4b").call_timeout_s, None);
    }

    #[test]
    fn the_model_less_knob_is_the_global_env_or_300() {
        // Not env-mutating: the pure core with the same shape the wrapper uses.
        assert_eq!(resolve_timeout_secs("", None, Some("900"), None), 900);
        assert_eq!(resolve_timeout_secs("", None, None, None), 300);
    }

    /// Every HTTP client in this crate — the legacy `ApiLLM` and the `provider/` backends — must
    /// take its timeout from the model-aware knob. E.TIMEOUT1 put the knob on a dead path while
    /// the live lane kept its own 300 s; E.PROFILE1 makes the model part of the answer, so the
    /// model-less `call_timeout()` is no longer acceptable at a client site either.
    #[test]
    fn every_timeout_site_in_every_client_uses_the_model_aware_knob() {
        for file in ["src/llm/api.rs", "src/provider/generic_openai.rs"] {
            let src = std::fs::read_to_string(concat!(env!("CARGO_MANIFEST_DIR"), "/").to_string() + file).unwrap();
            let src = src.split("#[cfg(test)]").next().unwrap_or("");
            let literal = src.matches(".timeout_global(Some(std::time::Duration::from_secs(").count();
            let model_less = src.matches("call_timeout()").count();
            let wired = src.matches("call_timeout_for(&self.model)").count();
            assert_eq!(literal, 0, "{file} still hardcodes a call timeout");
            assert_eq!(model_less, 0, "{file} uses the model-less knob at a client site");
            assert!(wired >= 1, "{file} has no model-aware timeout site");
        }
    }
}
