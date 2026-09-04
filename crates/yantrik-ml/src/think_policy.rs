//! Which thinking setting a call to a given model gets (E.PROFILE1). Measured 2026-09-04 (n=5):
//! thinking divides by model AND by provider path, so the decision belongs where the model is
//! known — the backend, at send time — not in the mind, which knows roles.
//!
//! Precedence: `YM_THINK_MODELS` ("qwen3.8=off,gpt-oss=on") → what the caller asked
//! (`GenerationConfig.think`, i.e. the mind's per-role knob) → the model's profile default →
//! `None`, which leaves the provider preset in charge exactly as before.

fn parse_switch(v: &str) -> Option<bool> {
    match v.trim().to_ascii_lowercase().as_str() {
        "on" | "true" | "1" | "yes" => Some(true),
        "off" | "false" | "0" | "no" => Some(false),
        _ => None,
    }
}

/// Pure precedence, testable without env races.
pub fn resolve_think(
    model: &str,
    per_model_csv: Option<&str>,
    requested: Option<bool>,
    profile_default: Option<bool>,
) -> Option<bool> {
    per_model_csv
        .and_then(|csv| crate::model_overrides::lookup(csv, model))
        .and_then(parse_switch)
        .or(requested)
        .or(profile_default)
}

/// The thinking setting for a call to `model` when the caller asked for `requested`.
pub fn think_for_model(model: &str, requested: Option<bool>) -> Option<bool> {
    let profile = crate::capability::ModelCapabilityProfile::from_model_name(model).think_default;
    resolve_think(
        model,
        std::env::var("YM_THINK_MODELS").ok().as_deref(),
        requested,
        profile,
    )
}

#[cfg(test)]
mod tests {
    use super::resolve_think;

    #[test]
    fn precedence_is_per_model_then_caller_then_profile_then_none() {
        let m = "qwen3.8:27b-q4_K_M";
        assert_eq!(resolve_think(m, Some("qwen3.8=off"), Some(true), Some(true)), Some(false));
        assert_eq!(resolve_think(m, Some("gpt-oss=off"), Some(true), Some(false)), Some(true));
        assert_eq!(resolve_think(m, None, None, Some(false)), Some(false));
        assert_eq!(resolve_think(m, None, None, None), None, "no opinion leaves the preset in charge");
    }

    #[test]
    fn an_unknown_switch_word_falls_through_instead_of_guessing() {
        assert_eq!(resolve_think("qwen3.8:27b", Some("qwen3.8=maybe"), Some(true), None), Some(true));
        assert_eq!(resolve_think("qwen3.8:27b", Some("qwen3.8=low"), None, None), None);
    }
}
