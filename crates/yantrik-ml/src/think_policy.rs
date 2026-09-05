//! Which thinking setting a call to a given model gets (E.PROFILE1, E.THINKLVL1). Measured
//! 2026-09-04 (n=5, native `/api/chat`): thinking divides by model AND by provider path, and the
//! useful per-model setting is a LEVEL, not a switch — gpt-oss:20b at `low` answers in 7.5 s with
//! files every run (`false`: 29 s, empty 60% of the time); qwen3.8:27b at `low`/`high` writes the
//! same three files every run where `false` is erratic. The native path honours `think` as a bool
//! or `"low"|"medium"|"high"`; the OpenAI-compatible path honours `reasoning_effort` only.
//!
//! Precedence: `YM_THINK_MODELS` ("qwen3.8=low,gpt-oss=low", on|off|low|medium|high) → what the
//! caller asked (`GenerationConfig.think`, the mind's per-role switch) → the model's profile
//! level default (measured families only) → the profile switch default → `None` (provider preset).

#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum ThinkLevel {
    Low,
    Medium,
    High,
}

impl ThinkLevel {
    pub fn as_str(self) -> &'static str {
        match self {
            ThinkLevel::Low => "low",
            ThinkLevel::Medium => "medium",
            ThinkLevel::High => "high",
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ThinkSetting {
    Off,
    On,
    Level(ThinkLevel),
}

impl ThinkSetting {
    /// The native `/api/chat` `think` field: a bool, or the level word.
    pub fn native_json(self) -> serde_json::Value {
        match self {
            ThinkSetting::Off => serde_json::json!(false),
            ThinkSetting::On => serde_json::json!(true),
            ThinkSetting::Level(l) => serde_json::json!(l.as_str()),
        }
    }
    /// The OpenAI-compatible `reasoning_effort` word. `On` has no effort word — the preset decides.
    pub fn reasoning_effort(self) -> Option<&'static str> {
        match self {
            ThinkSetting::Off => Some("none"),
            ThinkSetting::On => None,
            ThinkSetting::Level(l) => Some(l.as_str()),
        }
    }
    fn from_bool(b: bool) -> Self {
        if b { ThinkSetting::On } else { ThinkSetting::Off }
    }
}

fn parse_setting(v: &str) -> Option<ThinkSetting> {
    match v.trim().to_ascii_lowercase().as_str() {
        "on" | "true" | "1" | "yes" => Some(ThinkSetting::On),
        "off" | "false" | "0" | "no" => Some(ThinkSetting::Off),
        "low" => Some(ThinkSetting::Level(ThinkLevel::Low)),
        "medium" | "med" => Some(ThinkSetting::Level(ThinkLevel::Medium)),
        "high" => Some(ThinkSetting::Level(ThinkLevel::High)),
        _ => None,
    }
}

/// Pure precedence, testable without env races.
pub fn resolve_think_setting(
    model: &str,
    per_model_csv: Option<&str>,
    requested: Option<bool>,
    profile_level: Option<ThinkLevel>,
    profile_switch: Option<bool>,
) -> Option<ThinkSetting> {
    per_model_csv
        .and_then(|csv| crate::model_overrides::lookup(csv, model))
        .and_then(parse_setting)
        .or_else(|| requested.map(ThinkSetting::from_bool))
        .or_else(|| profile_level.map(ThinkSetting::Level))
        .or_else(|| profile_switch.map(ThinkSetting::from_bool))
}

/// The thinking setting for a call to `model` when the caller asked for `requested`.
pub fn think_setting_for_model(model: &str, requested: Option<bool>) -> Option<ThinkSetting> {
    let profile = crate::capability::ModelCapabilityProfile::from_model_name(model);
    resolve_think_setting(
        model,
        std::env::var("YM_THINK_MODELS").ok().as_deref(),
        requested,
        profile.think_level_default,
        profile.think_default,
    )
}

/// Kept for callers that only understand a switch: a level counts as "on".
pub fn resolve_think(
    model: &str,
    per_model_csv: Option<&str>,
    requested: Option<bool>,
    profile_default: Option<bool>,
) -> Option<bool> {
    resolve_think_setting(model, per_model_csv, requested, None, profile_default).map(|s| s != ThinkSetting::Off)
}

pub fn think_for_model(model: &str, requested: Option<bool>) -> Option<bool> {
    think_setting_for_model(model, requested).map(|s| s != ThinkSetting::Off)
}

#[cfg(test)]
mod tests {
    use super::*;

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
    }

    #[test]
    fn levels_are_a_setting_of_their_own_and_beat_the_callers_switch() {
        use ThinkSetting::*;
        let m = "gpt-oss-backup:20b";
        assert_eq!(resolve_think_setting(m, Some("gpt-oss=low"), Some(true), None, None), Some(Level(ThinkLevel::Low)));
        assert_eq!(resolve_think_setting(m, Some("gpt-oss=HIGH"), None, None, None), Some(Level(ThinkLevel::High)));
        assert_eq!(resolve_think_setting(m, Some("gpt-oss=med"), None, None, None), Some(Level(ThinkLevel::Medium)));
        // the caller's switch beats the profile level; the profile level beats the profile switch
        assert_eq!(resolve_think_setting(m, None, Some(false), Some(ThinkLevel::Low), Some(true)), Some(Off));
        assert_eq!(resolve_think_setting(m, None, None, Some(ThinkLevel::Low), Some(true)), Some(Level(ThinkLevel::Low)));
        assert_eq!(resolve_think_setting(m, None, None, None, Some(true)), Some(On));
        assert_eq!(resolve_think_setting(m, None, None, None, None), None);
    }

    #[test]
    fn the_wire_words_are_the_measured_ones() {
        assert_eq!(ThinkSetting::Level(ThinkLevel::Low).native_json(), serde_json::json!("low"));
        assert_eq!(ThinkSetting::Off.native_json(), serde_json::json!(false));
        assert_eq!(ThinkSetting::Off.reasoning_effort(), Some("none"));
        assert_eq!(ThinkSetting::Level(ThinkLevel::High).reasoning_effort(), Some("high"));
        assert_eq!(ThinkSetting::On.reasoning_effort(), None, "on has no effort word; the preset decides");
    }
}
