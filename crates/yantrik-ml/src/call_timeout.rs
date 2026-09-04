//! The wall-clock ceiling on one model call, as a KNOB shared by every HTTP client here.
//!
//! E.TIMEOUT1 put this on `llm/api.rs` (`ApiLLM`) only. That path is referenced from a single test
//! in the consumer, so the release binary strips it — and the LOCAL lane, the one that measured
//! 312 s and was cut, goes through `provider/generic_openai.rs`, which carried its own hardcoded
//! 300 s the knob never reached. `strings` on the deployed binary showed `YM_LLM_TIMEOUT_S` 0 times.
//! So the helper lives here, feature-independent, and BOTH clients call it.
//!
//! `YM_LLM_TIMEOUT_S` overrides; **the default is the same 300 s**, so an unset environment behaves
//! exactly as before. Every malformed value — empty, non-numeric, zero, beyond a day — falls back
//! to the default, never to something shorter: a typo must not turn a working lane into one that
//! fails closed, silently.

pub fn call_timeout() -> std::time::Duration {
    const DEFAULT_SECS: u64 = 300;
    let secs = std::env::var("YM_LLM_TIMEOUT_S")
        .ok()
        .and_then(|v| v.trim().parse::<u64>().ok())
        .filter(|s| *s > 0 && *s <= 86_400)
        .unwrap_or(DEFAULT_SECS);
    std::time::Duration::from_secs(secs)
}

#[cfg(test)]
mod tests {
    use super::call_timeout;
    static ENV_LOCK: std::sync::Mutex<()> = std::sync::Mutex::new(());
    fn with_env(value: Option<&str>, f: impl FnOnce()) {
        let _g = ENV_LOCK.lock().unwrap_or_else(|e| e.into_inner());
        let prev = std::env::var("YM_LLM_TIMEOUT_S").ok();
        match value { Some(v) => std::env::set_var("YM_LLM_TIMEOUT_S", v), None => std::env::remove_var("YM_LLM_TIMEOUT_S") }
        f();
        match prev { Some(p) => std::env::set_var("YM_LLM_TIMEOUT_S", p), None => std::env::remove_var("YM_LLM_TIMEOUT_S") }
    }
    #[test] fn unset_is_the_previous_constant() { with_env(None, || assert_eq!(call_timeout(), std::time::Duration::from_secs(300))); }
    #[test] fn a_valid_value_is_honoured() { with_env(Some("900"), || assert_eq!(call_timeout(), std::time::Duration::from_secs(900))); }
    #[test] fn every_malformed_value_falls_back_never_shorter() {
        for bad in ["", "   ", "abc", "-5", "0", "12.5", "300s", "99999999", "1e3", "86401"] {
            with_env(Some(bad), || assert_eq!(call_timeout(), std::time::Duration::from_secs(300), "{bad:?}"));
        }
    }
    /// EVERY client must use the knob. E.TIMEOUT1 wired three sites in one client and missed the
    /// client the product actually runs; this counts across BOTH files, cut at their test modules
    /// so the assertion strings below cannot match themselves.
    #[test]
    fn every_timeout_site_in_every_client_uses_the_knob() {
        for (name, whole) in [("llm/api.rs", include_str!("llm/api.rs")), ("provider/generic_openai.rs", include_str!("provider/generic_openai.rs"))] {
            let src = whole.split("#[cfg(test)]").next().unwrap_or(whole);
            let literal = src.matches(".timeout_global(Some(std::time::Duration::from_secs(").count();
            assert_eq!(literal, 0, "{name} still hardcodes a call timeout");
            let wired = src.matches("call_timeout()").count();
            assert!(wired >= 1, "{name} never calls the knob");
        }
    }
}
