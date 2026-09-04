//! Per-model overrides from one CSV env value: `"qwen3.8=600, gpt-oss=120"`. An entry applies
//! when its key (lowercased, trimmed) is a substring of the model name (lowercased) — the same
//! matching the household/private allowlists use, so `qwen3.8` covers `qwen3.8:27b-q4_K_M`.
//! Malformed entries are ignored, never fatal: a config typo must not take a lane down.

/// The value of the first entry whose key names `model`, or `None`.
pub fn lookup<'a>(csv: &'a str, model: &str) -> Option<&'a str> {
    let m = model.to_ascii_lowercase();
    csv.split(',')
        .filter_map(|e| e.split_once('='))
        .map(|(k, v)| (k.trim().to_ascii_lowercase(), v.trim()))
        .find(|(k, v)| !k.is_empty() && !v.is_empty() && m.contains(k.as_str()))
        .map(|(_, v)| v)
}

#[cfg(test)]
mod tests {
    use super::lookup;

    #[test]
    fn the_first_matching_entry_wins_and_matching_is_a_case_insensitive_substring() {
        let csv = "qwen3.8=600, GPT-OSS = 120 ,qwen=1";
        assert_eq!(lookup(csv, "qwen3.8:27b-q4_K_M"), Some("600"));
        assert_eq!(lookup(csv, "gpt-oss-backup:20b"), Some("120"));
        assert_eq!(lookup(csv, "QWEN2.5:7b"), Some("1"));
        assert_eq!(lookup(csv, "gemma4:e4b"), None);
    }

    #[test]
    fn malformed_entries_are_ignored_not_fatal() {
        assert_eq!(lookup("abc,=5,qwen=,qwen3.8=off", "qwen3.8:27b"), Some("off"));
        assert_eq!(lookup("", "qwen3.8:27b"), None);
        assert_eq!(lookup("   ", "qwen3.8:27b"), None);
    }
}
