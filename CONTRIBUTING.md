# Contributing to Yantrik Companion

## Architecture

Yantrik Companion is a workspace of 10+ crates. The main ones:

| Crate | Purpose |
|-------|---------|
| `yantrik-companion-core` | Config, types, bond system, permissions, sanitization |
| `yantrik-companion-instincts` | 50+ proactive instincts (email watch, routines, etc.) |
| `yantrik-companion-cortex` | Pattern recognition, playbooks, situation awareness |
| `yantrik-companion-tools` | 100+ tool implementations |
| `yantrik-companion` | Agent loop, proactive pipeline, orchestration |
| `yantrik-chat` | Multi-provider chat (Telegram, native protocol) |
| `yantrik-mcp` | MCP client/server integration |
| `yantrik` | CLI binary |
| `yantrik-os` | System observer (D-Bus, inotify, sysinfo) |
| `yantrikdb-core` | Memory database (SQLite + HNSW vector search) |

## Adding a New Tool

1. Create `crates/yantrik-companion-tools/src/mytool.rs`
2. Implement the `Tool` trait:

```rust
struct MyTool;

impl Tool for MyTool {
    fn name(&self) -> &'static str { "my_tool" }
    fn permission(&self) -> PermissionLevel { PermissionLevel::Safe }
    fn category(&self) -> &'static str { "my_category" }
    fn definition(&self) -> serde_json::Value { /* OpenAI function schema */ }
    fn execute(&self, ctx: &ToolContext, args: &serde_json::Value) -> String { /* ... */ }
}
```

3. Add `pub mod mytool;` to `tools/lib.rs`
4. Register in `build_registry()`
5. Run `cargo check -p yantrik-companion-tools`

**Permission levels:** `Safe` (read-only) < `Standard` (reversible) < `Sensitive` (system changes) < `Dangerous` (destructive)

**Tips:**
- Shell out to system CLIs when possible (fewer Rust deps)
- Truncate output to ~3000 chars for LLM-friendly responses
- Use `validate_path()` for file access (blocks `.ssh`, `.gnupg`)
- Use `expand_home()` for `~/` path expansion

## Adding a New Instinct

1. Create `crates/yantrik-companion-instincts/src/my_instinct.rs`
2. Implement the `Instinct` trait:

```rust
pub struct MyInstinct;

impl Instinct for MyInstinct {
    fn name(&self) -> &'static str { "my_instinct" }
    fn description(&self) -> &'static str { "What this instinct does" }
    fn default_enabled(&self) -> bool { true }
    fn cooldown_seconds(&self) -> u64 { 3600 }
    fn evaluate(&self, ctx: &InstinctContext) -> Option<Urge> { /* ... */ }
}
```

3. Add to `mod.rs` and register in the instinct registry
4. Run `cargo check -p yantrik-companion-instincts`

**Instinct guidelines:**
- Instincts should be lightweight — evaluation must be fast
- Use cooldowns to prevent notification fatigue
- Return `None` when there's nothing worth surfacing
- Set appropriate urgency scores (0.0–1.0) so the silence policy can prioritize

## Adding a Chat Provider

1. Create `crates/yantrik-chat/src/providers/myprovider.rs`
2. Implement the `ChatProvider` trait
3. Register in `providers/mod.rs`
4. Add configuration to `ChatConfig`

## Dev Setup

```bash
# Check all crates
cargo check

# Check specific crate
cargo check -p yantrik-companion
cargo check -p yantrik-companion-tools
cargo check -p yantrik-companion-instincts

# Test
cargo test

# Build CLI binary
cargo build -p yantrik
```

## Code Organization Guidelines

- **Keep files small and modular** — one tool per file, one instinct per file
- **No cross-crate circular dependencies** — the dependency graph flows: core → instincts/cortex/tools → companion
- **Use `tracing`** for logging, not `println!`
- **Sanitize all user inputs** before shell execution — use `sanitize.rs` helpers
- **Memory operations** go through `yantrikdb-core`, never raw SQL

## License

AGPL-3.0. By contributing, you agree your contributions will be licensed under AGPL-3.0.
