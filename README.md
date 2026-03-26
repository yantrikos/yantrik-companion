# Yantrik Companion

The AI agent that powers [Yantrik OS](https://github.com/yantrikos/yantrik-os). A proactive, personality-driven companion with 100+ tools, 50+ instincts, and a multi-stage cognitive pipeline — all in Rust.

## Features

- **Proactive intelligence** — 4-stage pipeline (Detect → Generate → Score → Deliver) decides when to speak, what to say, and how to say it
- **50+ instincts** — Email watch, open loops guardian, routine learning, health pulse, morning briefing, cooking companion, and more
- **100+ tools** — Files, git, browser automation, email, calendar, system commands, memory search, SSH, Docker, and more
- **Bond system** — Relationship evolves from Stranger → Acquaintance → Companion → Confidant → Partner based on interaction quality
- **Cortex** — Pattern recognition, situation awareness, playbooks, entity tracking, and reasoning
- **Silence policy** — Avoids notification fatigue with cooldowns, importance thresholds, and delivery timing
- **Chat providers** — Telegram, native protocol for multi-platform access
- **MCP client** — Connects to external MCP servers for extended capabilities
- **YAML plugins** — Extend tools without writing Rust

## Workspace Crates

```
yantrik-companion/
├── crates/
│   ├── yantrik-companion-core/      # Config, types, bond, permissions, sanitization
│   ├── yantrik-companion-instincts/ # 50+ proactive instincts
│   ├── yantrik-companion-cortex/    # Pattern recognition, playbooks, reasoning
│   ├── yantrik-companion-tools/     # 100+ tool implementations
│   ├── yantrik-companion/           # Agent loop, proactive pipeline, orchestration
│   ├── yantrik-chat/                # Multi-provider chat (Telegram, native)
│   ├── yantrik-mcp/                 # MCP client integration
│   ├── yantrik/                     # CLI binary
│   ├── yantrik-os/                  # System observer (D-Bus, inotify, sysinfo)
│   ├── yantrikdb-core/              # Memory database (SQLite + vector search)
│   └── yantrikdb-server/            # Database server mode
```

## Instincts (Selection)

| Instinct | Description |
|----------|-------------|
| Email Watch | Monitor inbox, surface important messages |
| Open Loops Guardian | Track uncommitted promises and follow-ups |
| Morning Brief | Daily summary of calendar, weather, priorities |
| Routine Learning | Detect and reinforce daily patterns |
| Health Pulse | Monitor system resource usage and alert |
| Cooking Companion | Step-by-step recipe guidance |
| Memory Weaver | Connect related memories across conversations |
| Curiosity | Surface interesting facts related to current context |
| Follow-Up | Remind about conversations that need continuation |

## Usage

This crate is used as a dependency by [yantrik-os](https://github.com/yantrikos/yantrik-os). It also ships a standalone CLI binary:

```bash
# Ask the companion a question
yantrik ask --config /path/to/config.yaml "What's using the most disk space?"

# With JSON output
yantrik ask --json "Summarize my recent emails"
```

## Dependencies

- [yantrik-ml](https://github.com/yantrikos/yantrik-ml) — AI inference backends
- [yantrikdb](https://github.com/yantrikos/yantrikdb) — Cognitive memory database

## License

AGPL-3.0. See [LICENSE](LICENSE) for details.
