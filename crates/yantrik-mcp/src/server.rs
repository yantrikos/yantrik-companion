use std::sync::Mutex;

use rmcp::{
    ErrorData as McpError, ServerHandler,
    handler::server::{
        router::tool::ToolRouter,
        wrapper::Parameters,
    },
    model::*,
    schemars, tool_handler, tool_router,
    tool,
};
use serde_json::json;
use yantrikdb_core::YantrikDB;

/// Input for the `remember` tool.
#[derive(Debug, serde::Deserialize, schemars::JsonSchema)]
pub struct RememberInput {
    /// The text content to store as a memory.
    pub text: String,
    /// Memory type: "fact", "episode", "preference", "skill", etc.
    #[serde(default = "default_memory_type")]
    pub memory_type: String,
    /// Importance score from 0.0 (trivial) to 1.0 (critical).
    #[serde(default = "default_importance")]
    pub importance: f64,
    /// Emotional valence from -1.0 (negative) to 1.0 (positive).
    #[serde(default)]
    pub valence: f64,
    /// Knowledge domain (e.g. "general", "work", "health").
    #[serde(default = "default_domain")]
    pub domain: String,
    /// Source attribution (e.g. "user", "mcp", "system").
    #[serde(default = "default_source")]
    pub source: String,
}

fn default_memory_type() -> String { "fact".into() }
fn default_importance() -> f64 { 0.5 }
fn default_domain() -> String { "general".into() }
fn default_source() -> String { "mcp".into() }

/// Input for the `recall` tool.
#[derive(Debug, serde::Deserialize, schemars::JsonSchema)]
pub struct RecallInput {
    /// Natural language query to search memories.
    pub query: String,
    /// Maximum number of results to return.
    #[serde(default = "default_top_k")]
    pub top_k: usize,
    /// Optional domain filter.
    pub domain: Option<String>,
    /// Optional source filter.
    pub source: Option<String>,
}

fn default_top_k() -> usize { 5 }

/// Input for the `relate` tool.
#[derive(Debug, serde::Deserialize, schemars::JsonSchema)]
pub struct RelateInput {
    /// Source entity name.
    pub source: String,
    /// Target entity name.
    pub target: String,
    /// Relationship type (e.g. "works_at", "likes", "is_a").
    pub relation_type: String,
    /// Relationship weight from 0.0 to 1.0.
    #[serde(default = "default_weight")]
    pub weight: f64,
}

fn default_weight() -> f64 { 1.0 }

/// Input for the `forget` tool.
#[derive(Debug, serde::Deserialize, schemars::JsonSchema)]
pub struct ForgetInput {
    /// Record ID of the memory to tombstone.
    pub rid: String,
}

/// Input for the `beliefs` tool.
#[derive(Debug, serde::Deserialize, schemars::JsonSchema)]
pub struct BeliefsInput {
    /// Minimum confidence threshold (0.0 to 1.0).
    #[serde(default = "default_min_confidence")]
    pub min_confidence: f64,
    /// Optional category filter.
    pub category: Option<String>,
}

fn default_min_confidence() -> f64 { 0.5 }

/// Input for the `conflicts` tool.
#[derive(Debug, serde::Deserialize, schemars::JsonSchema)]
pub struct ConflictsInput {
    /// Conflict status filter: "open", "resolved", "dismissed".
    #[serde(default = "default_status")]
    pub status: String,
    /// Maximum number of conflicts to return.
    #[serde(default = "default_limit")]
    pub limit: usize,
}

fn default_status() -> String { "open".into() }
fn default_limit() -> usize { 20 }

/// Input for the `entities` tool.
#[derive(Debug, serde::Deserialize, schemars::JsonSchema)]
pub struct EntitiesInput {
    /// Entity name to query edges for.
    pub entity: String,
}

/// Input for the `patterns` tool.
#[derive(Debug, serde::Deserialize, schemars::JsonSchema)]
pub struct PatternsInput {
    /// Pattern status filter: "active", "stale", "expired".
    #[serde(default = "default_pattern_status")]
    pub status: String,
    /// Maximum number of patterns to return.
    #[serde(default = "default_pattern_limit")]
    pub limit: usize,
}

fn default_pattern_status() -> String { "active".into() }
fn default_pattern_limit() -> usize { 10 }

/// Input for the `stats` tool.
#[derive(Debug, serde::Deserialize, schemars::JsonSchema)]
pub struct StatsInput {
    /// Optional namespace filter.
    pub namespace: Option<String>,
}

/// The MCP server wrapping YantrikDB.
#[derive(Clone)]
pub struct YantrikMcpServer {
    db: std::sync::Arc<Mutex<YantrikDB>>,
    tool_router: ToolRouter<YantrikMcpServer>,
}

impl YantrikMcpServer {
    pub fn new(db: YantrikDB) -> Self {
        Self {
            db: std::sync::Arc::new(Mutex::new(db)),
            tool_router: Self::tool_router(),
        }
    }
}

#[tool_router]
impl YantrikMcpServer {
    /// Store a new memory with automatic semantic embedding.
    #[tool(description = "Store a new memory in YantrikDB with automatic semantic embedding. Returns the record ID.")]
    fn remember(
        &self,
        Parameters(input): Parameters<RememberInput>,
    ) -> Result<CallToolResult, McpError> {
        let db = self.db.lock().map_err(|e| {
            McpError::internal_error(format!("lock error: {e}"), None)
        })?;
        let rid = db.record_text(
            &input.text,
            &input.memory_type,
            input.importance,
            input.valence,
            168.0, // 7-day half-life default
            &json!({}),
            "default",
            0.9,
            &input.domain,
            &input.source,
            None,
        ).map_err(|e| McpError::internal_error(format!("record failed: {e}"), None))?;

        Ok(CallToolResult::success(vec![Content::text(
            json!({ "rid": rid, "status": "stored" }).to_string(),
        )]))
    }

    /// Semantically search memories by natural language query.
    #[tool(description = "Search memories using natural language. Returns ranked results with scores and metadata.")]
    fn recall(
        &self,
        Parameters(input): Parameters<RecallInput>,
    ) -> Result<CallToolResult, McpError> {
        let db = self.db.lock().map_err(|e| {
            McpError::internal_error(format!("lock error: {e}"), None)
        })?;

        let results = if input.domain.is_some() || input.source.is_some() {
            db.recall_text_filtered(
                &input.query,
                input.top_k,
                input.domain.as_deref(),
                input.source.as_deref(),
            )
        } else {
            db.recall_text(&input.query, input.top_k)
        }.map_err(|e| McpError::internal_error(format!("recall failed: {e}"), None))?;

        let items: Vec<serde_json::Value> = results.iter().map(|r| {
            json!({
                "rid": r.rid,
                "text": r.text,
                "score": format!("{:.4}", r.score),
                "importance": r.importance,
                "valence": r.valence,
                "memory_type": r.memory_type,
                "domain": r.domain,
                "source": r.source,
                "created_at": r.created_at,
                "why": r.why_retrieved,
            })
        }).collect();

        Ok(CallToolResult::success(vec![Content::text(
            serde_json::to_string_pretty(&items).unwrap_or_default(),
        )]))
    }

    /// Create a relationship between two entities in the knowledge graph.
    #[tool(description = "Create a typed relationship between two entities (e.g. 'Alice' --works_at--> 'Acme Corp'). Returns the edge ID.")]
    fn relate(
        &self,
        Parameters(input): Parameters<RelateInput>,
    ) -> Result<CallToolResult, McpError> {
        let db = self.db.lock().map_err(|e| {
            McpError::internal_error(format!("lock error: {e}"), None)
        })?;
        let edge_id = db.relate(
            &input.source,
            &input.target,
            &input.relation_type,
            input.weight,
        ).map_err(|e| McpError::internal_error(format!("relate failed: {e}"), None))?;

        Ok(CallToolResult::success(vec![Content::text(
            json!({ "edge_id": edge_id, "status": "created" }).to_string(),
        )]))
    }

    /// Tombstone (soft-delete) a memory by its record ID.
    #[tool(description = "Forget a memory by tombstoning it. The record is preserved but excluded from search results.")]
    fn forget(
        &self,
        Parameters(input): Parameters<ForgetInput>,
    ) -> Result<CallToolResult, McpError> {
        let db = self.db.lock().map_err(|e| {
            McpError::internal_error(format!("lock error: {e}"), None)
        })?;
        let success = db.forget(&input.rid)
            .map_err(|e| McpError::internal_error(format!("forget failed: {e}"), None))?;

        Ok(CallToolResult::success(vec![Content::text(
            json!({ "rid": input.rid, "forgotten": success }).to_string(),
        )]))
    }

    /// List autonomous beliefs the system has formed from observations.
    #[tool(description = "List autonomous beliefs formed from memory patterns. Beliefs have confidence scores and lifecycle stages.")]
    fn beliefs(
        &self,
        Parameters(input): Parameters<BeliefsInput>,
    ) -> Result<CallToolResult, McpError> {
        let db = self.db.lock().map_err(|e| {
            McpError::internal_error(format!("lock error: {e}"), None)
        })?;

        let mut beliefs = db.get_beliefs_above(input.min_confidence)
            .map_err(|e| McpError::internal_error(format!("beliefs failed: {e}"), None))?;

        // Filter by category if specified
        if let Some(ref cat) = input.category {
            let cat_lower = cat.to_lowercase();
            beliefs.retain(|b| format!("{:?}", b.category).to_lowercase() == cat_lower);
        }

        let items: Vec<serde_json::Value> = beliefs.iter().map(|b| {
            json!({
                "id": b.id.to_string(),
                "description": b.description,
                "category": format!("{:?}", b.category),
                "confidence": format!("{:.3}", b.confidence),
                "stage": format!("{:?}", b.stage),
                "confirming": b.confirming_observations,
                "contradicting": b.contradicting_observations,
            })
        }).collect();

        Ok(CallToolResult::success(vec![Content::text(
            serde_json::to_string_pretty(&items).unwrap_or_default(),
        )]))
    }

    /// List detected contradictions in the memory store.
    #[tool(description = "List detected contradictions between memories. Conflicts have priority levels and resolution status.")]
    fn conflicts(
        &self,
        Parameters(input): Parameters<ConflictsInput>,
    ) -> Result<CallToolResult, McpError> {
        let db = self.db.lock().map_err(|e| {
            McpError::internal_error(format!("lock error: {e}"), None)
        })?;
        let conflicts = db.get_conflicts(
            Some(&input.status), None, None, None, input.limit,
        ).map_err(|e| McpError::internal_error(format!("conflicts failed: {e}"), None))?;

        let items: Vec<serde_json::Value> = conflicts.iter().map(|c| {
            json!({
                "conflict_id": c.conflict_id,
                "type": c.conflict_type,
                "priority": c.priority,
                "status": c.status,
                "memory_a": c.memory_a,
                "memory_b": c.memory_b,
                "entity": c.entity,
                "reason": c.detection_reason,
                "detected_at": c.detected_at,
            })
        }).collect();

        Ok(CallToolResult::success(vec![Content::text(
            serde_json::to_string_pretty(&items).unwrap_or_default(),
        )]))
    }

    /// Run a maintenance cycle: consolidation, conflict detection, pattern mining.
    #[tool(description = "Run a cognitive maintenance cycle: memory consolidation, conflict detection, and pattern mining.")]
    fn consolidate(&self) -> Result<CallToolResult, McpError> {
        let db = self.db.lock().map_err(|e| {
            McpError::internal_error(format!("lock error: {e}"), None)
        })?;
        let config = yantrikdb_core::types::ThinkConfig::default();
        let result = db.think(&config)
            .map_err(|e| McpError::internal_error(format!("think failed: {e}"), None))?;

        Ok(CallToolResult::success(vec![Content::text(
            json!({
                "triggers": result.triggers.len(),
                "consolidations": result.consolidation_count,
                "conflicts_found": result.conflicts_found,
                "patterns_new": result.patterns_new,
                "patterns_updated": result.patterns_updated,
                "duration_ms": result.duration_ms,
            }).to_string(),
        )]))
    }

    /// Get edges/relationships for an entity in the knowledge graph.
    #[tool(description = "Query the knowledge graph for all relationships connected to a given entity.")]
    fn entities(
        &self,
        Parameters(input): Parameters<EntitiesInput>,
    ) -> Result<CallToolResult, McpError> {
        let db = self.db.lock().map_err(|e| {
            McpError::internal_error(format!("lock error: {e}"), None)
        })?;
        let edges = db.get_edges(&input.entity)
            .map_err(|e| McpError::internal_error(format!("get_edges failed: {e}"), None))?;

        let items: Vec<serde_json::Value> = edges.iter().map(|e| {
            json!({
                "edge_id": e.edge_id,
                "source": e.src,
                "target": e.dst,
                "relation_type": e.rel_type,
                "weight": e.weight,
            })
        }).collect();

        Ok(CallToolResult::success(vec![Content::text(
            serde_json::to_string_pretty(&items).unwrap_or_default(),
        )]))
    }

    /// Get detected behavioral/temporal patterns.
    #[tool(description = "List detected behavioral and temporal patterns mined from memory data.")]
    fn patterns(
        &self,
        Parameters(input): Parameters<PatternsInput>,
    ) -> Result<CallToolResult, McpError> {
        let db = self.db.lock().map_err(|e| {
            McpError::internal_error(format!("lock error: {e}"), None)
        })?;
        let patterns = db.get_patterns(None, Some(&input.status), input.limit)
            .map_err(|e| McpError::internal_error(format!("patterns failed: {e}"), None))?;

        let items: Vec<serde_json::Value> = patterns.iter().map(|p| {
            json!({
                "pattern_id": p.pattern_id,
                "type": p.pattern_type,
                "description": p.description,
                "confidence": format!("{:.3}", p.confidence),
                "occurrences": p.occurrence_count,
                "entities": p.entity_names,
            })
        }).collect();

        Ok(CallToolResult::success(vec![Content::text(
            serde_json::to_string_pretty(&items).unwrap_or_default(),
        )]))
    }

    /// Get database statistics.
    #[tool(description = "Get YantrikDB statistics: memory counts, entity counts, conflict counts, and index sizes.")]
    fn stats(
        &self,
        Parameters(input): Parameters<StatsInput>,
    ) -> Result<CallToolResult, McpError> {
        let db = self.db.lock().map_err(|e| {
            McpError::internal_error(format!("lock error: {e}"), None)
        })?;
        let s = db.stats(input.namespace.as_deref())
            .map_err(|e| McpError::internal_error(format!("stats failed: {e}"), None))?;

        Ok(CallToolResult::success(vec![Content::text(
            json!({
                "active_memories": s.active_memories,
                "consolidated_memories": s.consolidated_memories,
                "tombstoned_memories": s.tombstoned_memories,
                "archived_memories": s.archived_memories,
                "edges": s.edges,
                "entities": s.entities,
                "open_conflicts": s.open_conflicts,
                "resolved_conflicts": s.resolved_conflicts,
                "pending_triggers": s.pending_triggers,
                "active_patterns": s.active_patterns,
                "scoring_cache_entries": s.scoring_cache_entries,
                "vec_index_entries": s.vec_index_entries,
            }).to_string(),
        )]))
    }
}

#[tool_handler]
impl ServerHandler for YantrikMcpServer {
    fn get_info(&self) -> ServerInfo {
        ServerInfo::new(
            ServerCapabilities::builder()
                .enable_tools()
                .build(),
        )
        .with_server_info(Implementation::new("yantrik-mcp", env!("CARGO_PKG_VERSION")))
        .with_protocol_version(ProtocolVersion::V_2024_11_05)
        .with_instructions(
            "YantrikDB is your persistent cognitive memory. Use it AUTOMATICALLY — never wait \
             for the user to ask you to remember or recall.\n\n\
             ## Auto-recall (before responding)\n\
             - At conversation start, call `recall` with a summary of the user's first message \
             to load relevant context.\n\
             - When the user references past work, decisions, people, or preferences, call \
             `recall` to retrieve what you know.\n\
             - When context seems missing or you're unsure about preferences, check memory.\n\n\
             ## Auto-remember (during conversation)\n\
             Proactively call `remember` whenever you encounter:\n\
             - Decisions made: architecture choices, tool selections, design decisions, and WHY.\n\
             - User preferences: coding style, communication style, workflow habits, likes/dislikes.\n\
             - People & relationships: names, roles, who works on what, team structure.\n\
             - Project context: goals, deadlines, blockers, current focus, infrastructure.\n\
             - Corrections: when the user corrects you, remember it to avoid repeating mistakes.\n\
             - Important facts: anything stated as important or useful in future sessions.\n\n\
             ## Auto-relate (knowledge graph)\n\
             Call `relate` when you learn about entity relationships:\n\
             - Person works_at Company, Project depends_on Technology, User prefers Tool, etc.\n\n\
             ## What NOT to remember\n\
             - Ephemeral task details (files being edited now, current debug state).\n\
             - Things derivable from code or git history.\n\
             - Verbatim code — just remember the decision, not the implementation.\n\n\
             ## Memory quality\n\
             - Use specific, searchable text — avoid vague descriptions.\n\
             - Set importance: 0.8-1.0 critical, 0.5-0.7 useful context, 0.3-0.5 minor.\n\
             - Set domain: \"work\", \"preference\", \"architecture\", \"people\", \"infrastructure\".\n\
             - Call `consolidate` during long sessions to keep memory healthy."
                .to_string(),
        )
    }
}
