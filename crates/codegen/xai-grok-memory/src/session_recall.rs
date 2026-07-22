//! Session-scoped top-K recall for the experimental "infinite context" mode
//! ([`xai_grok_config_types::ContextMode::Recall`]).
//!
//! Where [`crate::search`] retrieves across the whole workspace/global memory
//! tier, this module retrieves only chunks tagged with a given `session_id`
//! (written via [`MemoryIndex::index_session_turn`]). The caller embeds the
//! latest user turn, retrieves the top-K most relevant prior chunks from *this
//! conversation*, and injects them as a `<prior_context>` block.
//!
//! ## v1 scope (M1) — additive retrieval, not yet history replacement
//!
//! In this milestone the `<prior_context>` block is injected *in addition to*
//! the normal conversation history, and threshold auto-compaction stays active
//! as a safety net. The end-state design — injecting recalled context *in place
//! of* replaying older turns so per-turn prompt size stays flat, with
//! compaction suppressed — depends on outgoing-history truncation that is
//! deliberately deferred behind the long-session eval harness. Until then,
//! recall augments rather than replaces history.
//!
//! Retrieval itself is a deliberately naive top-K: FTS + vector candidates,
//! score normalization, an optional MMR diversity pass, and truncation. It does
//! **not** yet include the iterative draft/critique/refine loop (a later
//! milestone). It reuses [`crate::search::SearchResult`] and
//! [`crate::mmr::mmr_rerank`] so the retrieval result shape and diversity logic
//! match the cross-session path exactly.
//!
//! ## Session filtering of the vector pass
//!
//! `chunks_vec` is keyed by `chunk_id` only — it has no `session_id` column —
//! so a KNN query returns nearest neighbors across *all* sessions. We over-fetch
//! `top_k × vector_overfetch` candidates and drop any whose owning chunk is not
//! in the target session. The FTS pass is already session-scoped in SQL via
//! [`MemoryIndex::search_fts_by_session`].

use std::collections::HashMap;

use super::embedding::EmbeddingProvider;
use super::index::MemoryIndex;
use super::search::SearchResult;
use xai_grok_config_types::RecallConfig;

/// Theoretical maximum L2 distance between two unit-norm vectors (see the
/// identical constant and rationale in [`crate::search`]).
const MAX_L2_DISTANCE: f64 = 2.0;

/// Retrieve the top-K most relevant chunks from the current session's history.
///
/// Returns results ranked most-relevant first, already truncated to
/// `config.top_k`. Empty when the session has no matching indexed chunks.
///
/// Structured so `&MemoryIndex` is never held across the `.await`, keeping the
/// caller's future `Send` even though `MemoryIndex` is `!Sync`.
#[tracing::instrument(name = "memory.session_recall", skip_all, fields(
    session_id = %session_id,
    top_k = config.top_k,
))]
pub async fn session_recall(
    index: &MemoryIndex,
    embedding_provider: Option<&dyn EmbeddingProvider>,
    query: &str,
    session_id: &str,
    config: &RecallConfig,
) -> Result<Vec<SearchResult>, Box<dyn std::error::Error>> {
    let candidate_limit = config.top_k.max(1) * 3;

    // Phase 1 (sync): session-scoped FTS candidates.
    let fts_results = index
        .search_fts_by_session(query, candidate_limit, session_id)
        .unwrap_or_default();
    let vec_available = index.vec_available();

    // Phase 2 (async): embed the query — no &index borrow held across await.
    let query_embedding = if vec_available {
        if let Some(provider) = embedding_provider {
            match provider.embed_batch(&[query]).await {
                Ok(embeddings) => embeddings.into_iter().next(),
                Err(e) => {
                    tracing::warn!(error = %e, "session recall embed failed, FTS-only");
                    None
                }
            }
        } else {
            None
        }
    } else {
        None
    };

    // Phase 3 (sync): scoped vector search + scoring + merge.
    session_recall_merge(
        index,
        fts_results,
        query_embedding.as_deref(),
        session_id,
        config,
    )
}

/// Synchronous merge: session-filtered vector search, score normalization,
/// weighted combination, MMR, and truncation.
pub(super) fn session_recall_merge(
    index: &MemoryIndex,
    fts_results: Vec<super::index::FtsResult>,
    query_embedding: Option<&[f32]>,
    session_id: &str,
    config: &RecallConfig,
) -> Result<Vec<SearchResult>, Box<dyn std::error::Error>> {
    let top_k = config.top_k.max(1);
    let overfetch = config.vector_overfetch.max(1);

    // Over-fetch KNN candidates; the session filter is applied in the build
    // loop below (chunks_vec is not session-keyed — see the module docs).
    let vec_results = if let Some(embedding) = query_embedding {
        index
            .vector_search(embedding, top_k * overfetch)
            .unwrap_or_default()
    } else {
        vec![]
    };

    // Normalize FTS BM25 ranks to [0,1] (more-negative rank = better match).
    let mut fts_scores: HashMap<String, f64> = HashMap::new();
    if !fts_results.is_empty() {
        let min_rank = fts_results
            .iter()
            .map(|r| r.rank)
            .fold(f64::INFINITY, f64::min);
        let max_rank = fts_results
            .iter()
            .map(|r| r.rank)
            .fold(f64::NEG_INFINITY, f64::max);
        let range = (max_rank - min_rank).max(f64::EPSILON);
        for r in &fts_results {
            fts_scores.insert(r.chunk_id.clone(), 1.0 - (r.rank - min_rank) / range);
        }
    }

    // Normalize vector L2 distances to [0,1] similarity on an absolute scale.
    let mut vec_scores: HashMap<String, f64> = HashMap::new();
    for (chunk_id, distance) in &vec_results {
        let similarity = (1.0 - (*distance as f64 / MAX_L2_DISTANCE)).clamp(0.0, 1.0);
        vec_scores.insert(chunk_id.clone(), similarity);
    }

    let text_weight = config.text_weight as f64;
    let vector_weight = config.vector_weight as f64;

    let all_chunk_ids: std::collections::HashSet<&String> =
        fts_scores.keys().chain(vec_scores.keys()).collect();

    let mut ranked: Vec<(f64, SearchResult)> = Vec::new();
    // Diagnostics for the global-KNN over-fetch (see module docs): in a busy
    // index the nearest neighbors can be almost entirely out-of-session, and an
    // all-dropped vector pass silently degrades recall to FTS-only. Counting
    // kept vs. dropped makes under-retrieval visible in traces.
    let mut vec_candidates_dropped = 0usize;
    let mut vec_candidates_kept = 0usize;

    for chunk_id in all_chunk_ids {
        // Resolve the chunk and enforce the session filter here: FTS candidates
        // are already session-scoped, but over-fetched vector candidates may
        // belong to other sessions and must be dropped.
        let Some(chunk) = index.get_chunk(chunk_id).ok().flatten() else {
            continue;
        };
        let in_session = chunk.session_id.as_deref() == Some(session_id);
        if vec_scores.contains_key(chunk_id) {
            if in_session {
                vec_candidates_kept += 1;
            } else {
                vec_candidates_dropped += 1;
            }
        }
        if !in_session {
            continue;
        }

        let fts = fts_scores.get(chunk_id).copied().unwrap_or(0.0);
        let vec = vec_scores.get(chunk_id).copied().unwrap_or(0.0);

        // Same merge rule as the cross-session path: an FTS-only chunk keeps its
        // full FTS score rather than being penalized to `text_weight`.
        let score = if fts > 0.0 && vec > 0.0 {
            (text_weight * fts + vector_weight * vec).max(fts)
        } else if fts > 0.0 {
            fts
        } else {
            vector_weight * vec
        };

        if score >= config.min_score as f64 {
            ranked.push((
                score,
                SearchResult {
                    chunk_id: chunk.id.clone(),
                    path: chunk.path.clone(),
                    start_line: chunk.start_line,
                    end_line: chunk.end_line,
                    score,
                    snippet: chunk.text.clone(),
                    source: chunk.source.clone(),
                    created_at: chunk.created_at,
                },
            ));
        }
    }

    if !vec_results.is_empty() {
        tracing::debug!(
            session_id = %session_id,
            vec_candidates = vec_results.len(),
            vec_kept = vec_candidates_kept,
            vec_dropped = vec_candidates_dropped,
            "session_recall: global-KNN over-fetch session filter"
        );
    }

    ranked.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal));

    let mmr_enabled = config.mmr.enabled;
    let mut relevance: Vec<f64> = if mmr_enabled {
        Vec::with_capacity(ranked.len())
    } else {
        Vec::new()
    };
    let mut results: Vec<SearchResult> = Vec::with_capacity(ranked.len());
    for (raw_score, result) in ranked {
        if mmr_enabled {
            relevance.push(raw_score);
        }
        results.push(result);
    }

    super::mmr::mmr_rerank(&mut results, &relevance, &config.mmr);
    results.truncate(top_k);

    Ok(results)
}

/// Assemble a `<prior_context>` block from recalled chunks for injection into
/// the outgoing prompt.
///
/// In M1 this augments the normal history (see the module docs); the end-state
/// design injects it *in place of* replaying older turns once history
/// truncation lands.
///
/// Returns an empty string when `results` is empty, so callers can inject it
/// unconditionally without emitting an empty block. Chunks are listed
/// most-relevant first (the order [`session_recall`] returns).
pub fn assemble_prior_context(results: &[SearchResult]) -> String {
    if results.is_empty() {
        return String::new();
    }

    let mut out = String::from(
        "<prior_context>\n\
         Relevant excerpts retrieved from earlier in this session (most \
         relevant first), surfaced to keep them salient. If something you need \
         is missing, use the memory recall tool to search for it.\n\n",
    );
    for r in results {
        // Trim to avoid stacking blank lines from chunk boundaries.
        out.push_str("- ");
        out.push_str(r.snippet.trim());
        out.push('\n');
    }
    out.push_str("</prior_context>\n");
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::embedding::MockEmbeddingProvider;
    use crate::index::{MemoryIndex, init_sqlite_vec};
    use crate::storage::MemoryStorage;
    use tempfile::TempDir;
    use xai_grok_config_types::MemoryIndexConfig;

    fn test_index(tmp: &TempDir) -> MemoryIndex {
        init_sqlite_vec();
        let global = tmp.path().join("memory");
        let workspace = global.join("test_ws");
        let storage = MemoryStorage::with_paths(global, workspace);
        let db_path = tmp.path().join("test.sqlite");
        MemoryIndex::open_or_create(&db_path, storage, MemoryIndexConfig::default(), 4).unwrap()
    }

    fn recall_config() -> RecallConfig {
        RecallConfig {
            min_score: 0.0, // accept all by score; isolate the session filter
            ..Default::default()
        }
    }

    #[tokio::test]
    async fn recalls_only_current_session_chunks() {
        let tmp = TempDir::new().unwrap();
        let mut idx = test_index(&tmp);

        idx.index_session_turn(
            "sess-A",
            "session://sess-A/turn/1",
            "# Turn\n\nWe decided the database migration runs before deploy.",
        )
        .unwrap();
        idx.index_session_turn(
            "sess-B",
            "session://sess-B/turn/1",
            "# Turn\n\nWe decided the database migration runs before deploy.",
        )
        .unwrap();

        let cfg = recall_config();
        let results = session_recall(&idx, None, "database migration deploy", "sess-A", &cfg)
            .await
            .unwrap();

        assert!(!results.is_empty(), "should recall session A's chunk");
        assert!(
            results
                .iter()
                .all(|r| r.path.starts_with("session://sess-A/")),
            "recall must not leak chunks from other sessions: {:?}",
            results.iter().map(|r| &r.path).collect::<Vec<_>>(),
        );
    }

    #[tokio::test]
    async fn does_not_recall_workspace_tier_chunks() {
        let tmp = TempDir::new().unwrap();
        let mut idx = test_index(&tmp);

        // A workspace-tier file chunk (session_id = NULL) that matches the query.
        let f = tmp.path().join("ws.md");
        std::fs::write(&f, "# WS\n\nDatabase migration guidance for the workspace.").unwrap();
        idx.reindex_file(&f, "workspace").unwrap();

        // A session chunk that also matches.
        idx.index_session_turn(
            "sess-A",
            "session://sess-A/turn/1",
            "# Turn\n\nDatabase migration decision for this session.",
        )
        .unwrap();

        let cfg = recall_config();
        let results = session_recall(&idx, None, "database migration", "sess-A", &cfg)
            .await
            .unwrap();

        assert!(!results.is_empty());
        assert!(
            results.iter().all(|r| r.source == "session"),
            "session recall must exclude workspace/global (NULL session_id) chunks",
        );
    }

    #[tokio::test]
    async fn empty_when_session_has_no_chunks() {
        let tmp = TempDir::new().unwrap();
        let mut idx = test_index(&tmp);
        idx.index_session_turn("sess-A", "session://sess-A/turn/1", "# T\n\nHello world.")
            .unwrap();

        let cfg = recall_config();
        let results = session_recall(&idx, None, "hello", "sess-EMPTY", &cfg)
            .await
            .unwrap();
        assert!(results.is_empty(), "unknown session must recall nothing");
    }

    #[tokio::test]
    async fn respects_top_k() {
        let tmp = TempDir::new().unwrap();
        let mut idx = test_index(&tmp);
        for i in 0..10 {
            idx.index_session_turn(
                "sess-A",
                &format!("session://sess-A/turn/{i}"),
                &format!("# Turn {i}\n\nRust ownership note number {i}."),
            )
            .unwrap();
        }

        let cfg = RecallConfig {
            top_k: 3,
            min_score: 0.0,
            ..Default::default()
        };
        let results = session_recall(&idx, None, "rust ownership note", "sess-A", &cfg)
            .await
            .unwrap();
        assert!(results.len() <= 3, "must respect top_k, got {}", results.len());
    }

    #[tokio::test]
    async fn hybrid_path_uses_vector_and_fts() {
        let tmp = TempDir::new().unwrap();
        let mut idx = test_index(&tmp);
        let mock = MockEmbeddingProvider { dimensions: 4 };

        idx.index_session_turn(
            "sess-A",
            "session://sess-A/turn/1",
            "# Turn\n\nRust programming language ownership tutorial.",
        )
        .unwrap();

        // Embed the session chunk so the vector pass has something to match.
        crate::embed_missing_chunks(&idx, &mock).await;

        let cfg = recall_config();
        let results = session_recall(
            &idx,
            Some(&mock as &dyn EmbeddingProvider),
            "rust ownership",
            "sess-A",
            &cfg,
        )
        .await
        .unwrap();

        assert!(!results.is_empty(), "hybrid recall should find the chunk");
        assert!(results[0].score > 0.0);
    }

    #[test]
    fn assemble_prior_context_empty_is_blank() {
        assert_eq!(assemble_prior_context(&[]), "");
    }

    #[test]
    fn assemble_prior_context_wraps_snippets() {
        let results = vec![SearchResult {
            chunk_id: "session://s/turn/1:0".to_string(),
            path: "session://s/turn/1".to_string(),
            start_line: 0,
            end_line: 1,
            score: 0.9,
            snippet: "  Migration runs before deploy.  ".to_string(),
            source: "session".to_string(),
            created_at: 1,
        }];
        let block = assemble_prior_context(&results);
        assert!(block.starts_with("<prior_context>"));
        assert!(block.trim_end().ends_with("</prior_context>"));
        assert!(block.contains("- Migration runs before deploy."));
        // Snippet is trimmed, not raw.
        assert!(!block.contains("  Migration"));
    }

    #[tokio::test]
    async fn reindexing_same_turn_is_idempotent() {
        let tmp = TempDir::new().unwrap();
        let mut idx = test_index(&tmp);

        let first = idx
            .index_session_turn("sess-A", "session://sess-A/turn/1", "# T\n\nAlpha content here.")
            .unwrap();
        assert_eq!(first.added, 1);

        // Same content again: no churn.
        let again = idx
            .index_session_turn("sess-A", "session://sess-A/turn/1", "# T\n\nAlpha content here.")
            .unwrap();
        assert_eq!(again.added, 0);
        assert_eq!(again.updated, 0);

        // Changed content: the chunk updates in place.
        let changed = idx
            .index_session_turn("sess-A", "session://sess-A/turn/1", "# T\n\nBeta content now.")
            .unwrap();
        assert_eq!(changed.updated, 1);
    }
}
