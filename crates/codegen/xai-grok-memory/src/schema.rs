//! SQL schema constants for the memory index.
//!
//! The index uses three tables:
//! - `meta` — key-value metadata (embedding dimensions, schema version)
//! - `chunks` — indexed text chunks with blake3 content hashes
//! - `chunks_fts` — contentless FTS5 virtual table for BM25 keyword search
//!
//! When sqlite-vec is available, a fourth table is created:
//! - `chunks_vec` — vec0 virtual table for KNN vector search

/// Schema version, for documentation and future use.
///
/// Note: this constant is currently informational only — nothing reads or
/// writes a `schema_version` meta key, and migrations are driven by runtime
/// introspection (`pragma_table_info`) rather than a stored version. Treat it
/// as a changelog marker, not a migration gate, until a versioned migration
/// path exists.
///
/// v2 added the nullable `chunks.session_id` column (+ `idx_chunks_session`)
/// for session-scoped recall. It is applied additively via `ALTER TABLE` on
/// pre-existing v1 databases (see [`ADD_SESSION_ID_COLUMN_SQL`]), so it does
/// not force a drop/recreate.
pub const SCHEMA_VERSION: u32 = 2;

/// Generate the SQL schema for the memory index.
///
/// `dimensions` controls the embedding vector size for `chunks_vec`.
/// If `vec_available` is false, the `chunks_vec` table is not created.
///
/// Connection pragmas (busy_timeout, journal_mode) are applied on the open
/// path (`xai_sqlite_journal::JournalMode::open`) — the journal mode depends
/// on the database's filesystem.
pub fn schema_sql(dimensions: usize, vec_available: bool) -> String {
    let mut sql = format!(
        r#"
CREATE TABLE IF NOT EXISTS meta (
    key TEXT PRIMARY KEY,
    value TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS chunks (
    rowid INTEGER PRIMARY KEY AUTOINCREMENT,
    id TEXT UNIQUE NOT NULL,
    path TEXT NOT NULL,
    start_line INTEGER NOT NULL,
    end_line INTEGER NOT NULL,
    text TEXT NOT NULL,
    hash TEXT NOT NULL,
    source TEXT NOT NULL,
    created_at INTEGER NOT NULL,
    updated_at INTEGER NOT NULL,
    access_count INTEGER DEFAULT 0,
    last_accessed INTEGER,
    session_id TEXT
);

CREATE INDEX IF NOT EXISTS idx_chunks_path ON chunks(path);
CREATE INDEX IF NOT EXISTS idx_chunks_hash ON chunks(hash);

CREATE VIRTUAL TABLE IF NOT EXISTS chunks_fts USING fts5(text, content='');

INSERT OR IGNORE INTO meta(key, value) VALUES ('reindex_claim', '');
"#
    );

    if vec_available {
        sql.push_str(&format!(
            "\nCREATE VIRTUAL TABLE IF NOT EXISTS chunks_vec USING vec0(\n    \
             chunk_id TEXT PRIMARY KEY,\n    \
             embedding FLOAT[{dimensions}]\n);\n"
        ));
    }

    sql
}

/// SQL to insert or update an embedding dimension record in the meta table.
pub const UPSERT_META_SQL: &str = "INSERT OR REPLACE INTO meta(key, value) VALUES (?1, ?2)";

/// SQL to query a meta value by key.
pub const GET_META_SQL: &str = "SELECT value FROM meta WHERE key = ?1";

/// Additive v1→v2 migration: add the nullable `session_id` column to an
/// existing `chunks` table. Guarded by a `PRAGMA table_info` check on the
/// open path, since SQLite has no `ADD COLUMN IF NOT EXISTS`. Existing rows
/// get `session_id = NULL`, which the recall path treats as workspace-tier
/// (not scoped to any single session).
pub const ADD_SESSION_ID_COLUMN_SQL: &str = "ALTER TABLE chunks ADD COLUMN session_id TEXT";

/// Companion index for [`ADD_SESSION_ID_COLUMN_SQL`] on the migration path.
pub const CREATE_SESSION_INDEX_SQL: &str =
    "CREATE INDEX IF NOT EXISTS idx_chunks_session ON chunks(session_id)";

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_schema_sql_without_vec() {
        let sql = schema_sql(1536, false);
        assert!(sql.contains("CREATE TABLE IF NOT EXISTS chunks"));
        assert!(sql.contains("CREATE VIRTUAL TABLE IF NOT EXISTS chunks_fts"));
        assert!(!sql.contains("chunks_vec"));
        // Connection pragmas live on the open path, not in the schema batch.
        assert!(!sql.contains("PRAGMA"));
    }

    #[test]
    fn test_schema_sql_has_session_column() {
        // Fresh databases get the session_id column inline via CREATE TABLE.
        // The index is created separately (on the open path, after the column
        // is guaranteed) so it also covers migrated pre-v2 databases where
        // CREATE TABLE IF NOT EXISTS is a no-op — hence NOT in the schema batch.
        let sql = schema_sql(1024, true);
        assert!(sql.contains("session_id TEXT"));
        assert!(
            !sql.contains("idx_chunks_session"),
            "session index must be created on the open path, not the schema batch, \
             so it does not run against a pre-v2 table before ALTER adds the column"
        );
        assert!(ADD_SESSION_ID_COLUMN_SQL.contains("session_id"));
        assert!(CREATE_SESSION_INDEX_SQL.contains("idx_chunks_session"));
    }

    #[test]
    fn test_schema_sql_with_vec() {
        let sql = schema_sql(384, true);
        assert!(sql.contains("chunks_vec"));
        assert!(sql.contains("FLOAT[384]"));
    }

    #[test]
    fn test_schema_sql_different_dimensions() {
        let sql = schema_sql(768, true);
        assert!(sql.contains("FLOAT[768]"));
    }
}
