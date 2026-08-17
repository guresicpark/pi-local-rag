/**
 * Centralizes every raw SQL statement used across the pipeline: schema
 * creation, embedding-model migrations, chunk/vector/FTS/file/metadata
 * access. Nothing outside this file should contain a `.prepare` / `.exec`
 * call against the RAG database — that keeps the schema-to-code contract in
 * one place instead of scattered across several files with subtly
 * duplicated INSERT shapes.
 *
 * Runs on the built-in `node:sqlite` driver (DatabaseSync). Because that
 * driver has no statement cache, `preparedStatement()` below adds a
 * per-connection cache so hot loops (per-chunk inserts during indexing,
 * per-file lookups during rebuilds) do not re-parse SQL on every call.
 */
import type { DatabaseSync, StatementSync } from "node:sqlite";
import {
  EMBEDDING_MODEL,
  VECTOR_DIM,
  CODE_EMBEDDING_MODEL,
  CODE_VECTOR_DIM,
  CODE_EMBED_SCHEME,
  DEFAULT_CODE_EXTS,
} from "./constants.ts";

// ─── Prepared-statement cache ─────────────────────────────────────────────

/**
 * Per-connection cache of prepared statements, keyed by SQL text. Statements
 * are stable after initSchema() (schema shape never changes mid-connection),
 * so caching is safe and removes SQL parsing from every hot-loop iteration.
 */
const statementCaches = new WeakMap<DatabaseSync, Map<string, StatementSync>>();

/** Get (or lazily create) the cached prepared statement for `sql` on `db`. */
function preparedStatement(db: DatabaseSync, sql: string): StatementSync {
  let connectionCache = statementCaches.get(db);
  if (!connectionCache) {
    connectionCache = new Map();
    statementCaches.set(db, connectionCache);
  }
  let statement = connectionCache.get(sql);
  if (!statement) {
    statement = db.prepare(sql);
    connectionCache.set(sql, statement);
  }
  return statement;
}

// ─── Transactions ─────────────────────────────────────────────────────────

/**
 * Run `operation` inside a single SQLite transaction. The transaction is
 * rolled back and the error rethrown if `operation` throws, so callers never
 * observe partial writes.
 */
export function runInTransaction<T>(db: DatabaseSync, operation: () => T): T {
  db.exec("BEGIN");
  try {
    const result = operation();
    db.exec("COMMIT");
    return result;
  } catch (error) {
    db.exec("ROLLBACK");
    throw error;
  }
}

// ─── Schema ───────────────────────────────────────────────────────────────

/**
 * Create every table if missing and run the embedding-compatibility
 * migrations. Must be called once per connection before any other call.
 */
export function initSchema(db: DatabaseSync) {
  // Legacy stores from an early schema revision carried sync triggers that
  // are no longer used — always drop them so schema drift can't accumulate.
  db.exec("DROP TRIGGER IF EXISTS chunks_ai; DROP TRIGGER IF EXISTS chunks_ad;");
  db.exec(`
    CREATE TABLE IF NOT EXISTS metadata (
      key   TEXT PRIMARY KEY,
      value TEXT NOT NULL
    );

    CREATE TABLE IF NOT EXISTS chunks (
      id            TEXT PRIMARY KEY,
      file_path     TEXT NOT NULL,
      chunk_content TEXT NOT NULL,
      line_start    INTEGER NOT NULL,
      line_end      INTEGER NOT NULL,
      chunk_hash    TEXT NOT NULL,
      indexed_at    TEXT NOT NULL,
      tokens        INTEGER NOT NULL
    );

    CREATE VIRTUAL TABLE IF NOT EXISTS chunks_fts USING fts5(
      chunk_content,
      file_path,
      content_rowid=rowid
    );

    CREATE TRIGGER IF NOT EXISTS chunks_ai AFTER INSERT ON chunks BEGIN
      INSERT INTO chunks_fts(rowid, chunk_content, file_path)
      VALUES (new.rowid, new.chunk_content, new.file_path);
    END;

    CREATE TRIGGER IF NOT EXISTS chunks_ad AFTER DELETE ON chunks BEGIN
      DELETE FROM chunks_fts WHERE rowid = old.rowid;
    END;

    -- Text-model vectors (nomic). sqlite-vec virtual table keyed by the
    -- chunks table's rowid. NOTE: the rowid parameter must go through
    -- CAST(? AS INTEGER) — node:sqlite binds JS numbers as REAL and vec0
    -- rejects non-integer primary keys.
    CREATE VIRTUAL TABLE IF NOT EXISTS chunks_vec USING vec0(
      embedding float[${VECTOR_DIM}]
    );

    -- Code-group vectors (jina-embeddings-v2-base-code). A separate space
    -- from chunks_vec even at equal dimension — vectors from different
    -- models are not comparable, so each model gets its own table.
    CREATE VIRTUAL TABLE IF NOT EXISTS chunks_vec_code USING vec0(
      embedding float[${CODE_VECTOR_DIM}]
    );

    CREATE TABLE IF NOT EXISTS files (
      path      TEXT PRIMARY KEY,
      hash      TEXT NOT NULL,
      chunks    INTEGER NOT NULL,
      indexed   TEXT NOT NULL,
      size      INTEGER NOT NULL,
      embedded  INTEGER NOT NULL DEFAULT 0
    );

    -- Re-indexing deletes chunks per file (DELETE … WHERE file_path = ?);
    -- without this index each delete full-scans the chunks table.
    CREATE INDEX IF NOT EXISTS idx_chunks_file_path ON chunks(file_path);
  `);

  // Embedding-model swaps: vectors produced by a different model are
  // incompatible (different spaces even at equal dimension), and vec0
  // tables keep their original float[N] definition from when the store was
  // created.
  //
  // - text model changed → drop chunks_vec + wipe all content (legacy
  //   behavior; trackedPaths live in config.json and survive).
  // - code model added/changed → drop chunks_vec_code and delete ONLY
  //   code-classified files' chunks; prose chunks keep their nomic vectors,
  //   so upgrading to the dual-model scheme doesn't re-embed the docs.
  // `/rag rebuild` restores whatever was cleared.
  const storedTextModel = getMetadata(db, MetadataKey.EmbeddingModel);
  if (storedTextModel && storedTextModel !== EMBEDDING_MODEL) {
    db.exec("DROP TABLE IF EXISTS chunks_vec; DELETE FROM chunks_vec_code; DELETE FROM chunks; DELETE FROM files;");
    db.exec("INSERT INTO chunks_fts(chunks_fts) VALUES('rebuild')");
    db.exec(`CREATE VIRTUAL TABLE IF NOT EXISTS chunks_vec USING vec0(
      embedding float[${VECTOR_DIM}]
    );`);
    deleteMetadata(db, MetadataKey.EmbeddingModel);
    process.stderr.write(
      `[rag] text embedding model changed (${storedTextModel} → ${EMBEDDING_MODEL}); index cleared — run /rag rebuild to re-index\n`,
    );
  }

  // Classification here uses the DEFAULT code extensions: stores created
  // before the dual-model split only ever contained default-ext files, so
  // user-configured extras can't appear in them. The wipe only fires on a
  // genuine pre-dual-model index — text-model metadata present (every
  // indexFiles run records it) but code-model metadata absent/changed.
  const storedCodeModel = getMetadata(db, MetadataKey.EmbeddingCodeModel);
  if (storedCodeModel !== CODE_EMBEDDING_MODEL) {
    if (storedTextModel && countChunksTotal(db) > 0) {
      resetCodeEmbeddingSpace(db);
      process.stderr.write(
        `[rag] code embedding model ${storedCodeModel ? `changed (${storedCodeModel} → ${CODE_EMBEDDING_MODEL})` : `set (${CODE_EMBEDDING_MODEL})`}; ` +
        `code files cleared — run /rag rebuild to re-embed them with the code model\n`,
      );
    }
    db.exec(`CREATE VIRTUAL TABLE IF NOT EXISTS chunks_vec_code USING vec0(
      embedding float[${CODE_VECTOR_DIM}]
    );`);
    setMetadata(db, MetadataKey.EmbeddingCodeModel, CODE_EMBEDDING_MODEL);
  }

  // Code-document embedding *scheme* change (how code chunks are prepared
  // for the model, e.g. the filename-context header) — code vectors are
  // incompatible even though the model id is unchanged, so code-classified
  // files are cleared (prose/nomic vectors survive) and `/rag rebuild`
  // re-embeds them. Mirrors the model-change wipe above.
  const storedCodeScheme = getMetadata(db, MetadataKey.EmbeddingCodeScheme);
  if (storedCodeScheme !== CODE_EMBED_SCHEME) {
    if (storedTextModel && countChunksTotal(db) > 0) {
      resetCodeEmbeddingSpace(db);
      process.stderr.write(
        `[rag] code embedding scheme ${storedCodeScheme ? `changed (${storedCodeScheme} → ${CODE_EMBED_SCHEME})` : `set (${CODE_EMBED_SCHEME})`}; ` +
        `code files cleared — run /rag rebuild to re-embed them with the new scheme\n`,
      );
    }
    setMetadata(db, MetadataKey.EmbeddingCodeScheme, CODE_EMBED_SCHEME);
  }
}

/**
 * Drop + recreate the code vector table and delete every code-classified
 * file's chunks/vectors/file rows (FTS index rebuilt). Prose chunks + nomic
 * vectors are left untouched. Shared by the code-model-change and
 * code-scheme-change migrations.
 */
function resetCodeEmbeddingSpace(db: DatabaseSync) {
  const codeChunkPredicate = DEFAULT_CODE_EXTS.map(ext => `file_path LIKE '%${ext}'`).join(" OR ");
  const codeFilePredicate = DEFAULT_CODE_EXTS.map(ext => `path LIKE '%${ext}'`).join(" OR ");
  db.exec("DROP TABLE IF EXISTS chunks_vec_code;");
  db.exec(`
    DELETE FROM chunks_vec WHERE rowid IN (SELECT rowid FROM chunks WHERE ${codeChunkPredicate});
    DELETE FROM chunks WHERE ${codeChunkPredicate};
    DELETE FROM files WHERE ${codeFilePredicate};
  `);
  db.exec("INSERT INTO chunks_fts(chunks_fts) VALUES('rebuild')");
  db.exec(`CREATE VIRTUAL TABLE IF NOT EXISTS chunks_vec_code USING vec0(
    embedding float[${CODE_VECTOR_DIM}]
  );`);
}

// ─── Chunks ───────────────────────────────────────────────────────────────

/** Raw chunk row as stored in the chunks table (SQLite column naming). */
export interface ChunkRow {
  rowid: number;
  id: string;
  file_path: string;
  chunk_content: string;
  line_start: number;
  line_end: number;
  chunk_hash: string;
  indexed_at: string;
  tokens: number;
}

/** Chunk values for insertion (camelCase API naming). */
export interface NewChunk {
  id: string;
  filePath: string;
  content: string;
  lineStart: number;
  lineEnd: number;
  hash: string;
  indexedAt: string;
  tokens: number;
}

/** Fast existence check — LIMIT 1 avoids scanning the table. */
export function hasAnyChunks(db: DatabaseSync): boolean {
  return !!preparedStatement(db, "SELECT 1 FROM chunks LIMIT 1").get();
}

/**
 * Insert one chunk row. Returns the driver result so the caller can read
 * `lastInsertRowid` for the vector-table insert.
 */
export function insertChunk(db: DatabaseSync, chunk: NewChunk) {
  return preparedStatement(db, `
    INSERT INTO chunks(id, file_path, chunk_content, line_start, line_end, chunk_hash, indexed_at, tokens)
    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
  `).run(
    chunk.id, chunk.filePath, chunk.content,
    chunk.lineStart, chunk.lineEnd, chunk.hash,
    chunk.indexedAt, chunk.tokens,
  );
}

/** Delete every chunk belonging to `filePath` (re-index preparation). */
export function deleteChunksForFile(db: DatabaseSync, filePath: string) {
  preparedStatement(db, "DELETE FROM chunks WHERE file_path = ?").run(filePath);
}

/** Load full chunk rows for an explicit set of rowids (search hydration). */
export function getChunksByRowids(db: DatabaseSync, rowids: number[]): ChunkRow[] {
  if (rowids.length === 0) return [];
  const placeholders = rowids.map(() => "?").join(",");
  return preparedStatement(db, `
    SELECT rowid, id, file_path, chunk_content, line_start, line_end,
            chunk_hash, indexed_at, tokens
    FROM chunks
    WHERE rowid IN (${placeholders})
  `).all(...rowids) as unknown as ChunkRow[];
}

/** Legacy-shape chunk (camelCase aliases) used by loadIndex()/IndexMeta. */
export interface LoadedChunk {
  id: string;
  file: string;
  content: string;
  lineStart: number;
  lineEnd: number;
  hash: string;
  indexed: string;
  tokens: number;
}

/** Load every chunk in the store (legacy IndexMeta hydration). */
export function getAllChunks(db: DatabaseSync): LoadedChunk[] {
  return preparedStatement(db, `
    SELECT c.id, c.file_path as file, c.chunk_content as content,
            c.line_start as lineStart, c.line_end as lineEnd,
            c.chunk_hash as hash, c.indexed_at as indexed, c.tokens
    FROM chunks c
  `).all() as unknown as LoadedChunk[];
}

// ─── Vectors (sqlite-vec) ─────────────────────────────────────────────────

/**
 * View a float vector as the Buffer sqlite-vec expects. Float32Array input
 * is a zero-copy Buffer view over the existing store — safe because
 * node:sqlite copies the bytes into its binding at run() time, so the
 * source array stays reusable afterwards.
 */
function float32ToBuffer(vector: number[] | Float32Array): Buffer {
  if (vector instanceof Float32Array) {
    return Buffer.from(vector.buffer, vector.byteOffset, vector.byteLength);
  }
  const float32View = new Float32Array(vector);
  return Buffer.from(float32View.buffer, float32View.byteOffset, float32View.byteLength);
}

/** KNN match returned by sqlite-vec (smaller distance = more similar). */
export interface VecMatch {
  rowid: number;
  distance: number;
}

/** The two vector tables: prose (nomic) and code (jina). */
type VectorTable = "chunks_vec" | "chunks_vec_code";

/** Insert a vector, tied to a chunk rowid, into one of the vector tables. */
function insertVectorInto(db: DatabaseSync, table: VectorTable, rowid: number, vector: number[] | Float32Array) {
  // CAST is required: node:sqlite binds JS numbers as REAL and vec0 rejects
  // non-integer primary keys.
  preparedStatement(db, `INSERT INTO ${table}(rowid, embedding) VALUES (CAST(? AS INTEGER), ?)`)
    .run(rowid, float32ToBuffer(vector));
}

/** Delete every vector tied to a file's chunks, from one vector table. */
function deleteVectorsForFileFrom(db: DatabaseSync, table: VectorTable, filePath: string) {
  preparedStatement(db, `DELETE FROM ${table} WHERE rowid IN (SELECT rowid FROM chunks WHERE file_path = ?)`)
    .run(filePath);
}

/** KNN search one vector table for the query vector, limited to `limit`. */
function searchVectorsIn(db: DatabaseSync, table: VectorTable, queryVector: number[], limit: number): VecMatch[] {
  return preparedStatement(db, `
    SELECT rowid, distance
    FROM ${table}
    WHERE embedding MATCH ?
    LIMIT ?
  `).all(float32ToBuffer(queryVector), limit) as unknown as VecMatch[];
}

/** Count rows in one vector table (vector-coverage statistics). */
function countVectorsIn(db: DatabaseSync, table: VectorTable): number {
  const row = preparedStatement(db, `SELECT COUNT(*) as count FROM ${table}`).get() as { count: number };
  return row.count;
}

// Text-model vectors (chunks_vec / nomic).

/** Insert a prose chunk's nomic vector. */
export function insertVector(db: DatabaseSync, rowid: number, vector: number[] | Float32Array) {
  insertVectorInto(db, "chunks_vec", rowid, vector);
}

/** Delete a file's prose vectors (re-index preparation). */
export function deleteVectorsForFile(db: DatabaseSync, filePath: string) {
  deleteVectorsForFileFrom(db, "chunks_vec", filePath);
}

/** KNN search the prose vector space. */
export function searchVectors(db: DatabaseSync, queryVector: number[], limit: number): VecMatch[] {
  return searchVectorsIn(db, "chunks_vec", queryVector, limit);
}

/** Number of chunks with text-model (nomic) vectors. */
export function getEmbeddedCount(db: DatabaseSync): number {
  return countVectorsIn(db, "chunks_vec");
}

// Code-model vectors (chunks_vec_code / jina).

/** Insert a code chunk's jina vector. */
export function insertCodeVector(db: DatabaseSync, rowid: number, vector: number[] | Float32Array) {
  insertVectorInto(db, "chunks_vec_code", rowid, vector);
}

/** Delete a file's code vectors (re-index preparation). */
export function deleteCodeVectorsForFile(db: DatabaseSync, filePath: string) {
  deleteVectorsForFileFrom(db, "chunks_vec_code", filePath);
}

/** KNN search the code vector space. */
export function searchCodeVectors(db: DatabaseSync, queryVector: number[], limit: number): VecMatch[] {
  return searchVectorsIn(db, "chunks_vec_code", queryVector, limit);
}

/** Number of chunks with code-model (jina) vectors. */
export function getCodeEmbeddedCount(db: DatabaseSync): number {
  return countVectorsIn(db, "chunks_vec_code");
}

/**
 * Wipe all indexed content: both vector tables, all chunks + file rows, and
 * rebuild the FTS index. Embedding-model metadata is kept by the caller's
 * contract (it reflects the configured model, not indexed content).
 */
export function clearAllVectors(db: DatabaseSync) {
  db.exec("DELETE FROM chunks_vec; DELETE FROM chunks_vec_code; DELETE FROM chunks; DELETE FROM files;");
  db.exec("INSERT INTO chunks_fts(chunks_fts) VALUES('rebuild')");
}

// ─── FTS (chunks_fts / BM25) ──────────────────────────────────────────────

/** One BM25 match: the chunk rowid and its (lower-is-better) bm25 score. */
export interface FtsMatch {
  rowid: number;
  bm25_score: number;
}

/** Full-text search via FTS5, ordered best-first, limited to `limit`. */
export function searchFts(db: DatabaseSync, ftsQuery: string, limit: number): FtsMatch[] {
  return preparedStatement(db, `
    SELECT chunks_fts.rowid, bm25(chunks_fts) as bm25_score
    FROM chunks_fts
    WHERE chunks_fts MATCH ?
    ORDER BY bm25(chunks_fts)
    LIMIT ?
  `).all(ftsQuery, limit) as unknown as FtsMatch[];
}

// ─── Files ────────────────────────────────────────────────────────────────

/** One row of the files table (per-file index bookkeeping). */
export interface FileRow {
  path: string;
  hash: string;
  chunks: number;
  indexed: string;
  size: number;
  embedded: number;
}

/**
 * Bulk {path → hash/embedded} lookup for the indexing producer-side skip
 * check. Uses one cached point-lookup statement; primary-key lookups are
 * O(log n) each, so this stays fast even for tens of thousands of paths.
 */
export function getFilesByPaths(
  db: DatabaseSync,
  paths: string[],
): Map<string, { hash: string; embedded: number }> {
  const resultsByPath = new Map<string, { hash: string; embedded: number }>();
  if (paths.length === 0) return resultsByPath;
  const lookup = preparedStatement(db, "SELECT path, hash, embedded FROM files WHERE path = ?");
  for (const path of paths) {
    const row = lookup.get(path) as { path: string; hash: string; embedded: number } | undefined;
    if (row) resultsByPath.set(row.path, { hash: row.hash, embedded: row.embedded });
  }
  return resultsByPath;
}

/** Insert or update a file's bookkeeping row after (re-)indexing it. */
export function upsertFile(
  db: DatabaseSync,
  path: string,
  hash: string,
  chunks: number,
  indexed: string,
  size: number,
  embedded: boolean,
) {
  preparedStatement(db, `
    INSERT INTO files(path, hash, chunks, indexed, size, embedded)
    VALUES (?, ?, ?, ?, ?, ?)
    ON CONFLICT(path) DO UPDATE SET
      hash=excluded.hash, chunks=excluded.chunks, indexed=excluded.indexed,
      size=excluded.size, embedded=excluded.embedded
  `).run(path, hash, chunks, indexed, size, embedded ? 1 : 0);
}

/** Insert-or-replace variant used by the legacy JSON migration path. */
export function replaceFile(
  db: DatabaseSync,
  path: string,
  hash: string,
  chunks: number,
  indexed: string,
  size: number,
  embedded: boolean,
) {
  preparedStatement(db, `
    INSERT OR REPLACE INTO files(path, hash, chunks, indexed, size, embedded)
    VALUES (?, ?, ?, ?, ?, ?)
  `).run(path, hash, chunks, indexed, size, embedded ? 1 : 0);
}

/** Remove a file's bookkeeping row (file deleted / excluded). */
export function deleteFile(db: DatabaseSync, path: string) {
  preparedStatement(db, "DELETE FROM files WHERE path = ?").run(path);
}

/** Flag a file's chunks as embedded (or not) — used to force re-embedding. */
export function setFileEmbedded(db: DatabaseSync, path: string, embedded: boolean) {
  preparedStatement(db, "UPDATE files SET embedded = ? WHERE path = ?").run(embedded ? 1 : 0, path);
}

/** All file bookkeeping rows. */
export function listFiles(db: DatabaseSync): FileRow[] {
  return preparedStatement(db, "SELECT * FROM files").all() as unknown as FileRow[];
}

/** Just the indexed file paths (cheaper than full rows). */
export function listFilePaths(db: DatabaseSync): string[] {
  return (preparedStatement(db, "SELECT path FROM files").all() as unknown as Array<{ path: string }>).map(row => row.path);
}

/** Number of distinct indexed files. */
export function countFiles(db: DatabaseSync): number {
  const row = preparedStatement(db, "SELECT COUNT(*) as totalFiles FROM files").get() as { totalFiles: number };
  return row.totalFiles;
}

// ─── Metadata ─────────────────────────────────────────────────────────────

/**
 * Canonical metadata key names — the single source of truth so call sites
 * never spell out "last_build" / "embedding_model" as string literals.
 */
export const MetadataKey = {
  LastBuild: "last_build",
  EmbeddingModel: "embedding_model",
  EmbeddingCodeModel: "embedding_model_code",
  EmbeddingCodeScheme: "embedding_code_scheme",
} as const;

export type MetadataKey = typeof MetadataKey[keyof typeof MetadataKey];

/** Read a metadata value; undefined when the key is absent. */
export function getMetadata(db: DatabaseSync, key: MetadataKey): string | undefined {
  const row = preparedStatement(db, "SELECT value FROM metadata WHERE key = ?").get(key) as { value?: string } | undefined;
  return row?.value;
}

/** Write a metadata value (insert or replace). */
export function setMetadata(db: DatabaseSync, key: MetadataKey, value: string) {
  preparedStatement(db, "INSERT OR REPLACE INTO metadata(key, value) VALUES (?, ?)").run(key, value);
}

/** Remove a metadata key. */
export function deleteMetadata(db: DatabaseSync, key: MetadataKey) {
  preparedStatement(db, "DELETE FROM metadata WHERE key = ?").run(key);
}

// ─── Statistics ───────────────────────────────────────────────────────────

/** Total chunk count and summed token estimate, in one query. */
export function getChunkStats(db: DatabaseSync): { totalChunks: number; totalTokens: number } {
  return preparedStatement(db, `
    SELECT COUNT(*) as totalChunks, COALESCE(SUM(tokens), 0) as totalTokens
    FROM chunks
  `).get() as { totalChunks: number; totalTokens: number };
}

/** Plain chunk count — used by the schema-migration guards. */
export function countChunksTotal(db: DatabaseSync): number {
  const row = preparedStatement(db, "SELECT COUNT(*) as count FROM chunks").get() as { count: number };
  return row.count;
}
