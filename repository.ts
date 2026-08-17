import type Database from "better-sqlite3";
import { EMBEDDING_MODEL, VECTOR_DIM, CODE_EMBEDDING_MODEL, CODE_VECTOR_DIM, CODE_EMBED_SCHEME, DEFAULT_CODE_EXTS } from "./constants.ts";

/**
 * Centralizes every raw SQL statement used across db.ts, indexing.ts,
 * search.ts, and index.ts. Nothing outside this file should contain a
 * `.prepare` / `.exec` call against the rag database — that keeps the
 * schema-to-code contract in one place instead of scattered across four
 * files with subtly duplicated INSERT shapes.
 *
 * These are plain functions, not a class with cached prepared statements:
 * better-sqlite3 already caches statements per-connection internally via
 * db.prepare, and singleton lifetime (open/close) is still owned by
 * RagDatabase in db.ts. This module is just where the SQL text lives.
 */

// ─── Schema ──────────────────────────────────────────────────────────────

export function initSchema(db: Database.Database) {
  db.exec(`DROP TRIGGER IF EXISTS chunks_ai; DROP TRIGGER IF EXISTS chunks_ad;`);
  db.exec(`
    CREATE TABLE IF NOT EXISTS metadata (
      key   TEXT PRIMARY KEY,
      value TEXT NOT NULL
    );

    CREATE TABLE IF NOT EXISTS chunks (
      id          TEXT PRIMARY KEY,
      file_path   TEXT NOT NULL,
      chunk_content TEXT NOT NULL,
      line_start  INTEGER NOT NULL,
      line_end    INTEGER NOT NULL,
      chunk_hash  TEXT NOT NULL,
      indexed_at  TEXT NOT NULL,
      tokens      INTEGER NOT NULL
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
  // incompatible (different spaces even at equal dimension), and vec0 tables
  // keep their original float[N] definition from when the store was created.
  //
  // - text model changed → drop chunks_vec + wipe all content (legacy
  //   behavior; trackedPaths live in config.json and survive).
  // - code model added/changed → drop chunks_vec_code and delete ONLY
  //   code-classified files' chunks; prose chunks keep their nomic vectors,
  //   so upgrading to the dual-model scheme doesn't re-embed the docs.
  // `/rag rebuild` restores whatever was cleared.
  const storedModel = getMetadata(db, MetadataKey.EmbeddingModel);
  if (storedModel && storedModel !== EMBEDDING_MODEL) {
    db.exec("DROP TABLE IF EXISTS chunks_vec; DELETE FROM chunks_vec_code; DELETE FROM chunks; DELETE FROM files;");
    db.exec("INSERT INTO chunks_fts(chunks_fts) VALUES('rebuild')");
    db.exec(`CREATE VIRTUAL TABLE IF NOT EXISTS chunks_vec USING vec0(
      embedding float[${VECTOR_DIM}]
    );`);
    deleteMetadata(db, MetadataKey.EmbeddingModel);
    process.stderr.write(
      `[rag] text embedding model changed (${storedModel} → ${EMBEDDING_MODEL}); index cleared — run /rag rebuild to re-index\n`,
    );
  }

  // Classification here uses the DEFAULT code extensions: stores created
  // before the dual-model split only ever contained default-ext files, so
  // user-configured extras can't appear in them. The wipe only fires on a
  // genuine pre-dual-model index — text-model metadata present (every
  // indexFiles run records it) but code-model metadata absent/changed.
  const storedCodeModel = getMetadata(db, MetadataKey.EmbeddingCodeModel);
  if (storedCodeModel !== CODE_EMBEDDING_MODEL) {
    if (storedModel && countChunksTotal(db) > 0) {
      resetCodeSpace(db);
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

  // Code-document embedding *scheme* change (how code chunks are prepared for
  // the model, e.g. the filename-context header) — code vectors are
  // incompatible even though the model id is unchanged, so code-classified
  // files are cleared (prose/nomic vectors survive) and `/rag rebuild`
  // re-embeds them. Mirrors the model-change wipe above.
  const storedCodeScheme = getMetadata(db, MetadataKey.EmbeddingCodeScheme);
  if (storedCodeScheme !== CODE_EMBED_SCHEME) {
    if (storedModel && countChunksTotal(db) > 0) {
      resetCodeSpace(db);
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
 * file's chunks/vectors/files (FTS row rebuilt). Prose chunks + nomic vectors
 * are left untouched. Used by both the code-model-change and code-scheme-change
 * migrations.
 */
function resetCodeSpace(db: Database.Database) {
  const codeChunks = DEFAULT_CODE_EXTS.map(e => `file_path LIKE '%${e}'`).join(" OR ");
  const codePaths = DEFAULT_CODE_EXTS.map(e => `path LIKE '%${e}'`).join(" OR ");
  db.exec("DROP TABLE IF EXISTS chunks_vec_code;");
  db.exec(`
    DELETE FROM chunks_vec WHERE rowid IN (SELECT rowid FROM chunks WHERE ${codeChunks});
    DELETE FROM chunks WHERE ${codeChunks};
    DELETE FROM files WHERE ${codePaths};
  `);
  db.exec("INSERT INTO chunks_fts(chunks_fts) VALUES('rebuild')");
  db.exec(`CREATE VIRTUAL TABLE IF NOT EXISTS chunks_vec_code USING vec0(
    embedding float[${CODE_VECTOR_DIM}]
  );`);
}

// ─── Chunks ──────────────────────────────────────────────────────────────

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

export function hasAnyChunks(db: Database.Database): boolean {
  return !!db.prepare("SELECT 1 FROM chunks LIMIT 1").get();
}

export function insertChunk(db: Database.Database, c: NewChunk) {
  return db.prepare(`
    INSERT INTO chunks(id, file_path, chunk_content, line_start, line_end, chunk_hash, indexed_at, tokens)
    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
  `).run(c.id, c.filePath, c.content, c.lineStart, c.lineEnd, c.hash, c.indexedAt, c.tokens);
}

export function deleteChunksForFile(db: Database.Database, filePath: string) {
  db.prepare("DELETE FROM chunks WHERE file_path = ?").run(filePath);
}

export function getChunksByRowids(db: Database.Database, rowids: number[]): ChunkRow[] {
  if (rowids.length === 0) return [];
  const placeholders = rowids.map(() => "?").join(",");
  return db.prepare(`
    SELECT rowid, id, file_path, chunk_content, line_start, line_end,
            chunk_hash, indexed_at, tokens
    FROM chunks
    WHERE rowid IN (${placeholders})
  `).all(...rowids) as ChunkRow[];
}

export interface LoadedChunk {
  id: string; file: string; content: string;
  lineStart: number; lineEnd: number;
  hash: string; indexed: string; tokens: number;
}

export function getAllChunks(db: Database.Database): LoadedChunk[] {
  return db.prepare(`
    SELECT c.id, c.file_path as file, c.chunk_content as content,
            c.line_start as lineStart, c.line_end as lineEnd,
            c.chunk_hash as hash, c.indexed_at as indexed, c.tokens
    FROM chunks c
  `).all() as LoadedChunk[];
}

// ─── Vectors (chunks_vec) ────────────────────────────────────────────────

/**
 * Internal only — not part of the module's public surface. Vector params
 * come in as `number[]` or `Float32Array` (the embedBatchFor fast path);
 * every caller converts through this before it touches chunks_vec.
 **/
export function float32ToBuffer(arr: number[] | Float32Array): Buffer {
  // Float32Array: Buffer.view over the existing store — zero copy. Safe to
  // share: better-sqlite3 copies the bytes into its statement binding at
  // run() time, so the source array is reusable afterwards.
  if (arr instanceof Float32Array) return Buffer.from(arr.buffer, arr.byteOffset, arr.byteLength);
  const f = new Float32Array(arr);
  return Buffer.from(f.buffer, f.byteOffset, f.byteLength);
}

export interface VecMatch {
  rowid: number;
  distance: number
}

type VecTable = "chunks_vec" | "chunks_vec_code";

function insertVectorInto(db: Database.Database, table: VecTable, rowid: number, vector: number[] | Float32Array) {
  db.prepare(`INSERT INTO ${table}(rowid, embedding) VALUES (CAST(? AS INTEGER), ?)`)
    .run(rowid, float32ToBuffer(vector));
}

function deleteVectorsForFileFrom(db: Database.Database, table: VecTable, filePath: string) {
  db.prepare(`DELETE FROM ${table} WHERE rowid IN (SELECT rowid FROM chunks WHERE file_path = ?)`).run(filePath);
}

function searchVectorsIn(db: Database.Database, table: VecTable, queryVec: number[], limit: number): VecMatch[] {
  return db.prepare(`
    SELECT rowid, distance
    FROM ${table}
    WHERE embedding MATCH ?
    LIMIT ?
  `).bind(float32ToBuffer(queryVec), limit).all() as VecMatch[];
}

function countVectorsIn(db: Database.Database, table: VecTable): number {
  const row = db.prepare(`SELECT COUNT(*) as c FROM ${table}`).get() as { c: number };
  return row.c;
}

// ─── Text vectors (chunks_vec / nomic) ───────────────────────────────────

export function insertVector(db: Database.Database, rowid: number, vector: number[] | Float32Array) {
  insertVectorInto(db, "chunks_vec", rowid, vector);
}

export function deleteVectorsForFile(db: Database.Database, filePath: string) {
  deleteVectorsForFileFrom(db, "chunks_vec", filePath);
}

export function searchVectors(db: Database.Database, queryVec: number[], limit: number): VecMatch[] {
  return searchVectorsIn(db, "chunks_vec", queryVec, limit);
}

export function getEmbeddedCount(db: Database.Database): number {
  return countVectorsIn(db, "chunks_vec");
}

// ─── Code vectors (chunks_vec_code / jina) ───────────────────────────────

export function insertCodeVector(db: Database.Database, rowid: number, vector: number[] | Float32Array) {
  insertVectorInto(db, "chunks_vec_code", rowid, vector);
}

export function deleteCodeVectorsForFile(db: Database.Database, filePath: string) {
  deleteVectorsForFileFrom(db, "chunks_vec_code", filePath);
}

export function searchCodeVectors(db: Database.Database, queryVec: number[], limit: number): VecMatch[] {
  return searchVectorsIn(db, "chunks_vec_code", queryVec, limit);
}

export function getCodeEmbeddedCount(db: Database.Database): number {
  return countVectorsIn(db, "chunks_vec_code");
}

export function clearAllVectors(db: Database.Database) {
  db.exec("DELETE FROM chunks_vec; DELETE FROM chunks_vec_code; DELETE FROM chunks; DELETE FROM files;");
  db.exec("INSERT INTO chunks_fts(chunks_fts) VALUES('rebuild')");
}

// ─── Fts (chunks_fts / BM25) ─────────────────────────────────────────────

export interface FtsMatch { rowid: number; bm25_score: number }

export function searchFts(db: Database.Database, ftsQuery: string, limit: number): FtsMatch[] {
  return db.prepare(`
    SELECT chunks_fts.rowid, bm25(chunks_fts) as bm25_score
    FROM chunks_fts
    WHERE chunks_fts MATCH ?
    ORDER BY bm25(chunks_fts)
    LIMIT ?
  `).all(ftsQuery, limit) as FtsMatch[];
}

// ─── Files ───────────────────────────────────────────────────────────────

export interface FileRow {
  path: string;
  hash: string;
  chunks: number;
  indexed: string;
  size: number;
  embedded: number;
}

/** Bulk {path → hash/embedded} lookup for indexFiles' producer-side skip
 *  check — one prepared statement, one query per path, no result-object
 *  overhead for paths absent from the store. */
export function getFilesByPaths(
  db: Database.Database,
  paths: string[],
): Map<string, { hash: string; embedded: number }> {
  const out = new Map<string, { hash: string; embedded: number }>();
  if (paths.length === 0) return out;
  const stmt = db.prepare("SELECT path, hash, embedded FROM files WHERE path = ?");
  for (const p of paths) {
    const row = stmt.get(p) as { path: string; hash: string; embedded: number } | undefined;
    if (row) out.set(row.path, { hash: row.hash, embedded: row.embedded });
  }
  return out;
}

export function upsertFile(
  db: Database.Database,
  path: string, hash: string, chunks: number, indexed: string, size: number, embedded: boolean,
) {
  db.prepare(`
    INSERT INTO files(path, hash, chunks, indexed, size, embedded)
    VALUES (?, ?, ?, ?, ?, ?)
    ON CONFLICT(path) DO UPDATE SET
      hash=excluded.hash, chunks=excluded.chunks, indexed=excluded.indexed,
      size=excluded.size, embedded=excluded.embedded
  `).run(path, hash, chunks, indexed, size, embedded ? 1 : 0);
}

/** Insert-or-replace variant used by the JSON migration path (no upsert semantics needed there). */
export function replaceFile(
  db: Database.Database,
  path: string, hash: string, chunks: number, indexed: string, size: number, embedded: boolean,
) {
  db.prepare(`
    INSERT OR REPLACE INTO files(path, hash, chunks, indexed, size, embedded)
    VALUES (?, ?, ?, ?, ?, ?)
  `).run(path, hash, chunks, indexed, size, embedded ? 1 : 0);
}

export function deleteFile(db: Database.Database, path: string) {
  db.prepare("DELETE FROM files WHERE path = ?").run(path);
}

export function setFileEmbedded(db: Database.Database, path: string, embedded: boolean) {
  db.prepare("UPDATE files SET embedded = ? WHERE path = ?").run(embedded ? 1 : 0, path);
}

export function listFiles(db: Database.Database): FileRow[] {
  return db.prepare("SELECT * FROM files").all() as FileRow[];
}

export function listFilePaths(db: Database.Database): string[] {
  return (db.prepare("SELECT path FROM files").all() as Array<{ path: string }>).map(r => r.path);
}

export function countFiles(db: Database.Database): number {
  return (db.prepare("SELECT COUNT(*) as totalFiles FROM files").get() as { totalFiles: number }).totalFiles;
}

// ─── Metadata ────────────────────────────────────────────────────────────
/**
 * Canonical metadata key names — the single source of truth so call sites
 * never spell out "last_build" / "embedding_model" as string literals.
 **/
export const MetadataKey = {
  LastBuild: "last_build",
  EmbeddingModel: "embedding_model",
  EmbeddingCodeModel: "embedding_model_code",
  EmbeddingCodeScheme: "embedding_code_scheme",
} as const;

export type MetadataKey = typeof MetadataKey[keyof typeof MetadataKey];

export function getMetadata(db: Database.Database, key: MetadataKey): string | undefined {
  return (db.prepare("SELECT value FROM metadata WHERE key = ?").get(key) as { value?: string } | undefined)?.value;
}

export function setMetadata(db: Database.Database, key: MetadataKey, value: string) {
  db.prepare("INSERT OR REPLACE INTO metadata(key, value) VALUES (?, ?)").run(key, value);
}

export function deleteMetadata(db: Database.Database, key: MetadataKey) {
  db.prepare("DELETE FROM metadata WHERE key = ?").run(key);
}

export function getChunkStats(db: Database.Database): { totalChunks: number; totalTokens: number } {
  return db.prepare(`
    SELECT COUNT(*) as totalChunks, COALESCE(SUM(tokens), 0) as totalTokens
    FROM chunks
  `).get() as { totalChunks: number; totalTokens: number };
}

export function countChunksTotal(db: Database.Database): number {
  return (db.prepare("SELECT COUNT(*) as c FROM chunks").get() as { c: number }).c;
}
