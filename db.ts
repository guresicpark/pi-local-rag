import { existsSync, readFileSync, unlinkSync } from "node:fs";
import Database from "better-sqlite3";
import { load as loadVec } from "sqlite-vec";
import { getRagDir, dbFile, legacyIndexFile, ensureDir } from "./store.ts";
import { VECTOR_DIM } from "./constants.ts";

export interface Chunk {
  id: string;
  file: string;
  content: string;
  lineStart: number;
  lineEnd: number;
  hash: string;
  indexed: string;
  tokens: number;
}

interface FileDbEntry {
  path: string;
  hash: string;
  chunks: number;
  indexed: string;
  size: number;
  embedded: number;
}

interface FileEntry {
  hash: string;
  chunks: number;
  indexed: string;
  size: number;
  embedded: boolean;
}

export interface IndexMeta {
  chunks: Chunk[];
  files: Record<string, FileEntry>;
  lastBuild: string;
  embeddingModel?: string;
}

export interface IndexStats {
  totalChunks: number;
  totalFiles: number;
  totalTokens: number;
  embeddedCount: number;
  lastBuild: string;
  embeddingModel: string;
}

export class RagDatabase {
  private static _instance: Database.Database | null = null;
  private constructor() {}

  static get instance(): Database.Database {
    return RagDatabase._instance ??= RagDatabase.open();
  }

  static get isOpen(): boolean { return RagDatabase._instance !== null; }

  static close(): void {
    const db = RagDatabase._instance;
    RagDatabase._instance = null;
    try {
      db?.close();
    } catch (err) {
      process.stderr.write(`[rag] closeDb() failed: ${(err as Error).message}\n`);
    }
  }

  static open(ragDir?: string): Database.Database {
    const dir = ragDir ?? getRagDir();
    ensureDir(dir);
    const path = dbFile(dir);
    const db = new Database(path);
    db.pragma("journal_mode = WAL");
    db.pragma("foreign_keys = ON");
    loadVec(db);
    initSchema(db);

    const legacyPath = legacyIndexFile(dir);
    if (existsSync(legacyPath)) {
      const chunkCount = db.prepare("SELECT COUNT(*) as c FROM chunks").get() as { c: number };
      if (chunkCount.c === 0) {
        migrateFromJson(db, legacyPath);
      }
    }

    return db;
  }

  static getFreshDbConn(ragDir?: string): Database.Database & Disposable {
    const db = RagDatabase.open(ragDir);
    return Object.assign(db, {
      [Symbol.dispose]: () => db.close(),
    });
  }
}

export const getDbConn   = () => RagDatabase.instance;
export const closeDbConn = () => { RagDatabase.close(); };

/**
 * Returns a brand-new, throwaway DB connection. **Bypasses the singleton** —
 * the caller is responsible for closing it. Use `getDbConn()` for normal access.
 */
export const getFreshDbConn = (dir?: string) => RagDatabase.getFreshDbConn(dir);

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
}

function migrateFromJson(db: Database.Database, jsonPath: string): void {
  let data: IndexMeta;
  try {
    data = JSON.parse(readFileSync(jsonPath, "utf-8"));
  } catch { return; }

  if (!data.chunks || data.chunks.length === 0) {
    try { unlinkSync(jsonPath); } catch {}
    return;
  }

  const tx = db.transaction(() => {
    const insChunk = db.prepare(`
      INSERT INTO chunks(id, file_path, chunk_content, line_start, line_end, chunk_hash, indexed_at, tokens)
      VALUES (?, ?, ?, ?, ?, ?, ?, ?)
    `);
    const insFile = db.prepare(`
      INSERT OR REPLACE INTO files(path, hash, chunks, indexed, size, embedded)
      VALUES (?, ?, ?, ?, ?, ?)
    `);

    for (const c of data.chunks) {
      insChunk.run(c.id, c.file, c.content, c.lineStart, c.lineEnd, c.hash, c.indexed, c.tokens);
    }

    for (const [fp, info] of Object.entries(data.files || {})) {
      insFile.run(fp, info.hash, info.chunks, info.indexed, info.size, info.embedded ? 1 : 0);
    }

    if (data.lastBuild) {
      db.prepare("INSERT OR REPLACE INTO metadata(key, value) VALUES ('last_build', ?)").run(data.lastBuild);
    }
    if (data.embeddingModel) {
      db.prepare("INSERT OR REPLACE INTO metadata(key, value) VALUES ('embedding_model', ?)").run(data.embeddingModel);
    }
  });

  tx();
  try { unlinkSync(jsonPath); } catch {}
}

export function float32ToBuffer(arr: number[]): Buffer {
  const f = new Float32Array(arr);
  return Buffer.from(f.buffer, f.byteOffset, f.byteLength);
}

export function getIndexStats(db?: Database.Database): IndexStats {
  const dbConn = db ?? getDbConn();
  const chunkRow = dbConn.prepare(`
    SELECT COUNT(*) as totalChunks,
          COALESCE(SUM(tokens), 0) as totalTokens
    FROM chunks
  `).get() as { totalChunks: number; totalTokens: number };

  const fileRow = dbConn.prepare("SELECT COUNT(*) as totalFiles FROM files").get() as { totalFiles: number };
  const vecRow = dbConn.prepare("SELECT COUNT(*) as embeddedCount FROM chunks_vec").get() as { embeddedCount: number };
  const lastBuild = dbConn.prepare("SELECT value FROM metadata WHERE key = 'last_build'").get() as { value?: string } | undefined;
  const embeddingModel = dbConn.prepare("SELECT value FROM metadata WHERE key = 'embedding_model'").get() as { value?: string } | undefined;

  return {
    totalChunks: chunkRow.totalChunks,
    totalFiles: fileRow.totalFiles,
    totalTokens: chunkRow.totalTokens,
    embeddedCount: vecRow.embeddedCount,
    lastBuild: lastBuild?.value ?? "",
    embeddingModel: embeddingModel?.value ?? "",
  };
}

/** No-op shim — JSON-era callers (and tests) compile against this. SQLite
 *  writes are committed by indexFiles' transactions; there is no separate
 *  save step. Kept on the public surface to avoid breaking external imports. */
export function saveIndex(_index: IndexMeta) { /* writes are transactional in indexFiles */ }

export function loadIndex(): IndexMeta {
  const db = getDbConn();
  const chunks = db.prepare(`
    SELECT c.id, c.file_path as file, c.chunk_content as content,
            c.line_start as lineStart, c.line_end as lineEnd,
            c.chunk_hash as hash, c.indexed_at as indexed, c.tokens
    FROM chunks c
  `).all() as Chunk[];

  const filesRaw = db.prepare("SELECT * FROM files").all() as Array<FileDbEntry>;
  const files: IndexMeta["files"] = {};
  for (const f of filesRaw) {
    files[f.path] = {hash: f.hash, chunks: f.chunks, indexed: f.indexed, size: f.size, embedded: !!f.embedded};
  }

  const lastBuild = db.prepare("SELECT value FROM metadata WHERE key = 'last_build'").get() as { value?: string } | undefined;
  const embeddingModel = db.prepare("SELECT value FROM metadata WHERE key = 'embedding_model'").get() as { value?: string } | undefined;

  return {
    chunks, files,
    lastBuild: lastBuild?.value ?? "",
    embeddingModel: embeddingModel?.value,
  };
}

export function getEmbeddedCount(): number {
  const db = getDbConn();
  const vecRow = db.prepare("SELECT COUNT(*) as embeddedCount FROM chunks_vec").get() as { embeddedCount: number };
  return vecRow.embeddedCount;
}

export function getIndexedFiles(): FileDbEntry[] {
  const db = getDbConn();
  return (db.prepare("SELECT * FROM files").all() as Array<FileDbEntry>);
}
