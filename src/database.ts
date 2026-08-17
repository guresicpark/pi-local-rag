/**
 * SQLite connection lifecycle and store-level operations on top of the
 * repository layer: the shared singleton connection, throwaway connections
 * (with `Symbol.dispose` support for `using` declarations), index
 * statistics, the legacy index.json one-shot migration, and the full store
 * reset.
 *
 * Runs on the built-in `node:sqlite` driver — connections are opened with
 * `allowExtension: true` so sqlite-vec's loadable extension can be attached.
 */
import { existsSync, readFileSync, readdirSync, rmSync, unlinkSync } from "node:fs";
import { join } from "node:path";
import { DatabaseSync } from "node:sqlite";
import { load as loadSqliteVec } from "sqlite-vec";
import { getRagDir, databaseFilePath, legacyJsonIndexPath, ensureStoreDirectory } from "./store-paths.ts";
import { defaultConfig, saveConfig } from "./config.ts";
import * as sqlRepository from "./repository.ts";

export { initSchema } from "./repository.ts";

/** A chunk as exposed through the public API (legacy camelCase naming). */
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

/** Per-file bookkeeping entry inside IndexMeta. */
interface FileEntry {
  hash: string;
  chunks: number;
  indexed: string;
  size: number;
  embedded: boolean;
}

/** Legacy JSON-era index shape, now hydrated from SQLite. */
export interface IndexMeta {
  chunks: Chunk[];
  files: Record<string, FileEntry>;
  lastBuild: string;
  embeddingModel?: string;
}

/** Aggregate statistics about the index, for /rag status and tools. */
export interface IndexStats {
  totalChunks: number;
  totalFiles: number;
  totalTokens: number;
  /** Chunks with text-model (nomic) vectors. */
  embeddedCount: number;
  /** Chunks with code-model (jina) vectors. */
  embeddedCodeCount: number;
  lastBuild: string;
  embeddingModel: string;
  codeEmbeddingModel: string;
}

// ─── Connection management ────────────────────────────────────────────────

/**
 * Open a brand-new connection to the store in `ragDir` (or the directory
 * resolved by getRagDir()). Creates the directory, applies WAL, loads the
 * sqlite-vec extension, initializes the schema, and consumes a legacy
 * index.json when one is sitting in the store.
 */
export function openDatabase(ragDir?: string): DatabaseSync {
  const storeDirectory = ragDir ?? getRagDir();
  ensureStoreDirectory(storeDirectory);
  const connection = new DatabaseSync(databaseFilePath(storeDirectory), { allowExtension: true });
  connection.exec("PRAGMA journal_mode = WAL;");
  connection.exec("PRAGMA foreign_keys = ON;");
  loadSqliteVec(connection);
  sqlRepository.initSchema(connection);

  const legacyJsonPath = legacyJsonIndexPath(storeDirectory);
  if (existsSync(legacyJsonPath)) {
    // One-shot import: only when the database is still empty, so a leftover
    // JSON file can never clobber an existing SQLite index.
    if (sqlRepository.countChunksTotal(connection) === 0) {
      migrateFromJsonIndex(connection, legacyJsonPath);
    }
  }

  return connection;
}

/** The shared singleton connection — opened lazily on first use. */
let sharedConnection: DatabaseSync | null = null;

/** Get the shared connection for the active store (lazily opened). */
export function getDbConn(): DatabaseSync {
  return (sharedConnection ??= openDatabase());
}

/** Close the shared connection if one is open (safe to call repeatedly). */
export function closeDbConn(): void {
  const connection = sharedConnection;
  sharedConnection = null;
  try {
    connection?.close();
  } catch (error) {
    process.stderr.write(`[rag] closeDbConn() failed: ${(error as Error).message}\n`);
  }
}

/**
 * Run `operation` against the shared connection and close it afterwards
 * (also on throw), so the singleton never leaks an open handle — or the WAL
 * / file descriptor backing it — across agent turns. This is the one place
 * that should pair getDbConn() with closeDbConn().
 */
export async function withDb<T>(operation: (db: DatabaseSync) => T | Promise<T>): Promise<T> {
  const db = getDbConn();
  try {
    return await operation(db);
  } finally {
    closeDbConn();
  }
}

/**
 * Returns a brand-new, throwaway connection. **Bypasses the singleton** —
 * the caller is responsible for closing it (directly, or automatically via
 * a `using` declaration thanks to the attached Symbol.dispose). Use
 * getDbConn() for normal access.
 */
export function getFreshDbConn(ragDir?: string): DatabaseSync & Disposable {
  const connection = openDatabase(ragDir);
  return Object.assign(connection, {
    [Symbol.dispose]: () => connection.close(),
  });
}

// ─── Legacy index.json migration ──────────────────────────────────────────

/**
 * One-shot import of the pre-SQLite JSON index into the database, then
 * unlink the JSON file so the migration never runs twice. Silently no-ops
 * on unreadable/empty payloads.
 */
function migrateFromJsonIndex(db: DatabaseSync, jsonPath: string): void {
  let legacyIndex: IndexMeta;
  try {
    legacyIndex = JSON.parse(readFileSync(jsonPath, "utf-8"));
  } catch {
    return;
  }

  if (!legacyIndex.chunks || legacyIndex.chunks.length === 0) {
    try { unlinkSync(jsonPath); } catch { /* already gone */ }
    return;
  }

  sqlRepository.runInTransaction(db, () => {
    for (const chunk of legacyIndex.chunks) {
      sqlRepository.insertChunk(db, {
        id: chunk.id,
        filePath: chunk.file,
        content: chunk.content,
        lineStart: chunk.lineStart,
        lineEnd: chunk.lineEnd,
        hash: chunk.hash,
        indexedAt: chunk.indexed,
        tokens: chunk.tokens,
      });
    }

    for (const [filePath, fileInfo] of Object.entries(legacyIndex.files ?? {})) {
      sqlRepository.replaceFile(
        db, filePath, fileInfo.hash, fileInfo.chunks, fileInfo.indexed, fileInfo.size, fileInfo.embedded,
      );
    }

    if (legacyIndex.lastBuild) {
      sqlRepository.setMetadata(db, sqlRepository.MetadataKey.LastBuild, legacyIndex.lastBuild);
    }
    if (legacyIndex.embeddingModel) {
      sqlRepository.setMetadata(db, sqlRepository.MetadataKey.EmbeddingModel, legacyIndex.embeddingModel);
    }
  });

  try { unlinkSync(jsonPath); } catch { /* already gone */ }
}

// ─── Statistics ───────────────────────────────────────────────────────────

/** Compute aggregate index statistics (optionally on an explicit connection). */
export function getIndexStats(db?: DatabaseSync): IndexStats {
  const connection = db ?? getDbConn();
  const { totalChunks, totalTokens } = sqlRepository.getChunkStats(connection);

  return {
    totalChunks,
    totalFiles: sqlRepository.countFiles(connection),
    totalTokens,
    embeddedCount: sqlRepository.getEmbeddedCount(connection),
    embeddedCodeCount: sqlRepository.getCodeEmbeddedCount(connection),
    lastBuild: sqlRepository.getMetadata(connection, sqlRepository.MetadataKey.LastBuild) ?? "",
    embeddingModel: sqlRepository.getMetadata(connection, sqlRepository.MetadataKey.EmbeddingModel) ?? "",
    codeEmbeddingModel: sqlRepository.getMetadata(connection, sqlRepository.MetadataKey.EmbeddingCodeModel) ?? "",
  };
}

/**
 * No-op shim — JSON-era callers (and tests) compile against this. SQLite
 * writes are committed by indexFiles' transactions; there is no separate
 * save step. Kept on the public surface to avoid breaking external imports.
 */
export function saveIndex(_index: IndexMeta) {
  /* writes are transactional in indexFiles */
}

// ─── Wipe / reset ─────────────────────────────────────────────────────────

/**
 * Wipe all indexed data: chunks, vectors, FTS rows, file entries, and the
 * last-build timestamp. Embedding-model metadata is kept (it reflects the
 * configured model, not indexed content).
 */
export function clearIndex(db?: DatabaseSync) {
  const connection = db ?? getDbConn();
  sqlRepository.clearAllVectors(connection);
  sqlRepository.deleteMetadata(connection, sqlRepository.MetadataKey.LastBuild);
}

/**
 * Factory-reset the active RAG store (what /rag clear calls): close the
 * shared connection, delete every entry in the store directory (rag.db +
 * WAL/SHM sidecars, config.json, legacy index.json, anything else), then
 * regenerate fresh defaults — a default config.json and an empty
 * schema-initialized rag.db. Returns the store directory path.
 */
export function resetStore(): string {
  const storeDirectory = getRagDir();
  closeDbConn();
  if (existsSync(storeDirectory)) {
    for (const entry of readdirSync(storeDirectory)) {
      rmSync(join(storeDirectory, entry), { recursive: true, force: true });
    }
  }
  ensureStoreDirectory(storeDirectory);
  saveConfig(defaultConfig());
  openDatabase(storeDirectory).close();
  return storeDirectory;
}

// ─── Legacy IndexMeta hydration ───────────────────────────────────────────

/**
 * Rebuild the legacy IndexMeta shape (chunks + files + metadata) from
 * SQLite. Materializes every chunk's content — prefer getIndexedPaths()
 * when only paths are needed.
 */
export function loadIndex(db?: DatabaseSync): IndexMeta {
  const connection = db ?? getDbConn();
  const chunks = sqlRepository.getAllChunks(connection) as Chunk[];

  const fileRows = sqlRepository.listFiles(connection);
  const files: IndexMeta["files"] = {};
  for (const fileRow of fileRows) {
    files[fileRow.path] = {
      hash: fileRow.hash,
      chunks: fileRow.chunks,
      indexed: fileRow.indexed,
      size: fileRow.size,
      embedded: !!fileRow.embedded,
    };
  }

  return {
    chunks,
    files,
    lastBuild: sqlRepository.getMetadata(connection, sqlRepository.MetadataKey.LastBuild) ?? "",
    embeddingModel: sqlRepository.getMetadata(connection, sqlRepository.MetadataKey.EmbeddingModel),
  };
}

/**
 * Just the indexed file paths — cheaper than loadIndex(), which also
 * materializes every chunk's content string.
 */
export function getIndexedPaths(db?: DatabaseSync): string[] {
  return sqlRepository.listFilePaths(db ?? getDbConn());
}
