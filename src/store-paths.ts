/**
 * Resolution of the RAG store directory and the well-known files inside it.
 *
 * Storage is per-project: walk upward from the working directory looking for
 * a `.pi/rag/` project store; fall back to `~/.pi/rag/` as the global
 * default. The first `/rag index` in a directory with no parent store
 * creates one at the current cwd.
 */
import { existsSync, mkdirSync, renameSync } from "node:fs";
import { join, dirname } from "node:path";
import { homedir } from "node:os";

/**
 * Legacy `~/.pi/lens` store from the pre-rename era. Overridable via the
 * `PI_RAG_LEGACY_DIR` environment variable so tests can exercise the
 * one-shot directory migration.
 */
export const LEGACY_DIR = process.env.PI_RAG_LEGACY_DIR ?? join(homedir(), ".pi", "lens");

/**
 * Global fallback store directory. Lazily evaluated (a function, not a
 * constant) so tests that override `$HOME` are honored at call time.
 */
export const GLOBAL_RAG_DIR = () => join(homedir(), ".pi", "rag");

/**
 * Resolve the active RAG store directory for the current cwd.
 *
 * Priority order:
 * 1. `$PI_RAG_DIR` — environment variable override.
 * 2. Walk upward from `process.cwd()` looking for an existing `.pi/rag/`,
 *    stopping before `homedir()` so the global store at `~/.pi/rag/` is
 *    only reached as an explicit fallback (not via the walk-up).
 * 3. With `createIfMissing`, create `${cwd}/.pi/rag/`.
 * 4. Otherwise, fall back to `${homedir()}/.pi/rag/`.
 */
export function getRagDir(options: { createIfMissing?: boolean } = {}): string {
  // Environment override wins over everything.
  const envOverride = process.env.PI_RAG_DIR;
  if (envOverride) {
    if (!existsSync(envOverride)) mkdirSync(envOverride, { recursive: true });
    return envOverride;
  }

  const homeDirectory = homedir();
  let currentDirectory = process.cwd();

  // Walk-up search, stopping before $HOME so we don't accidentally pick up
  // ~/.pi/rag via the walk (that path is reached only as the explicit
  // fallback below).
  while (true) {
    if (currentDirectory === homeDirectory) break;
    const candidateStore = join(currentDirectory, ".pi", "rag");
    if (existsSync(candidateStore)) return candidateStore;
    const parentDirectory = dirname(currentDirectory);
    if (parentDirectory === currentDirectory) break; // reached filesystem root
    currentDirectory = parentDirectory;
  }

  // Anchor a new project-local store at the current cwd when asked to.
  if (options.createIfMissing) {
    const projectStore = join(process.cwd(), ".pi", "rag");
    mkdirSync(projectStore, { recursive: true });
    return projectStore;
  }

  // Fallback: home-dir global store. ensureStoreDirectory handles creation
  // plus the legacy lens→rag migration.
  const globalStore = GLOBAL_RAG_DIR();
  ensureStoreDirectory(globalStore);
  return globalStore;
}

/** Path of the SQLite database file inside a store directory. */
export function databaseFilePath(ragDir: string): string {
  return join(ragDir, "rag.db");
}

/**
 * Path of the legacy JSON index inside a store directory. Kept for the
 * one-shot auto-migration performed when a database connection is opened.
 */
export function legacyJsonIndexPath(ragDir: string): string {
  return join(ragDir, "index.json");
}

/** Path of the JSON config file inside a store directory. */
export function configFilePath(ragDir: string): string {
  return join(ragDir, "config.json");
}

/**
 * Create `ragDir` if it does not exist yet. When the global home-dir store
 * is being created and a legacy `~/.pi/lens` directory is present, it is
 * renamed into place instead (one-shot migration of the whole store).
 */
export function ensureStoreDirectory(ragDir: string): void {
  if (existsSync(ragDir)) return;
  // The lens→rag migration only applies to the home-dir global store.
  if (ragDir === GLOBAL_RAG_DIR() && existsSync(LEGACY_DIR)) {
    try {
      renameSync(LEGACY_DIR, ragDir);
      return;
    } catch {
      // Fall through to plain directory creation (e.g. cross-device rename).
    }
  }
  mkdirSync(ragDir, { recursive: true });
}
