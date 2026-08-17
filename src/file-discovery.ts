/**
 * Indexable-file discovery: directory walkers (synchronous and
 * event-loop-friendly asynchronous variants), tracked-path expansion, and
 * gitignore-style exclusion matching.
 */
import { existsSync, readdirSync, statSync, promises as fsPromises } from "node:fs";
import { extname, basename, join, relative } from "node:path";
import ignore from "ignore";
import {
  BINARY_DOC_EXTS,
  TEXT_MAX_BYTES,
  BINARY_DOC_MAX_BYTES,
  SKIP_DIRS,
} from "./constants.ts";
import { loadConfig, resolveExtensions, type RagConfig } from "./config.ts";
import { yieldToEventLoop } from "./runtime-utils.ts";

/**
 * Shared extension/size + exclude-pattern filter for the sync and async
 * walkers — keeps the predicates defined once instead of being copy-pasted
 * into both walkers.
 */
function buildFileFilter(
  allowedExtensions: Set<string>,
  excludeMatcher: ReturnType<typeof ignore> | null,
  walkRoot: string,
) {
  /** Extension + size check; true when the file may be indexed. */
  function isAcceptableFile(filePath: string, fileSize: number): boolean {
    const extension = extname(filePath).toLowerCase();
    if (allowedExtensions.has(extension)) return fileSize < TEXT_MAX_BYTES;
    if (BINARY_DOC_EXTS.has(extension)) return fileSize < BINARY_DOC_MAX_BYTES;
    return false;
  }

  /** gitignore-style check relative to the walk root; false for paths
   *  outside the root (those can't be expressed relative to it). */
  function isExcludedPath(absolutePath: string): boolean {
    if (!excludeMatcher) return false;
    const pathRelativeToRoot = relative(walkRoot, absolutePath);
    if (!pathRelativeToRoot || pathRelativeToRoot.startsWith("..")) return false;
    return excludeMatcher.ignores(pathRelativeToRoot);
  }

  return { isAcceptableFile, isExcludedPath };
}

/**
 * Synchronously collect every indexable file under `dirPath` (or the
 * single file itself when `dirPath` points at one). Applies the configured
 * extension allowlist (or `exts` when given), size caps, skip-directories,
 * and gitignore-style `excludePatterns`. Returns [] for unreadable paths.
 */
export function collectFiles(
  dirPath: string,
  allowedExtensionSet?: Set<string>,
  excludePatterns: string[] = [],
): string[] {
  const allowedExtensions = allowedExtensionSet ?? resolveExtensions(loadConfig());
  const excludeMatcher = excludePatterns.length ? ignore().add(excludePatterns) : null;
  const discoveredFiles: string[] = [];
  const walkRoot = dirPath;
  const { isAcceptableFile, isExcludedPath } = buildFileFilter(allowedExtensions, excludeMatcher, walkRoot);

  // A file path short-circuits the walk: check it directly.
  try {
    const pathStat = statSync(dirPath);
    if (pathStat.isFile()) {
      if (!isAcceptableFile(dirPath, pathStat.size)) return [];
      if (excludeMatcher && excludeMatcher.ignores(basename(dirPath))) return [];
      return [dirPath];
    }
  } catch {
    return [];
  }

  function walk(directory: string) {
    let entries: import("node:fs").Dirent[];
    try {
      entries = readdirSync(directory, { withFileTypes: true });
    } catch {
      return;
    }
    for (const entry of entries) {
      const entryPath = join(directory, entry.name);
      if (entry.isDirectory()) {
        // Skip well-known junk directories and any dot-directory.
        if (SKIP_DIRS.has(entry.name) || entry.name.startsWith(".")) continue;
        if (isExcludedPath(entryPath)) continue;
        walk(entryPath);
      } else {
        const extension = extname(entry.name).toLowerCase();
        if (!allowedExtensions.has(extension) && !BINARY_DOC_EXTS.has(extension)) continue;
        if (isExcludedPath(entryPath)) continue;
        try {
          if (isAcceptableFile(entryPath, statSync(entryPath).size)) discoveredFiles.push(entryPath);
        } catch { /* file vanished mid-walk */ }
      }
    }
  }
  walk(walkRoot);
  return discoveredFiles;
}

/**
 * Asynchronous variant of collectFiles that uses fs.promises and yields to
 * the event loop between directories. Required for /rag rebuild on large
 * trackedPaths (45k+ files) — the synchronous walk pegs the event loop
 * long enough that the TUI freezes before reaching the embed phase.
 */
export async function collectFilesAsync(
  dirPath: string,
  allowedExtensionSet?: Set<string>,
  excludePatterns: string[] = [],
): Promise<string[]> {
  const allowedExtensions = allowedExtensionSet ?? resolveExtensions(loadConfig());
  const excludeMatcher = excludePatterns.length ? ignore().add(excludePatterns) : null;
  const discoveredFiles: string[] = [];
  const walkRoot = dirPath;
  const { isAcceptableFile, isExcludedPath } = buildFileFilter(allowedExtensions, excludeMatcher, walkRoot);

  // A file path short-circuits the walk: check it directly.
  try {
    const pathStat = await fsPromises.stat(dirPath);
    if (pathStat.isFile()) {
      if (!isAcceptableFile(dirPath, pathStat.size)) return [];
      if (excludeMatcher && excludeMatcher.ignores(basename(dirPath))) return [];
      return [dirPath];
    }
  } catch {
    return [];
  }

  async function walk(directory: string): Promise<void> {
    let entries: import("node:fs").Dirent[];
    try {
      entries = await fsPromises.readdir(directory, { withFileTypes: true });
    } catch {
      return;
    }
    for (const entry of entries) {
      const entryPath = join(directory, entry.name);
      if (entry.isDirectory()) {
        if (SKIP_DIRS.has(entry.name) || entry.name.startsWith(".")) continue;
        if (isExcludedPath(entryPath)) continue;
        await walk(entryPath);
      } else {
        const extension = extname(entry.name).toLowerCase();
        if (!allowedExtensions.has(extension) && !BINARY_DOC_EXTS.has(extension)) continue;
        if (isExcludedPath(entryPath)) continue;
        try {
          const entryStat = await fsPromises.stat(entryPath);
          if (isAcceptableFile(entryPath, entryStat.size)) discoveredFiles.push(entryPath);
        } catch { /* file vanished mid-walk */ }
      }
    }
    // Yield between directories so the event loop can process UI updates.
    await yieldToEventLoop();
  }

  await walk(walkRoot);
  return discoveredFiles;
}

/**
 * Expand the config's trackedPaths into the full set of currently
 * indexable files (deduplicated across overlapping tracked roots).
 * Non-existent tracked paths are silently skipped.
 */
export function collectFromTracked(
  config: Pick<RagConfig, "trackedPaths" | "excludePatterns">,
): string[] {
  const uniqueFiles = new Set<string>();
  for (const trackedPath of config.trackedPaths) {
    if (!existsSync(trackedPath)) continue;
    for (const file of collectFiles(trackedPath, undefined, config.excludePatterns)) uniqueFiles.add(file);
  }
  return [...uniqueFiles];
}

/** Async variant of collectFromTracked (see collectFilesAsync for why). */
export async function collectFromTrackedAsync(
  config: Pick<RagConfig, "trackedPaths" | "excludePatterns">,
): Promise<string[]> {
  const uniqueFiles = new Set<string>();
  for (const trackedPath of config.trackedPaths) {
    if (!existsSync(trackedPath)) continue;
    for (const file of await collectFilesAsync(trackedPath, undefined, config.excludePatterns)) uniqueFiles.add(file);
  }
  return [...uniqueFiles];
}

/**
 * True if `file` is matched by `excludePatterns` relative to any of
 * `roots`. Files outside every root are never considered excluded.
 */
export function isExcludedByConfig(file: string, roots: string[], excludePatterns: string[]): boolean {
  if (!excludePatterns.length) return false;
  const excludeMatcher = ignore().add(excludePatterns);
  for (const root of roots) {
    const pathRelativeToRoot = relative(root, file);
    if (!pathRelativeToRoot || pathRelativeToRoot.startsWith("..")) continue;
    if (excludeMatcher.ignores(pathRelativeToRoot)) return true;
  }
  return false;
}
