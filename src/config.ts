/**
 * User configuration: persistence (config.json in the active store) and the
 * extension-allowlist arithmetic that combines defaults with user adds and
 * removals.
 */
import { existsSync, readFileSync, writeFileSync } from "node:fs";
import { extname } from "node:path";
import { configFilePath, getRagDir } from "./store-paths.ts";
import {
  DEFAULT_TEXT_EXTS,
  DEFAULT_CODE_EXTS,
  DEFAULT_DOC_EXTS,
  type EmbedGroup,
} from "./constants.ts";

/** Full persisted configuration for the RAG store. */
export interface RagConfig {
  /** Whether retrieved chunks are auto-injected before each agent turn. */
  ragEnabled: boolean;
  /** Maximum number of chunks injected per turn. */
  ragTopK: number;
  /** Minimum hybrid score a chunk needs to be injected. */
  ragScoreThreshold: number;
  /** Hybrid blend weight: 0 = pure vector, 1 = pure BM25. */
  ragAlpha: number;
  /** User-added text-group extensions (embedded by the text model). */
  extraExtensions: string[];
  /** User-added code-group extensions (embedded by the code model). */
  extraCodeExtensions: string[];
  /** Extensions removed from both default sets. */
  excludeExtensions: string[];
  /** Absolute paths previously passed to /rag index. */
  trackedPaths: string[];
  /** gitignore-style path patterns excluded from indexing. */
  excludePatterns: string[];
}

/**
 * Structural subset of RagConfig accepted by the extension helpers, so
 * callers can pass partial literals (tests, hand-built configs) without
 * casting.
 */
export interface ExtConfig {
  extraExtensions?: string[];
  extraCodeExtensions?: string[];
  excludeExtensions?: string[];
}

/** Fresh configuration with every field at its documented default. */
export function defaultConfig(): RagConfig {
  return {
    ragEnabled: false,
    ragTopK: 5,
    ragScoreThreshold: 0.1,
    ragAlpha: 0.4,
    extraExtensions: [],
    extraCodeExtensions: [],
    excludeExtensions: [],
    trackedPaths: [],
    excludePatterns: [],
  };
}

/**
 * Load the config for the active store directory. Returns defaults when no
 * config file exists or its JSON is malformed — a broken config must never
 * take the whole indexer down.
 */
export function loadConfig(): RagConfig {
  const filePath = configFilePath(getRagDir());
  if (!existsSync(filePath)) return defaultConfig();
  try {
    return { ...defaultConfig(), ...JSON.parse(readFileSync(filePath, "utf-8")) };
  } catch {
    return defaultConfig();
  }
}

/** Persist the config to the active store directory (pretty-printed JSON). */
export function saveConfig(config: RagConfig): void {
  writeFileSync(configFilePath(getRagDir()), JSON.stringify(config, null, 2));
}

/**
 * Normalize a user-supplied extension to lowercase ".ext" form. Returns ""
 * for blank input so callers can skip empty entries.
 */
export function normalizeExt(extension: string): string {
  const trimmed = extension.trim().toLowerCase();
  if (!trimmed) return "";
  return trimmed.startsWith(".") ? trimmed : `.${trimmed}`;
}

/**
 * Effective code-group allowlist: code defaults + user code extras −
 * exclusions. Extensions in this set are embedded by the code model.
 */
export function resolveCodeExtensions(config: ExtConfig): Set<string> {
  const codeExtensions = new Set(DEFAULT_CODE_EXTS);
  for (const extension of config.extraCodeExtensions ?? []) {
    const normalized = normalizeExt(extension);
    if (normalized) codeExtensions.add(normalized);
  }
  for (const extension of config.excludeExtensions ?? []) {
    const normalized = normalizeExt(extension);
    if (normalized) codeExtensions.delete(normalized);
  }
  return codeExtensions;
}

/**
 * Effective text-group allowlist: text defaults + generic user extras −
 * exclusions. Generic extras land here too, except those that are code
 * defaults (they route to the code group instead).
 */
export function resolveDocExtensions(config: ExtConfig): Set<string> {
  const codeDefaults = new Set(DEFAULT_CODE_EXTS);
  const docExtensions = new Set(DEFAULT_DOC_EXTS);
  for (const extension of config.extraExtensions ?? []) {
    const normalized = normalizeExt(extension);
    if (normalized && !codeDefaults.has(normalized)) docExtensions.add(normalized);
  }
  for (const extension of config.excludeExtensions ?? []) {
    const normalized = normalizeExt(extension);
    if (normalized) docExtensions.delete(normalized);
  }
  return docExtensions;
}

/**
 * Effective overall extension allowlist (both groups combined) from
 * defaults + user config. Answers "is this file indexable at all".
 */
export function resolveExtensions(config: ExtConfig): Set<string> {
  const allowed = new Set(DEFAULT_TEXT_EXTS);
  for (const extension of [...(config.extraExtensions ?? []), ...(config.extraCodeExtensions ?? [])]) {
    const normalized = normalizeExt(extension);
    if (normalized) allowed.add(normalized);
  }
  for (const extension of config.excludeExtensions ?? []) {
    const normalized = normalizeExt(extension);
    if (normalized) allowed.delete(normalized);
  }
  return allowed;
}

/**
 * Which embedding group a file belongs to, based on its extension and the
 * effective allowlists. Binary docs (PDF/DOCX) and unknown extensions fall
 * back to the text group.
 */
export function classifyFile(filePath: string, config?: ExtConfig): EmbedGroup {
  const extension = normalizeExt(extname(filePath));
  if (!extension) return "text";
  const codeExtensions = config ? resolveCodeExtensions(config) : new Set(DEFAULT_CODE_EXTS);
  return codeExtensions.has(extension) ? "code" : "text";
}
