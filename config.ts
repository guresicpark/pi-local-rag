import { existsSync, readFileSync, writeFileSync } from "node:fs";
import { extname } from "node:path";
import { configFile, getRagDir } from "./store.ts";
import { DEFAULT_TEXT_EXTS, DEFAULT_CODE_EXTS, DEFAULT_DOC_EXTS, type EmbedGroup } from "./constants.ts";

export interface RagConfig {
  ragEnabled: boolean;
  ragTopK: number;
  ragScoreThreshold: number;
  ragAlpha: number; // 0 = pure vector, 1 = pure BM25
  extraExtensions: string[];   // user-added text-group extensions (embedded by the text model)
  extraCodeExtensions: string[]; // user-added code-group extensions (embedded by the code model)
  excludeExtensions: string[]; // extensions to drop from both default sets
  trackedPaths: string[];      // absolute paths previously passed to /rag index
  excludePatterns: string[];   // gitignore-style path patterns
}

/** Structural subset of RagConfig accepted by the ext helpers, so callers
 *  can pass partial literals (tests, hand-built configs) without casting. */
export interface ExtConfig {
  extraExtensions?: string[];
  extraCodeExtensions?: string[];
  excludeExtensions?: string[];
}

export function defaultConfig(): RagConfig {
  return {
    ragEnabled: true, ragTopK: 5, ragScoreThreshold: 0.1, ragAlpha: 0.4,
    extraExtensions: [], extraCodeExtensions: [], excludeExtensions: [],
    trackedPaths: [], excludePatterns: [],
  };
}

export function loadConfig(): RagConfig {
  const cfgFile = configFile(getRagDir());
  if (!existsSync(cfgFile)) return defaultConfig();
  try {
    return { ...defaultConfig(), ...JSON.parse(readFileSync(cfgFile, "utf-8")) };
  } catch { return defaultConfig(); }
}

export function saveConfig(config: RagConfig) {
  writeFileSync(configFile(getRagDir()), JSON.stringify(config, null, 2));
}

/** Normalize a user-supplied extension to lowercase ".ext" form. */
export function normalizeExt(ext: string): string {
  const trimmed = ext.trim().toLowerCase();
  if (!trimmed) return "";
  return trimmed.startsWith(".") ? trimmed : `.${trimmed}`;
}

/** Effective code-group allowlist: defaults + user code extras − excludes. */
export function resolveCodeExtensions(config: ExtConfig): Set<string> {
  const set = new Set(DEFAULT_CODE_EXTS);
  for (const e of config.extraCodeExtensions ?? []) {
    const n = normalizeExt(e);
    if (n) set.add(n);
  }
  for (const e of config.excludeExtensions ?? []) {
    const n = normalizeExt(e);
    if (n) set.delete(n);
  }
  return set;
}

/** Effective text-group allowlist. Generic extras land here too, except
 *  those that belong to the code defaults (they route to the code group). */
export function resolveDocExtensions(config: ExtConfig): Set<string> {
  const codeDefaults = new Set(DEFAULT_CODE_EXTS);
  const set = new Set(DEFAULT_DOC_EXTS);
  for (const e of config.extraExtensions ?? []) {
    const n = normalizeExt(e);
    if (n && !codeDefaults.has(n)) set.add(n);
  }
  for (const e of config.excludeExtensions ?? []) {
    const n = normalizeExt(e);
    if (n) set.delete(n);
  }
  return set;
}

/** Build the effective extension allowlist (both groups) from defaults + user config. */
export function resolveExtensions(config: ExtConfig): Set<string> {
  const set = new Set(DEFAULT_TEXT_EXTS);
  for (const e of [...(config.extraExtensions ?? []), ...(config.extraCodeExtensions ?? [])]) {
    const n = normalizeExt(e);
    if (n) set.add(n);
  }
  for (const e of config.excludeExtensions ?? []) {
    const n = normalizeExt(e);
    if (n) set.delete(n);
  }
  return set;
}

/** Which embedding model group a file belongs to, based on its extension.
 *  Binary docs (PDF/DOCX) and unknown extensions → text group. */
export function classifyFile(fp: string, config?: ExtConfig): EmbedGroup {
  const ext = normalizeExt(extname(fp));
  if (!ext) return "text";
  const codeExts = config ? resolveCodeExtensions(config) : new Set(DEFAULT_CODE_EXTS);
  return codeExts.has(ext) ? "code" : "text";
}
