/**
 * pi-local-rag — Hybrid RAG Pipeline (BM25 + dual-model vectors + auto-injection)
 *
 * Index local files → chunk → embed → store → retrieve → inject into LLM
 * context. Uses Transformers.js (ONNX) for local embeddings — zero cloud
 * dependency.
 *
 * Two embedding models: code files go through jina-embeddings-v2-base-code,
 * everything else (prose, markup, data/config, PDF/DOCX) through
 * nomic-embed-text-v1.5. Their vectors live in separate sqlite-vec tables
 * and hybrid search queries both spaces, merging per-space-normalized
 * scores.
 *
 * Storage is per-cwd: walk up from the working directory looking for a
 * `.pi/rag/` project store; fall back to `~/.pi/rag/` as the global
 * default. The first `/rag index` in a directory with no parent store
 * creates one at cwd.
 *
 * /rag                  → show index stats (toggle)
 * /rag index <path>     → index a new path; refresh paths that already have chunks
 * /rag search <query>   → hybrid search (BM25 + vector)
 * /rag find <glob>      → list indexed files matching a glob
 * /rag rebuild          → re-embed all tracked files (forced re-embed)
 * /rag clear            → clear index + reset the store to fresh defaults
 * /rag exclude <pat>    → add gitignore-style pattern (use -<pat> to remove; omit arg to list)
 * /rag on|off           → toggle auto-injection
 * /rag ext list         → list extension groups (code / text)
 * /rag ext add <.ext> [code|text] → add an extra extension to a group
 * /rag ext remove <.ext>→ remove an extension from the active set
 * /rag ext reset        → restore default extensions
 * /rag help             → show all /rag commands
 *
 * Tools: rag_index, rag_query, rag_status
 *
 * Implementation is split across (see src/):
 *   src/constants.ts        — shared constants, dual model ids, ext groups, size limits
 *   src/runtime-utils.ts    — event-loop yielding + stderr progress lines
 *   src/store-paths.ts      — store-dir resolution, well-known paths, legacy migration
 *   src/config.ts           — RagConfig, loadConfig / saveConfig, ext-group helpers
 *   src/hashing.ts          — sha256 for text + streamed binary hashing
 *   src/repository.ts       — every SQL statement, schema + model migrations
 *   src/database.ts         — node:sqlite connection lifecycle, stats, reset
 *   src/chunking.ts         — allocation-conscious line-based chunkText
 *   src/file-discovery.ts   — sync/async walkers, tracked paths, exclusion matching
 *   src/text-extraction.ts  — extractText (txt/pdf/docx/html) + OCR fallback
 *   src/embedding.ts        — dual ONNX pipelines (nomic + jina-code)
 *   src/search.ts           — cosineSimilarity, normalize, hybridSearch, splitResultQuotas
 *   src/indexing.ts         — indexFiles (parallel reads, per-model embed batches)
 *   src/extension/*         — Pi extension wiring (UI, hooks, /rag command, tools)
 *   index.ts                — extension entry point (this file) + re-exports
 */
import type { ExtensionAPI } from "@earendil-works/pi-coding-agent";
import type { AutocompleteItem } from "@earendil-works/pi-tui";
import { Box, Text } from "@earendil-works/pi-tui";

import { createBeforeAgentStartHandler, createSessionStartHandler } from "./src/extension/hooks.ts";
import { createRagCommandHandler, getRagSubcommandCompletions } from "./src/extension/rag-command.ts";
import { registerRagTools } from "./src/extension/rag-tools.ts";

// Re-export the public surface so consumers of `pi-local-rag` (tests,
// downstream code importing from the package root) keep working.
export {
  DEFAULT_TEXT_EXTS, DEFAULT_CODE_EXTS, DEFAULT_DOC_EXTS, BINARY_DOC_EXTS,
  EMBEDDING_MODEL, CODE_EMBEDDING_MODEL, CODE_EMBED_SCHEME, MAX_LINE_CHARS, MAX_CHUNK_CHARS,
} from "./src/constants.ts";
export type { EmbedGroup } from "./src/constants.ts";
export { getRagDir, GLOBAL_RAG_DIR, LEGACY_DIR } from "./src/store-paths.ts";
export type { RagConfig } from "./src/config.ts";
export {
  loadConfig, saveConfig, defaultConfig, normalizeExt, resolveExtensions,
  resolveCodeExtensions, resolveDocExtensions, classifyFile,
} from "./src/config.ts";
export type { Chunk, IndexMeta, IndexStats } from "./src/database.ts";
export {
  getDbConn, closeDbConn, getFreshDbConn, loadIndex, saveIndex, clearIndex,
  resetStore, getIndexStats, initSchema,
} from "./src/database.ts";
export { sha256 } from "./src/hashing.ts";
export { chunkText } from "./src/chunking.ts";
export {
  collectFiles, collectFilesAsync, collectFromTracked, collectFromTrackedAsync,
  isExcludedByConfig,
} from "./src/file-discovery.ts";
export { extractText, getOcrTooling, isSparsePdfText } from "./src/text-extraction.ts";
export {
  embed, embedQueryFor, embedBatchFor, buildQueryInput, buildDocumentInput,
  cleanQuery, resolveModelCacheDir,
} from "./src/embedding.ts";
export type { ScoredChunk } from "./src/search.ts";
export { cosineSimilarity, normalize, hybridSearch, splitResultQuotas } from "./src/search.ts";
export { isIndexStale, indexFiles } from "./src/indexing.ts";
export type { ProgressCallbacks } from "./src/indexing.ts";

// ─── Extension entry point ──────────────────────────────────────────────────

export default function piLocalRagExtension(pi: ExtensionAPI) {
  // Render the auto-injected "rag" message as a single summary line in the
  // TUI. The full chunk context in `content` still goes to the model;
  // `details.summary` carries the one-line form shown to the user.
  pi.registerMessageRenderer("rag", (message, { outputPad }, theme) => {
    const details = (message.details as { summary?: string; error?: boolean } | undefined);
    const summary = details?.summary;
    if (!summary) return undefined;
    const isError = details?.error === true;
    // Success → green band; error → red band (the same bgs the TUI uses for
    // succeeded/failed tools). Foreground stays the theme's base text color
    // so it contrasts with the tinted band; the label "RAG lookup" gets the
    // matching success/error fg for reinforcement.
    const box = new Box(outputPad, 1, (boxTheme) =>
      theme.bg(isError ? "toolErrorBg" : "toolSuccessBg", boxTheme));
    const [label, ...rest] = summary.split("—");
    const rendered = rest.length
      ? `${theme.fg(isError ? "error" : "success", label.trim())} —${theme.fg("text", rest.join("—"))}`
      : theme.fg("text", summary);
    box.addChild(new Text(rendered, 0, 0));
    return box;
  });

  // Auto-inject RAG context before every agent turn.
  pi.on("before_agent_start", createBeforeAgentStartHandler());

  // Auto-enable RAG at startup when indexed chunks exist for the cwd.
  pi.on("session_start", createSessionStartHandler());

  // The /rag command.
  pi.registerCommand("rag", {
    description: "pi-local-rag: /rag (status)|index|search|find|rebuild [--force]|clear|exclude|on|off|ext",
    getArgumentCompletions: (prefix: string): AutocompleteItem[] | null =>
      getRagSubcommandCompletions(prefix),
    handler: createRagCommandHandler(),
  });

  // Tools: rag_index, rag_query, rag_status.
  registerRagTools(pi);
}
