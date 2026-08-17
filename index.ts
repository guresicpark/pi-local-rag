/**
 * pi-local-rag — Hybrid RAG Pipeline (BM25 + dual-model vectors + Auto-injection)
 *
 * Index local files → chunk → embed → store → retrieve → inject into LLM context.
 * Uses Transformers.js (ONNX) for local embeddings — zero cloud dependency.
 *
 * Two embedding models: code files go through jina-embeddings-v2-base-code,
 * everything else (prose, markup, data/config, PDF/DOCX) through
 * nomic-embed-text-v1.5. Their vectors live in separate sqlite-vec tables
 * and hybrid search queries both spaces, merging per-space-normalized scores.
 *
 * Storage is per-cwd: walk up from the working directory looking for a `.pi/rag/`
 * project store; fall back to `~/.pi/rag/` as the global default. The first
 * `/rag index` in a directory with no parent store creates one at cwd.
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
 * Implementation is split across:
 *   constants.ts     — shared constants, dual model ids, ext groups, size limits
 *   store.ts         — RAG_DIR / LEGACY_DIR / file paths / ensureDir + legacy migration
 *   config.ts        — RagConfig type, loadConfig / saveConfig, ext-group helpers
 *   chunking.ts      — sha256, chunkText, collectFiles, extractText (txt/pdf/docx/html)
 *   embed.ts         — dual ONNX pipelines (nomic + jina-code), embed, embedBatchFor
 *   search.ts        — cosineSimilarity, normalize, hybridSearch (both vector spaces)
 *   indexing.ts      — indexFiles (parallel Phase 1 read, per-model Phase 2 embed)
 *   index.ts         — extension entry point (this file) + re-exports
 */
import type { ExtensionAPI } from "@earendil-works/pi-coding-agent";
import type { AutocompleteItem } from "@earendil-works/pi-tui";
import { Box, Text } from "@earendil-works/pi-tui";
import { Type } from "typebox";
import { existsSync } from "node:fs";
import { resolve, extname, basename, relative, isAbsolute, sep } from "node:path";
import ignore from "ignore";

import { RST, B, D, GREEN, CYAN, type EmbedGroup, CODE_EMBEDDING_MODEL, DEFAULT_CODE_EXTS, EMBEDDING_MODEL } from "./constants.ts";
import { getRagDir, GLOBAL_RAG_DIR, dbFile } from "./store.ts";
import {
  loadConfig, saveConfig, normalizeExt,
  resolveCodeExtensions, resolveDocExtensions, classifyFile,
} from "./config.ts";
import { resetStore, getIndexStats, withDb, getIndexedPaths } from "./db.ts";
import { collectFiles, collectFromTracked, collectFromTrackedAsync, isExcludedByConfig } from "./chunking.ts";
import { hybridSearch } from "./search.ts";
import { indexFiles, isIndexStale } from "./indexing.ts";
import * as repo from "./repository.ts";

// Re-export the public surface so existing consumers of `pi-local-rag` keep
// working (tests, downstream code that imports from the package root).
export {
  DEFAULT_TEXT_EXTS, DEFAULT_CODE_EXTS, DEFAULT_DOC_EXTS, BINARY_DOC_EXTS,
  EMBEDDING_MODEL, CODE_EMBEDDING_MODEL, CODE_EMBED_SCHEME, MAX_LINE_CHARS, MAX_CHUNK_CHARS,
} from "./constants.ts";
export type { EmbedGroup } from "./constants.ts";
export { getRagDir, GLOBAL_RAG_DIR, LEGACY_DIR } from "./store.ts";
export type { RagConfig } from "./config.ts";
export {
  loadConfig, saveConfig, defaultConfig, normalizeExt, resolveExtensions,
  resolveCodeExtensions, resolveDocExtensions, classifyFile,
} from "./config.ts";
export type { Chunk, IndexMeta, IndexStats } from "./db.ts";
export { openDb, getDb, getDbConn, closeDbConn, getFreshDbConn, loadIndex, saveIndex, clearIndex, resetStore, getIndexStats, initSchema, float32ToBuffer } from "./db.ts";
export {
  sha256, chunkText, collectFiles, collectFilesAsync, collectFromTracked, collectFromTrackedAsync,
  isExcludedByConfig, extractText, getOcrTooling, isSparsePdfText,
} from "./chunking.ts";
export { embed, embedBatch, embedQueryFor, embedBatchFor, EMBED_MODELS, buildQueryInput, buildDocumentInput, cleanQuery, resolveModelCacheDir } from "./embed.ts";
export type { ScoredChunk } from "./search.ts";
export { cosineSimilarity, normalize, hybridSearch } from "./search.ts";
export { isIndexStale, indexFiles } from "./indexing.ts";
export type { ProgressCallbacks } from "./indexing.ts";

// ─── Extension ────────────────────────────────────────────────────────────────

/** The slice of ctx.ui the progress renderers touch. */
interface RagUi {
  setStatus: (k: string, v: string | undefined) => void;
  setWidget: (k: string, v: string[] | undefined) => void;
  notify: (m: string, t?: "info" | "error" | "warning") => void;
}

/** Shared progress-bar renderer (24-cell block bar, CYAN filled / dim empty). */
function progressBar(n: number, total: number, width = 24): string {
  const filled = Math.round((n / total) * width);
  return CYAN + "█".repeat(filled) + D + "░".repeat(width - filled) + RST;
}

/** "project" for a cwd-scoped store, "global" for the home-dir fallback. */
function storeScope(ragDir: string): "global" | "project" {
  return ragDir === GLOBAL_RAG_DIR() ? "global" : "project";
}

/** True when `filePath` is `root` itself or nested beneath it. */
function isUnderRoot(filePath: string, root: string): boolean {
  const rel = relative(root, filePath);
  return rel === "" || (rel !== ".." && !rel.startsWith(`..${sep}`) && !isAbsolute(rel));
}

/** cwd-relative path for display (e.g. `src/myfile.php`); falls back to the
 *  full path for files outside cwd so the model still gets a resolvable path. */
function displayPath(filePath: string, cwd: string): string {
  return isUnderRoot(filePath, cwd) ? relative(cwd, filePath) : filePath;
}

/**
 * Shared embed-progress UI for /rag index|rebuild: renders one line
 * per embedding model (code → jina, text → nomic) plus a combined bar, and
 * notifies when a model is about to be downloaded (a cold cache can stall
 * for minutes with no other visual feedback).
 */
function makeEmbedProgress(ctx: { ui: RagUi }, verb: string) {
  const state: Partial<Record<EmbedGroup, { done: number; total: number }>> = {};
  return {
    onEmbed(done: number, total: number, group: EmbedGroup) {
      state[group] = { done, total };
      const groups = (Object.keys(state) as EmbedGroup[]).sort();
      const doneAll = groups.reduce((s, g) => s + state[g]!.done, 0);
      const totalAll = groups.reduce((s, g) => s + state[g]!.total, 0);
      const pct = totalAll ? Math.round((doneAll / totalAll) * 100) : 0;
      const bar = progressBar(doneAll, totalAll);
      ctx.ui.setStatus("rag", `■ ${verb} ${pct}% │ ${doneAll}/${totalAll} chunks`);
      ctx.ui.setWidget("rag", [
        `${B}${CYAN}${verb}${RST}  ${bar}  ${GREEN}${pct}%${RST}`,
        ...groups.map(g =>
          `${D}${g.padEnd(5)} ${RST}${state[g]!.done}/${state[g]!.total}${D}  ${g === "code" ? "jina-code" : "nomic"}${RST}`),
      ]);
    },
    onModelLoad(group: EmbedGroup, model: string) {
      ctx.ui.notify(`⏳ Loading ${group} embedding model: ${model} — first run downloads it (this can take a few minutes)`, "info");
    },
  };
}

/**
 * Shared file/embed/save progress UI for /rag index|rebuild. The
 * two commands differ only in the active verb and the per-file "done"
 * label, so the callback bundle is built once and parameterized.
 */
function makeIndexProgress(ctx: { ui: RagUi }, verb: string, doneLabel: string) {
  return {
    onFile(current: number, total: number, filename: string, skipped: number) {
      const pct = Math.round((current / total) * 100);
      const bar = progressBar(current, total);
      ctx.ui.setStatus("rag", `■ ${verb} ${pct}% │ ${current}/${total} files │ ${skipped} unchanged`);
      ctx.ui.setWidget("rag", [
        `${B}${CYAN}${verb}${RST}  ${bar}  ${GREEN}${pct}%${RST}`,
        `${D}file:    ${RST}${filename}`,
        `${D}done:    ${RST}${GREEN}${current - skipped} ${doneLabel}${RST}  ${D}${skipped} unchanged${RST}`,
      ]);
    },
    ...makeEmbedProgress(ctx, "Embedding"),
    onChunk(ci: number, total: number, filename: string) {
      ctx.ui.setStatus("rag", `■ Embedding ${filename} — chunk ${ci}/${total}`);
    },
    onSave() {
      ctx.ui.setStatus("rag", `■ Saving index...`);
    },
  };
}

export default function (pi: ExtensionAPI) {
  // Render the auto-injected "rag" message as a single summary line in the TUI.
  // The full chunk context in `content` still goes to the model; `details.summary`
  // carries the one-line form shown to the user.
  pi.registerMessageRenderer("rag", (message, { outputPad }, theme) => {
    const summary = (message.details as { summary?: string } | undefined)?.summary;
    if (!summary) return undefined;
    const box = new Box(outputPad, 1, (t) => theme.bg("customMessageBg", t));
    box.addChild(new Text(theme.fg("dim", summary), 0, 0));
    return box;
  });

  // Throttle stale-index checks to once per hour so we don't repeatedly stat
  // the filesystem on every agent turn (matches the upstream fork's
  // lastStaleCheckMs pattern from kallewoof@849e485).
  let lastStaleCheckMs = 0;
  const STALE_CHECK_INTERVAL_MS = 60 * 60 * 1000;

  // ── Auto-inject RAG context before every agent turn ──
  pi.on("before_agent_start", async (event, ctx) => {
    const config = loadConfig();
    if (!config.ragEnabled) return;
    const cwd = ctx.cwd ?? process.cwd();

    return withDb(async (database) => {
      const stats = getIndexStats(database);
      if (stats.totalChunks === 0) return;

      const now = Date.now();
      if (isIndexStale(stats) && now - lastStaleCheckMs > STALE_CHECK_INTERVAL_MS) {
        lastStaleCheckMs = now;
        // Re-walk tracked paths so new files (and files of newly-supported
        // extensions, e.g. PDF/DOCX added in a later version) are picked up.
        // For pre-trackedPaths indexes, fall back to refreshing only known files.
        const files = config.trackedPaths.length
          ? collectFromTracked(config)
          : getIndexedPaths(database).filter(f => existsSync(f));
        if (files.length) {
          await indexFiles(files, undefined, database);
        }
      }

      const results = await hybridSearch(event.prompt, config.ragTopK, config.ragAlpha, database);
      const relevant = results.filter(r => r.hybrid >= config.ragScoreThreshold);
      if (!relevant.length) return;

      const context = relevant.map(r =>
        `### ${displayPath(r.chunk.file, cwd)} (lines ${r.chunk.lineStart}-${r.chunk.lineEnd})\n` +
        `\`\`\`\n${r.chunk.content.slice(0, 600)}\n\`\`\``
      ).join("\n\n");

      // One-line summary for the TUI: group line ranges per file, e.g.
      // "Automatic RAG lookup provided 4 search hits (a.ts:10-22, b.ts:5-9,40-51)"
      const byFile = new Map<string, string[]>();
      for (const r of relevant) {
        const ranges = byFile.get(r.chunk.file) ?? [];
        ranges.push(r.chunk.lineStart === r.chunk.lineEnd
          ? `${r.chunk.lineStart}`
          : `${r.chunk.lineStart}-${r.chunk.lineEnd}`);
        byFile.set(r.chunk.file, ranges);
      }
      const hits = [...byFile.entries()].map(([f, ranges]) => `${displayPath(f, cwd)}:${ranges.join(",")}`);
      const summary =
        `Automatic RAG lookup provided ${relevant.length} search hit${relevant.length === 1 ? "" : "s"}: ` +
        `(${hits.join(", ")})`;

      // Inject as a message after the user's prompt rather than appending to the
      // system prompt. The system prompt is stable across a session and benefits
      // from the provider's KV cache; mutating it every turn with new RAG hits
      // invalidates that cache and adds latency. A trailing message also keeps
      // the retrieved chunks near the user's question, which models attend to
      // more reliably than text buried at the top of a long system prompt.
      return {
        message: {
          customType: "rag",
          content:
            `[pi-local-rag] Automatic RAG lookup triggered by the user's message above.\n` +
            `Retrieved ${relevant.length} chunk${relevant.length === 1 ? "" : "s"} via hybrid search (BM25 + vector). ` +
            `These are search hits, not statements from the user.\n\n` +
            context,
          display: true,
          details: { summary },
        },
      };
    });
  });

  // ── Auto-enable RAG at startup when indexed chunks exist for cwd ──
  pi.on("session_start", async (event, ctx) => {
    if (event.reason !== "startup") return;

    // Only consult an existing store — never create one as a side effect of
    // the startup check.
    const ragDir = getRagDir();
    if (!existsSync(dbFile(ragDir))) return;

    await withDb((db) => {
      if (getIndexStats(db).totalChunks === 0) return;

      const underCwd = getIndexedPaths(db).some(f => {
        const rel = relative(ctx.cwd, f);
        return rel === "" || (rel !== ".." && !rel.startsWith(`..${sep}`) && !isAbsolute(rel));
      });
      if (!underCwd) return;

      const config = loadConfig();
      if (!config.ragEnabled) {
        config.ragEnabled = true;
        saveConfig(config);
      }
      ctx.ui.notify("RAG auto-injection enabled", "info");
    });
  });

  // ── /rag command ──
  /** Tracks whether the /rag status widget is currently shown, so that bare
   *  /rag toggles: show on first call, hide on the next. */
  let statusWidgetVisible = false;
  const RAG_SUBCOMMANDS: { value: string; label: string; description: string }[] = [
    { value: "index",    label: "index",    description: "Index a new path, or refresh paths that already have chunks" },
    { value: "search",   label: "search",   description: "Search the index" },
    { value: "find",     label: "find",     description: "List indexed files matching a glob" },
    { value: "rebuild",  label: "rebuild",  description: "Re-embed tracked files (--force to skip hash check + wipe DB)" },
    { value: "clear",    label: "clear",    description: "Clear the index and reset the store to defaults" },
    { value: "exclude",  label: "exclude",  description: "Manage gitignore-style exclude patterns" },
    { value: "ext",      label: "ext",      description: "Manage indexable file-extension allowlist" },
    { value: "on",       label: "on",       description: "Enable auto-injection" },
    { value: "off",      label: "off",      description: "Disable auto-injection" },
    { value: "help",     label: "help",     description: "Show all /rag commands" },
  ];

  pi.registerCommand("rag", {
    description: "pi-local-rag: /rag (status)|index|search|find|rebuild [--force]|clear|exclude|on|off|ext",
    getArgumentCompletions: (prefix: string): AutocompleteItem[] | null => {
      const filtered = RAG_SUBCOMMANDS
        .filter((s) => s.value.startsWith(prefix))
        .map((s) => ({ value: s.value, label: s.label, description: s.description }));
      return filtered.length > 0 ? filtered : null;
    },
    handler: async (args, ctx) => {
      const parts = (args || "").trim().split(/\s+/);
      const cmd = parts[0] || "";

      // ── index (also refreshes already-indexed tracked paths) ──
      if (cmd === "index") {
        const path = parts[1] || ".";
        if (!existsSync(path)) { ctx.ui.notify(`Path not found: ${path}`, "error"); return; }
        // Anchor a project-local store at cwd if there isn't one in scope yet.
        getRagDir({ createIfMissing: true });
        const config = loadConfig();
        const absPath = resolve(path);
        if (!config.trackedPaths.includes(absPath)) {
          config.trackedPaths.push(absPath);
          saveConfig(config);
        }

        // First time this path is seen there are no chunks under it — do a
        // normal index. Once chunks exist for the tracked path, `/rag index`
        // acts as an incremental refresh: it re-walks every tracked path and
        // only re-embeds new/changed files.
        const alreadyIndexed = await withDb(db =>
          getIndexedPaths(db).some(f => isUnderRoot(f, absPath)),
        );

        let files: string[];
        let verb: string;
        let doneLabel: string;
        if (!alreadyIndexed) {
          files = collectFiles(absPath, undefined, config.excludePatterns);
          if (!files.length) { ctx.ui.notify(`No indexable files found in: ${path}`, "warning"); return; }
          verb = "Indexing";
          doneLabel = "chunked";
          ctx.ui.notify(`Found ${files.length} files to index`, "info");
        } else {
          files = config.trackedPaths.length
            ? collectFromTracked(config)
            : await withDb(db => getIndexedPaths(db).filter(f => existsSync(f)));
          if (!files.length) { ctx.ui.notify("No tracked files to refresh.", "warning"); return; }
          verb = "Refreshing";
          doneLabel = "new/changed";
          ctx.ui.notify(`Refreshing ${files.length} files...`, "info");
        }

        const { result, enabledNow } = await withDb(async (db) => {
          const result = await indexFiles(files, makeIndexProgress(ctx, verb, doneLabel), db);
          // ragEnabled defaults to false; flip it on as soon as the store
          // actually has chunks (mirrors the session_start auto-enable).
          const enabledNow = !config.ragEnabled && getIndexStats(db).totalChunks > 0;
          return { result, enabledNow };
        });

        ctx.ui.setStatus("rag", undefined);
        ctx.ui.setWidget("rag", undefined);

        const secs = (result.durationMs / 1000).toFixed(1);
        const scope = storeScope(getRagDir());
        const summary = alreadyIndexed
          ? `✅ Refreshed ${result.indexed} new/changed · ${result.skipped} unchanged · ${result.chunks} chunks (${result.chunksByGroup.code} code · ${result.chunksByGroup.text} text) · ${secs}s`
          : `✅ Indexed ${result.indexed} files (${result.chunks} chunks: ${result.chunksByGroup.code} code · ${result.chunksByGroup.text} text) · ${result.skipped} unchanged · ${secs}s · tracking ${config.trackedPaths.length} path(s) · ${scope} store`;
        ctx.ui.notify(summary, "info");

        if (enabledNow) {
          config.ragEnabled = true;
          saveConfig(config);
          ctx.ui.notify("RAG auto-injection enabled", "info");
        }
        return;
      }

      // ── search ──
      if (cmd === "search") {
        const query = parts.slice(1).join(" ");
        if (!query) { ctx.ui.notify("Usage: /rag search <query>", "warning"); return; }
        const config = loadConfig();
        const { results, hasVectors } = await withDb(async (db) => {
          const results = await hybridSearch(query, 10, config.ragAlpha, db);
          const stats = getIndexStats(db);
          return { results, hasVectors: stats.embeddedCount > 0 || stats.embeddedCodeCount > 0 };
        });
        if (!results.length) { ctx.ui.notify(`No results for: ${query}`, "warning"); return; }

        const th = ctx.ui.theme;
        const lines: string[] = [
          th.bold(th.fg("accent", "🔍 ") + `${results.length} results for "${query}"`) +
            "  " + th.fg("dim", hasVectors ? "hybrid BM25+vector" : "BM25 only"),
          "",
        ];
        for (const r of results) {
          lines.push(
            th.fg("success", displayPath(r.chunk.file, ctx.cwd ?? process.cwd())) +
            th.fg("muted", `:${r.chunk.lineStart}-${r.chunk.lineEnd}`) +
            "  " + th.fg("dim", `score=${r.hybrid.toFixed(2)}`)
          );
          const preview = r.chunk.content.split("\n").slice(0, 3).join("\n");
          lines.push(th.fg("dim", preview.slice(0, 200)));
          lines.push("");
        }
        ctx.ui.setWidget("rag-search", lines);
        return;
      }

      // ── on/off toggle ──
      if (cmd === "on" || cmd === "off") {
        const config = loadConfig();
        config.ragEnabled = cmd === "on";
        saveConfig(config);
        ctx.ui.notify(cmd === "on" ? "RAG auto-injection enabled" : "RAG auto-injection disabled", "info");
        return;
      }

      // ── rebuild ──
      if (cmd === "rebuild") {
        // Parse --force flag from any position after "rebuild".
        const rebuildArgs = parts.slice(1);
        const force = rebuildArgs.includes("--force");
        const config = loadConfig();

        // Walking tracked paths can stall the event loop on large trees
        // (45k+ files). Use the async variant + yield up-front so the user
        // gets immediate feedback before the heavy work begins.
        ctx.ui.notify("Scanning tracked paths...", "info");
        const trackedFiles = await collectFromTrackedAsync(config);

        try {
          const outcome = await withDb(async (database) => {
            const indexedFileSet = new Set(repo.listFilePaths(database));

            // Union of currently-indexed files and files discovered by walking tracked paths.
            const targetSet = new Set<string>([...trackedFiles]);
            for (const f of indexedFileSet) {
              if (existsSync(f) && !isExcludedByConfig(f, config.trackedPaths, config.excludePatterns)) {
                targetSet.add(f);
              }
            }
            const targetFiles = [...targetSet];

            if (!targetFiles.length && !indexedFileSet.size) return null;

            // Files in the index but no longer present (deleted, excluded, or untracked).
            const droppedFiles = [...indexedFileSet].filter(f => !targetSet.has(f));
            for (const f of droppedFiles) {
              repo.deleteVectorsForFile(database, f);
              repo.deleteCodeVectorsForFile(database, f);
              repo.deleteChunksForFile(database, f);
              repo.deleteFile(database, f);
            }
            if (force) {
              // --force: wipe everything and rebuild the FTS index. indexFiles
              // will then insert fresh rows for every targetFile, bypassing the
              // skip-on-equal-hash check.
              repo.clearAllVectors(database);
            } else {
              for (const f of targetFiles) repo.setFileEmbedded(database, f, false);
            }

            const newFiles = targetFiles.filter(f => !indexedFileSet.has(f));
            ctx.ui.notify(`Rebuilding ${targetFiles.length} files${force ? " (forced)" : ""}...`, "info");
            if (droppedFiles.length) ctx.ui.notify(`Pruned ${droppedFiles.length} files (deleted/excluded)`, "info");
            if (newFiles.length) ctx.ui.notify(`Discovered ${newFiles.length} new files`, "info");

            // Yield so the TUI can paint the "Rebuilding" message before
            // indexFiles starts hammering the event loop.
            await new Promise<void>(r => setTimeout(r, 0));

            const result = await indexFiles(targetFiles, makeIndexProgress(ctx, "Rebuilding", "re-embedded"), database, force);
            return { result, droppedCount: droppedFiles.length };
          });

          if (!outcome) {
            ctx.ui.notify("No files to rebuild. Run /rag index <path> first.", "warning");
            return;
          }

          ctx.ui.setStatus("rag", undefined);
          ctx.ui.setWidget("rag", undefined);

          const secs = (outcome.result.durationMs / 1000).toFixed(1);
          ctx.ui.notify(`✅ Rebuilt: ${outcome.result.indexed} re-indexed · ${outcome.result.skipped} unchanged · ${outcome.droppedCount} deleted · ${outcome.result.chunks} chunks (${outcome.result.chunksByGroup.code} code · ${outcome.result.chunksByGroup.text} text) · ${secs}s`, "info");
        } catch (err) {
          ctx.ui.notify(`Rebuild failed: ${(err as Error).message}`, "error");
        }
        return;
      }

      // ── ext (configure file extensions) ──
      if (cmd === "ext") {
        const sub = (parts[1] || "list").toLowerCase();
        const config = loadConfig();

        if (sub === "list") {
          const th = ctx.ui.theme;
          const code = Array.from(resolveCodeExtensions(config)).sort();
          const docs = Array.from(resolveDocExtensions(config)).sort();
          const lines: string[] = [
            th.bold("Active file extensions") + "  " + th.fg("dim", `(${code.length} code · ${docs.length} text)`),
            th.fg("dim", "  code ") + th.fg("muted", code.join(" ")),
            th.fg("dim", "        ") + th.fg("dim", `→ ${CODE_EMBEDDING_MODEL}`),
            th.fg("dim", "  text ") + th.fg("muted", docs.join(" ")),
            th.fg("dim", "        ") + th.fg("dim", `→ ${EMBEDDING_MODEL}  (+ .pdf .docx)`),
          ];
          if (config.extraExtensions.length || config.extraCodeExtensions.length)
            lines.push("  " + th.fg("dim", "extra:   ") + th.fg("success", [...config.extraExtensions, ...config.extraCodeExtensions].join(" ")));
          if (config.excludeExtensions.length)
            lines.push("  " + th.fg("dim", "excluded:") + " " + th.fg("warning", config.excludeExtensions.join(" ")));
          lines.push("", th.fg("dim", "Edit via /rag ext add <.ext> [code|text] / remove <.ext> / reset"));
          ctx.ui.setWidget("rag-ext", lines);
          return;
        }

        if (sub === "add") {
          const ext = normalizeExt(parts[2] || "");
          if (!ext) { ctx.ui.notify("Usage: /rag ext add <.ext> [code|text]", "warning"); return; }
          const groupArg = (parts[3] || "").toLowerCase();
          const group: EmbedGroup =
            groupArg === "code" || groupArg === "text" ? groupArg
            : DEFAULT_CODE_EXTS.includes(ext) ? "code"
            : "text";
          config.excludeExtensions = config.excludeExtensions.filter(e => normalizeExt(e) !== ext);
          if (group === "code") {
            config.extraExtensions = config.extraExtensions.filter(e => normalizeExt(e) !== ext);
            if (!config.extraCodeExtensions.map(normalizeExt).includes(ext)) config.extraCodeExtensions.push(ext);
          } else {
            config.extraCodeExtensions = config.extraCodeExtensions.filter(e => normalizeExt(e) !== ext);
            if (!config.extraExtensions.map(normalizeExt).includes(ext)) config.extraExtensions.push(ext);
          }
          saveConfig(config);
          const model = group === "code" ? CODE_EMBEDDING_MODEL : EMBEDDING_MODEL;
          ctx.ui.notify(`Added ${ext} to the ${group} group (embedded by ${model}). Run /rag index <path> to pick up matching files.`, "info");
          return;
        }

        if (sub === "remove" || sub === "rm") {
          const ext = normalizeExt(parts[2] || "");
          if (!ext) { ctx.ui.notify("Usage: /rag ext remove <.ext>", "warning"); return; }
          const wasExtra = [...config.extraExtensions, ...config.extraCodeExtensions].map(normalizeExt).includes(ext);
          config.extraExtensions = config.extraExtensions.filter(e => normalizeExt(e) !== ext);
          config.extraCodeExtensions = config.extraCodeExtensions.filter(e => normalizeExt(e) !== ext);
          if (!wasExtra && !config.excludeExtensions.map(normalizeExt).includes(ext)) config.excludeExtensions.push(ext);
          saveConfig(config);
          ctx.ui.notify(`Removed ${ext} from indexable extensions.`, "info");
          return;
        }

        if (sub === "reset") {
          config.extraExtensions = [];
          config.extraCodeExtensions = [];
          config.excludeExtensions = [];
          saveConfig(config);
          ctx.ui.notify("Extension lists reset to defaults.", "info");
          return;
        }

        ctx.ui.notify("Usage: /rag ext list|add <.ext> [code|text]|remove <.ext>|reset", "warning");
        return;
      }

      // ── clear ──
      if (cmd === "clear") {
        const storeDir = resetStore();
        ctx.ui.notify(`✅ Index cleared and store reset to fresh defaults: ${storeDir}`, "info");
        return;
      }

      // ── exclude ──
      if (cmd === "exclude") {
        const config = loadConfig();
        const expr = parts.slice(1).join(" ").trim();
        const th = ctx.ui.theme;

        if (!expr) {
          if (!config.excludePatterns.length) {
            ctx.ui.notify("No exclude patterns set. Add one with: /rag exclude <pattern>", "info");
            return;
          }
          const lines: string[] = [
            th.bold(`Exclude patterns (${config.excludePatterns.length})`),
            "",
          ];
          for (const p of config.excludePatterns) lines.push("  " + th.fg("muted", p));
          ctx.ui.setWidget("rag-exclude", lines);
          return;
        }

        if (expr.startsWith("-")) {
          const target = expr.slice(1);
          const before = config.excludePatterns.length;
          config.excludePatterns = config.excludePatterns.filter(p => p !== target);
          if (config.excludePatterns.length === before) {
            ctx.ui.notify(`Pattern not found: ${target}`, "warning");
            return;
          }
          saveConfig(config);
          ctx.ui.notify(`✅ Removed exclude: ${target} · ${config.excludePatterns.length} pattern(s) remain. Run /rag rebuild to re-apply.`, "info");
          return;
        }

        if (config.excludePatterns.includes(expr)) {
          ctx.ui.notify(`Already excluded: ${expr}`, "warning");
          return;
        }
        config.excludePatterns.push(expr);
        saveConfig(config);
        ctx.ui.notify(`✅ Added exclude: ${expr} · ${config.excludePatterns.length} pattern(s) total. Run /rag rebuild to re-apply.`, "info");
        return;
      }

      // ── find ──
      if (cmd === "find") {
        const glob = parts.slice(1).join(" ").trim();
        if (!glob) {
          ctx.ui.notify("Usage: /rag find <glob>   e.g. *.html, page*, foo.js, src/*.ts", "warning");
          return;
        }

        const indexedPaths = await withDb(db => getIndexedPaths(db));
        const cwd = process.cwd();
        const ig = ignore().add([glob]);

        const matches: string[] = [];
        for (const fp of indexedPaths) {
          const rel = relative(cwd, fp);
          const candidate = rel && !rel.startsWith("..") ? rel : basename(fp);
          if (ig.ignores(candidate)) matches.push(fp);
        }
        matches.sort();

        if (!matches.length) {
          ctx.ui.notify(`No indexed files match: ${glob}`, "warning");
          return;
        }
        const th = ctx.ui.theme;
        const lines: string[] = [
          th.bold(`🔍 ${matches.length} indexed file${matches.length === 1 ? "" : "s"} matching "${glob}"`),
          "",
        ];
        for (const fp of matches) lines.push(th.fg("success", fp));
        ctx.ui.setWidget("rag-find", lines);
        return;
      }

      // ── help ──
      if (cmd === "help") {
        const pad = (s: string, n: number) => s + " ".repeat(Math.max(0, n - s.length));
        const cmds: [string, string][] = [
          ["/rag index <path>",       "Index a new path, or refresh paths that already have chunks"],
          ["/rag search <query>",     "Hybrid BM25 + vector search over the index"],
          ["/rag find <glob>",        "List indexed files matching a glob (e.g. *.ts, src/*)"],
          ["/rag",                    "Show index stats and active configuration (toggle)"],
          ["/rag rebuild [--force]",  "Re-embed tracked files; --force wipes DB and bypasses hash skip"],
          ["/rag clear",              "Wipe the index and reset the store (config + DB) to fresh defaults"],
          ["/rag exclude <pattern>",  "Add a gitignore-style exclude pattern (omit to list; -<pattern> to remove)"],
          ["/rag ext list",            "Show extension groups (code → jina, text → nomic)"],
          ["/rag ext add <.ext> [code|text]", "Add an extension (group defaults by extension)"],
          ["/rag ext remove <.ext>",   "Remove an extension from the active set"],
          ["/rag ext reset",           "Restore default extension groups"],
          ["/rag on",                 "Enable automatic RAG injection before each agent turn"],
          ["/rag off",                "Disable automatic RAG injection"],
          ["/rag help",               "Show this help"],
        ];
        const COL = 36;
        const th = ctx.ui.theme;
        const lines: string[] = [th.bold("pi-local-rag commands"), ""];
        for (const [usage, desc] of cmds) {
          lines.push("  " + th.fg("success", pad(usage, COL)) + "  " + th.fg("dim", desc));
        }
        ctx.ui.setWidget("rag-help", lines);
        return;
      }

      // ── unknown subcommand ──
      if (cmd) {
        ctx.ui.notify(`Unknown /rag command: ${cmd}. Try /rag help`, "error");
        return;
      }

      // ── bare /rag (default, toggles the status widget) ──
      if (statusWidgetVisible) {
        statusWidgetVisible = false;
        ctx.ui.setWidget("rag-status", undefined);
        return;
      }
      const config = loadConfig();
      const { stats, indexedPaths } = await withDb(db => ({
        stats: getIndexStats(db),
        indexedPaths: getIndexedPaths(db),
      }));
      const fileCount = stats.totalFiles;
      const totalTokens = stats.totalTokens;
      const totalVectors = stats.embeddedCount + stats.embeddedCodeCount;
      const vectorCoverage = stats.totalChunks ? Math.round(totalVectors / stats.totalChunks * 100) : 0;

      const th = ctx.ui.theme;
      const label = (k: string) => th.fg("dim", k.padEnd(18));
      const val = (v: string | number) => th.fg("success", String(v));
      const ragDir = getRagDir();
      const scope = storeScope(ragDir);
      const lines: string[] = [
        th.bold("🔍 pi-local-rag"),
        "",
        "  " + label("Files indexed:")  + val(fileCount),
        "  " + label("Chunks:")         + val(stats.totalChunks),
        "  " + label("Vectors:")        + val(`${stats.embeddedCount} text · ${stats.embeddedCodeCount} code`) + "  " + th.fg("dim", `(${vectorCoverage}% coverage)`),
        "  " + label("Total tokens:")   + val(totalTokens.toLocaleString()),
        "  " + label("Text model:")     + th.fg("dim", stats.embeddingModel || "none"),
        "  " + label("Code model:")     + th.fg("dim", stats.codeEmbeddingModel || "none"),
        "  " + label("Last build:")     + (stats.lastBuild || th.fg("dim", "never")),
        "  " + label("Storage:")        + th.fg("dim", `${ragDir} (${scope})`),
        "",
        "  " + label("RAG injection:")  +
          (config.ragEnabled ? th.fg("success", "enabled") : th.fg("warning", "disabled")) +
          th.fg("dim", `  topK=${config.ragTopK}  threshold=${config.ragScoreThreshold}  alpha=${config.ragAlpha}`),
      ];

      if (fileCount) {
        lines.push("", "  " + th.bold("File types:"));
        const byExt: Record<string, number> = {};
        for (const f of indexedPaths) byExt[extname(f)] = (byExt[extname(f)] || 0) + 1;
        for (const [ext, count] of Object.entries(byExt).sort((a, b) => b[1] - a[1]).slice(0, 8)) {
          const group = classifyFile(ext) === "code" ? th.fg("accent", " code") : th.fg("muted", " text");
          lines.push("    " + th.fg("muted", ext) + "  " + th.fg("dim", String(count)) + group);
        }
      }

      lines.push("", "  " + th.bold("Tracked paths:"));
      if (config.trackedPaths.length) {
        for (const p of config.trackedPaths) lines.push("    " + th.fg("muted", p));
      } else {
        lines.push("    " + th.fg("dim", "(none — run /rag index <path> to track)"));
      }

      lines.push("", "  " + th.bold("Exclude patterns:"));
      if (config.excludePatterns.length) {
        for (const p of config.excludePatterns) lines.push("    " + th.fg("muted", p));
      } else {
        lines.push("    " + th.fg("dim", "(none — add with /rag exclude <pattern>)"));
      }

      ctx.ui.setWidget("rag-status", lines);
      statusWidgetVisible = true;
    },
  });

  // ── Tools ──

  pi.registerTool({
    name: "rag_index",
    label: "RAG index",
    description: "Index a file or directory into the local pi-local-rag pipeline. Chunks text files (including PDF and DOCX), generates embeddings, stores for hybrid BM25+vector search.",
    parameters: Type.Object({
      path: Type.String({ description: "File or directory path to index" }),
    }),
    execute: async (_toolCallId, params) => {
      if (!existsSync(params.path)) return { content: [{ type: "text" as const, text: `Path not found: ${params.path}` }], details: undefined };
      // Anchor a project-local store at cwd if there isn't one in scope yet.
      getRagDir({ createIfMissing: true });
      const config = loadConfig();
      const absPath = resolve(params.path);
      if (!config.trackedPaths.includes(absPath)) {
        config.trackedPaths.push(absPath);
        saveConfig(config);
      }
      const files = collectFiles(absPath, undefined, config.excludePatterns);
      if (!files.length) return { content: [{ type: "text" as const, text: `No indexable files found in: ${params.path}` }], details: undefined };
      const { result, enabledNow } = await withDb(async (db) => {
        const result = await indexFiles(files, {}, db);
        // Enable auto-injection now that chunks exist (default is off).
        return { result, enabledNow: !config.ragEnabled && getIndexStats(db).totalChunks > 0 };
      });
      process.stderr.write(`\n`);
      if (enabledNow) {
        config.ragEnabled = true;
        saveConfig(config);
      }
      return { content: [{ type: "text" as const, text: `Indexed ${result.indexed} files (${result.chunks} chunks, embeddings generated). ${result.skipped} unchanged. ${(result.durationMs / 1000).toFixed(1)}s${enabledNow ? " · RAG auto-injection enabled" : ""}` }], details: undefined };
    },
  });

  pi.registerTool({
    name: "rag_query",
    label: "RAG query",
    description: "Search the local pi-local-rag index using hybrid BM25+vector search. Returns relevant chunks with file paths, line numbers, and relevance scores.",
    parameters: Type.Object({
      query: Type.String({ description: "Search query" }),
      limit: Type.Optional(Type.Number({ description: "Max results (default 10)" })),
    }),
    execute: async (_toolCallId, params, _signal, _onUpdate, ctx) => {
      const config = loadConfig();
      const outcome = await withDb(async (db) => {
        if (!getIndexStats(db).totalChunks) return { empty: true as const };
        const results = await hybridSearch(params.query, params.limit ?? 10, config.ragAlpha, db);
        return { empty: false as const, results };
      });
      if (outcome.empty) return { content: [{ type: "text" as const, text: "pi-local-rag index is empty. Run rag_index first." }], details: undefined };
      if (!outcome.results.length) return { content: [{ type: "text" as const, text: `No results for: ${params.query}` }], details: undefined };
      const text = JSON.stringify(outcome.results.map(r => ({
        file: displayPath(r.chunk.file, ctx.cwd ?? process.cwd()),
        lines: `${r.chunk.lineStart}-${r.chunk.lineEnd}`,
        tokens: r.chunk.tokens,
        scores: { bm25: r.bm25.toFixed(3), vector: r.vector.toFixed(3), hybrid: r.hybrid.toFixed(3) },
        preview: r.chunk.content.slice(0, 300),
      })), null, 2);
      return { content: [{ type: "text" as const, text }], details: undefined };
    },
  });

  pi.registerTool({
    name: "rag_status",
    label: "RAG status",
    description: "Show pi-local-rag index statistics: file count, chunk count, vector coverage, embedding model, RAG config.",
    parameters: Type.Object({}),
    execute: async (_toolCallId) => {
      const config = loadConfig();
      const ragDir = getRagDir();
      const stats = await withDb(db => getIndexStats(db));
      const totalVectors = stats.embeddedCount + stats.embeddedCodeCount;
      const text = JSON.stringify({
        files: stats.totalFiles,
        chunks: stats.totalChunks,
        vectorsEmbedded: {
          text: stats.embeddedCount,
          code: stats.embeddedCodeCount,
        },
        vectorCoverage: stats.totalChunks ? `${Math.round(totalVectors / stats.totalChunks * 100)}%` : "0%",
        embeddingModels: {
          text: stats.embeddingModel || "none",
          code: stats.codeEmbeddingModel || "none",
        },
        totalTokens: stats.totalTokens,
        lastBuild: stats.lastBuild || "never",
        ragConfig: config,
        storagePath: ragDir,
        storageScope: storeScope(ragDir),
      }, null, 2);
      return { content: [{ type: "text" as const, text }], details: undefined };
    },
  });
}
