/**
 * The /rag command: status (bare, toggling), index, search, find, rebuild,
 * clear, exclude, ext-group management, on/off, and help.
 */
import { existsSync } from "node:fs";
import { resolve, extname, basename, relative } from "node:path";
import ignore from "ignore";
import {
  CODE_EMBEDDING_MODEL,
  EMBEDDING_MODEL,
  DEFAULT_CODE_EXTS,
  type EmbedGroup,
} from "../constants.ts";
import { getRagDir } from "../store-paths.ts";
import {
  loadConfig,
  saveConfig,
  normalizeExt,
  resolveCodeExtensions,
  resolveDocExtensions,
  classifyFile,
} from "../config.ts";
import { resetStore, getIndexStats, withDb, getIndexedPaths } from "../database.ts";
import {
  collectFiles,
  collectFromTracked,
  collectFromTrackedAsync,
  isExcludedByConfig,
} from "../file-discovery.ts";
import { hybridSearch } from "../search.ts";
import { indexFiles } from "../indexing.ts";
import * as sqlRepository from "../repository.ts";
import { storeScope, displayPath, isUnderRoot } from "./paths.ts";
import { createIndexProgressRenderer, type RagUi } from "./ui.ts";

/** Context the command handlers receive from the Pi extension runtime. */
export interface RagCommandContext {
  cwd?: string;
  ui: RagUi & { theme: Record<"bold" | "fg", (...args: any[]) => string> };
}

/** Subcommand table for autocomplete and the help widget. */
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

/** Autocomplete items for the /rag command, filtered by the typed prefix. */
export function getRagSubcommandCompletions(prefix: string) {
  const matches = RAG_SUBCOMMANDS
    .filter(subcommand => subcommand.value.startsWith(prefix))
    .map(subcommand => ({ value: subcommand.value, label: subcommand.label, description: subcommand.description }));
  return matches.length > 0 ? matches : null;
}

/**
 * Create the /rag handler. Returned as a factory because the bare `/rag`
 * status widget is a toggle — its visibility state belongs to the
 * extension instance.
 */
export function createRagCommandHandler() {
  /** Tracks whether the /rag status widget is currently shown. */
  let statusWidgetVisible = false;

  return async (args: string, ctx: RagCommandContext) => {
    const parts = (args || "").trim().split(/\s+/);
    const subcommand = parts[0] || "";

    // ── index (also refreshes already-indexed tracked paths) ──
    if (subcommand === "index") {
      await handleIndexSubcommand(parts, ctx);
      return;
    }

    // ── search ──
    if (subcommand === "search") {
      await handleSearchSubcommand(parts, ctx);
      return;
    }

    // ── on/off toggle ──
    if (subcommand === "on" || subcommand === "off") {
      const config = loadConfig();
      config.ragEnabled = subcommand === "on";
      saveConfig(config);
      ctx.ui.notify(
        subcommand === "on" ? "RAG auto-injection enabled" : "RAG auto-injection disabled",
        "info",
      );
      return;
    }

    // ── rebuild ──
    if (subcommand === "rebuild") {
      await handleRebuildSubcommand(parts, ctx);
      return;
    }

    // ── ext (configure file extensions) ──
    if (subcommand === "ext") {
      handleExtSubcommand(parts, ctx);
      return;
    }

    // ── clear ──
    if (subcommand === "clear") {
      const storeDir = resetStore();
      ctx.ui.notify(`✅ Index cleared and store reset to fresh defaults: ${storeDir}`, "info");
      return;
    }

    // ── exclude ──
    if (subcommand === "exclude") {
      handleExcludeSubcommand(parts, ctx);
      return;
    }

    // ── find ──
    if (subcommand === "find") {
      await handleFindSubcommand(parts, ctx);
      return;
    }

    // ── help ──
    if (subcommand === "help") {
      handleHelpSubcommand(ctx);
      return;
    }

    // ── unknown subcommand ──
    if (subcommand) {
      ctx.ui.notify(`Unknown /rag command: ${subcommand}. Try /rag help`, "error");
      return;
    }

    // ── bare /rag (default: toggles the status widget) ──
    if (statusWidgetVisible) {
      statusWidgetVisible = false;
      ctx.ui.setWidget("rag-status", undefined);
      return;
    }
    await renderStatusWidget(ctx);
    statusWidgetVisible = true;
  };
}

// ─── /rag index ─────────────────────────────────────────────────────────────

async function handleIndexSubcommand(parts: string[], ctx: RagCommandContext) {
  const pathArgument = parts[1] || ".";
  if (!existsSync(pathArgument)) {
    ctx.ui.notify(`Path not found: ${pathArgument}`, "error");
    return;
  }
  // Anchor a project-local store at cwd if there isn't one in scope yet.
  getRagDir({ createIfMissing: true });
  const config = loadConfig();
  const absolutePath = resolve(pathArgument);
  if (!config.trackedPaths.includes(absolutePath)) {
    config.trackedPaths.push(absolutePath);
    saveConfig(config);
  }

  // First time this path is seen there are no chunks under it — do a
  // normal index. Once chunks exist for the tracked path, `/rag index`
  // acts as an incremental refresh: it re-walks every tracked path and
  // only re-embeds new/changed files.
  const alreadyIndexed = await withDb(database =>
    getIndexedPaths(database).some(filePath => isUnderRoot(filePath, absolutePath)),
  );

  let filesToProcess: string[];
  let verb: string;
  let doneLabel: string;
  if (!alreadyIndexed) {
    filesToProcess = collectFiles(absolutePath, undefined, config.excludePatterns);
    if (!filesToProcess.length) {
      ctx.ui.notify(`No indexable files found in: ${pathArgument}`, "warning");
      return;
    }
    verb = "Indexing";
    doneLabel = "chunked";
    ctx.ui.notify(`Found ${filesToProcess.length} files to index`, "info");
  } else {
    filesToProcess = config.trackedPaths.length
      ? collectFromTracked(config)
      : await withDb(database => getIndexedPaths(database).filter(filePath => existsSync(filePath)));
    if (!filesToProcess.length) {
      ctx.ui.notify("No tracked files to refresh.", "warning");
      return;
    }
    verb = "Refreshing";
    doneLabel = "new/changed";
    ctx.ui.notify(`Refreshing ${filesToProcess.length} files...`, "info");
  }

  const { result, enabledNow } = await withDb(async (database) => {
    const result = await indexFiles(filesToProcess, createIndexProgressRenderer(ctx, verb, doneLabel), database);
    // ragEnabled defaults to false; flip it on as soon as the store
    // actually has chunks (mirrors the session_start auto-enable).
    const enabledNow = !config.ragEnabled && getIndexStats(database).totalChunks > 0;
    return { result, enabledNow };
  });

  ctx.ui.setStatus("rag", undefined);
  ctx.ui.setWidget("rag", undefined);

  const seconds = (result.durationMs / 1000).toFixed(1);
  const scope = storeScope(getRagDir());
  const summary = alreadyIndexed
    ? `✅ Refreshed ${result.indexed} new/changed · ${result.skipped} unchanged · ${result.chunks} chunks (${result.chunksByGroup.code} code · ${result.chunksByGroup.text} text) · ${seconds}s`
    : `✅ Indexed ${result.indexed} files (${result.chunks} chunks: ${result.chunksByGroup.code} code · ${result.chunksByGroup.text} text) · ${result.skipped} unchanged · ${seconds}s · tracking ${config.trackedPaths.length} path(s) · ${scope} store`;
  ctx.ui.notify(summary, "info");

  if (enabledNow) {
    config.ragEnabled = true;
    saveConfig(config);
    ctx.ui.notify("RAG auto-injection enabled", "info");
  }
}

// ─── /rag search ────────────────────────────────────────────────────────────

async function handleSearchSubcommand(parts: string[], ctx: RagCommandContext) {
  const query = parts.slice(1).join(" ");
  if (!query) {
    ctx.ui.notify("Usage: /rag search <query>", "warning");
    return;
  }
  const config = loadConfig();
  const { results, hasVectors } = await withDb(async (database) => {
    const results = await hybridSearch(query, 10, config.ragAlpha, database);
    const stats = getIndexStats(database);
    return { results, hasVectors: stats.embeddedCount > 0 || stats.embeddedCodeCount > 0 };
  });
  if (!results.length) {
    ctx.ui.notify(`No results for: ${query}`, "warning");
    return;
  }

  const theme = ctx.ui.theme;
  const lines: string[] = [
    theme.bold(theme.fg("accent", "🔍 ") + `${results.length} results for "${query}"`) +
      "  " + theme.fg("dim", hasVectors ? "hybrid BM25+vector" : "BM25 only"),
    "",
  ];
  for (const result of results) {
    lines.push(
      theme.fg("success", displayPath(result.chunk.file, ctx.cwd ?? process.cwd())) +
      theme.fg("muted", `:${result.chunk.lineStart}-${result.chunk.lineEnd}`) +
      "  " + theme.fg("dim", `score=${result.hybrid.toFixed(2)}`),
    );
    const preview = result.chunk.content.split("\n").slice(0, 3).join("\n");
    lines.push(theme.fg("dim", preview.slice(0, 200)));
    lines.push("");
  }
  ctx.ui.setWidget("rag-search", lines);
}

// ─── /rag rebuild ───────────────────────────────────────────────────────────

async function handleRebuildSubcommand(parts: string[], ctx: RagCommandContext) {
  // Parse --force flag from any position after "rebuild".
  const rebuildArguments = parts.slice(1);
  const force = rebuildArguments.includes("--force");
  const config = loadConfig();

  // Walking tracked paths can stall the event loop on large trees (45k+
  // files). Use the async variant + yield up-front so the user gets
  // immediate feedback before the heavy work begins.
  ctx.ui.notify("Scanning tracked paths...", "info");
  const trackedFiles = await collectFromTrackedAsync(config);

  try {
    const outcome = await withDb(async (database) => {
      const indexedFileSet = new Set(sqlRepository.listFilePaths(database));

      // Union of currently-indexed files and files discovered by walking
      // tracked paths (skipping deleted/excluded/untracked ones).
      const targetFileSet = new Set<string>(trackedFiles);
      for (const indexedFile of indexedFileSet) {
        if (existsSync(indexedFile) && !isExcludedByConfig(indexedFile, config.trackedPaths, config.excludePatterns)) {
          targetFileSet.add(indexedFile);
        }
      }
      const targetFiles = [...targetFileSet];

      if (!targetFiles.length && !indexedFileSet.size) return null;

      // Files in the index but no longer present (deleted, excluded, or
      // untracked) are pruned outright.
      const droppedFiles = [...indexedFileSet].filter(file => !targetFileSet.has(file));
      for (const droppedFile of droppedFiles) {
        sqlRepository.deleteVectorsForFile(database, droppedFile);
        sqlRepository.deleteCodeVectorsForFile(database, droppedFile);
        sqlRepository.deleteChunksForFile(database, droppedFile);
        sqlRepository.deleteFile(database, droppedFile);
      }
      if (force) {
        // --force: wipe everything and rebuild the FTS index. indexFiles
        // then inserts fresh rows for every targetFile, bypassing the
        // skip-on-equal-hash check.
        sqlRepository.clearAllVectors(database);
      } else {
        for (const targetFile of targetFiles) sqlRepository.setFileEmbedded(database, targetFile, false);
      }

      const newFiles = targetFiles.filter(file => !indexedFileSet.has(file));
      ctx.ui.notify(`Rebuilding ${targetFiles.length} files${force ? " (forced)" : ""}...`, "info");
      if (droppedFiles.length) ctx.ui.notify(`Pruned ${droppedFiles.length} files (deleted/excluded)`, "info");
      if (newFiles.length) ctx.ui.notify(`Discovered ${newFiles.length} new files`, "info");

      // Yield so the TUI can paint the "Rebuilding" message before
      // indexFiles starts hammering the event loop.
      await new Promise<void>(resolve => setTimeout(resolve, 0));

      const result = await indexFiles(
        targetFiles,
        createIndexProgressRenderer(ctx, "Rebuilding", "re-embedded"),
        database,
        force,
      );
      return { result, droppedCount: droppedFiles.length };
    });

    if (!outcome) {
      ctx.ui.notify("No files to rebuild. Run /rag index <path> first.", "warning");
      return;
    }

    ctx.ui.setStatus("rag", undefined);
    ctx.ui.setWidget("rag", undefined);

    const seconds = (outcome.result.durationMs / 1000).toFixed(1);
    ctx.ui.notify(
      `✅ Rebuilt: ${outcome.result.indexed} re-indexed · ${outcome.result.skipped} unchanged · ${outcome.droppedCount} deleted · ${outcome.result.chunks} chunks (${outcome.result.chunksByGroup.code} code · ${outcome.result.chunksByGroup.text} text) · ${seconds}s`,
      "info",
    );
  } catch (error) {
    ctx.ui.notify(`Rebuild failed: ${(error as Error).message}`, "error");
  }
}

// ─── /rag ext ───────────────────────────────────────────────────────────────

function handleExtSubcommand(parts: string[], ctx: RagCommandContext) {
  const extSubcommand = (parts[1] || "list").toLowerCase();
  const config = loadConfig();

  if (extSubcommand === "list") {
    const theme = ctx.ui.theme;
    const codeExtensions = Array.from(resolveCodeExtensions(config)).sort();
    const docExtensions = Array.from(resolveDocExtensions(config)).sort();
    const lines: string[] = [
      theme.bold("Active file extensions") + "  " + theme.fg("dim", `(${codeExtensions.length} code · ${docExtensions.length} text)`),
      theme.fg("dim", "  code ") + theme.fg("muted", codeExtensions.join(" ")),
      theme.fg("dim", "        ") + theme.fg("dim", `→ ${CODE_EMBEDDING_MODEL}`),
      theme.fg("dim", "  text ") + theme.fg("muted", docExtensions.join(" ")),
      theme.fg("dim", "        ") + theme.fg("dim", `→ ${EMBEDDING_MODEL}  (+ .pdf .docx)`),
    ];
    if (config.extraExtensions.length || config.extraCodeExtensions.length) {
      lines.push("  " + theme.fg("dim", "extra:   ") + theme.fg("success", [...config.extraExtensions, ...config.extraCodeExtensions].join(" ")));
    }
    if (config.excludeExtensions.length) {
      lines.push("  " + theme.fg("dim", "excluded:") + " " + theme.fg("warning", config.excludeExtensions.join(" ")));
    }
    lines.push("", theme.fg("dim", "Edit via /rag ext add <.ext> [code|text] / remove <.ext> / reset"));
    ctx.ui.setWidget("rag-ext", lines);
    return;
  }

  if (extSubcommand === "add") {
    const extension = normalizeExt(parts[2] || "");
    if (!extension) {
      ctx.ui.notify("Usage: /rag ext add <.ext> [code|text]", "warning");
      return;
    }
    const groupArgument = (parts[3] || "").toLowerCase();
    const group: EmbedGroup =
      groupArgument === "code" || groupArgument === "text" ? groupArgument
      : DEFAULT_CODE_EXTS.includes(extension) ? "code"
      : "text";
    // Adding an extension removes it from the exclusion list and from the
    // other group's extras, so a group is always unambiguous.
    config.excludeExtensions = config.excludeExtensions.filter(ext => normalizeExt(ext) !== extension);
    if (group === "code") {
      config.extraExtensions = config.extraExtensions.filter(ext => normalizeExt(ext) !== extension);
      if (!config.extraCodeExtensions.map(normalizeExt).includes(extension)) config.extraCodeExtensions.push(extension);
    } else {
      config.extraCodeExtensions = config.extraCodeExtensions.filter(ext => normalizeExt(ext) !== extension);
      if (!config.extraExtensions.map(normalizeExt).includes(extension)) config.extraExtensions.push(extension);
    }
    saveConfig(config);
    const model = group === "code" ? CODE_EMBEDDING_MODEL : EMBEDDING_MODEL;
    ctx.ui.notify(`Added ${extension} to the ${group} group (embedded by ${model}). Run /rag index <path> to pick up matching files.`, "info");
    return;
  }

  if (extSubcommand === "remove" || extSubcommand === "rm") {
    const extension = normalizeExt(parts[2] || "");
    if (!extension) {
      ctx.ui.notify("Usage: /rag ext remove <.ext>", "warning");
      return;
    }
    const wasExtra = [...config.extraExtensions, ...config.extraCodeExtensions].map(normalizeExt).includes(extension);
    config.extraExtensions = config.extraExtensions.filter(ext => normalizeExt(ext) !== extension);
    config.extraCodeExtensions = config.extraCodeExtensions.filter(ext => normalizeExt(ext) !== extension);
    // Removing a *default* extension adds it to the exclusion list so it
    // stays gone; removing a user extra just drops it.
    if (!wasExtra && !config.excludeExtensions.map(normalizeExt).includes(extension)) {
      config.excludeExtensions.push(extension);
    }
    saveConfig(config);
    ctx.ui.notify(`Removed ${extension} from indexable extensions.`, "info");
    return;
  }

  if (extSubcommand === "reset") {
    config.extraExtensions = [];
    config.extraCodeExtensions = [];
    config.excludeExtensions = [];
    saveConfig(config);
    ctx.ui.notify("Extension lists reset to defaults.", "info");
    return;
  }

  ctx.ui.notify("Usage: /rag ext list|add <.ext> [code|text]|remove <.ext>|reset", "warning");
}

// ─── /rag exclude ───────────────────────────────────────────────────────────

function handleExcludeSubcommand(parts: string[], ctx: RagCommandContext) {
  const config = loadConfig();
  const expression = parts.slice(1).join(" ").trim();
  const theme = ctx.ui.theme;

  if (!expression) {
    if (!config.excludePatterns.length) {
      ctx.ui.notify("No exclude patterns set. Add one with: /rag exclude <pattern>", "info");
      return;
    }
    const lines: string[] = [
      theme.bold(`Exclude patterns (${config.excludePatterns.length})`),
      "",
    ];
    for (const pattern of config.excludePatterns) lines.push("  " + theme.fg("muted", pattern));
    ctx.ui.setWidget("rag-exclude", lines);
    return;
  }

  // "-<pattern>" removes an existing pattern.
  if (expression.startsWith("-")) {
    const target = expression.slice(1);
    const countBefore = config.excludePatterns.length;
    config.excludePatterns = config.excludePatterns.filter(pattern => pattern !== target);
    if (config.excludePatterns.length === countBefore) {
      ctx.ui.notify(`Pattern not found: ${target}`, "warning");
      return;
    }
    saveConfig(config);
    ctx.ui.notify(`✅ Removed exclude: ${target} · ${config.excludePatterns.length} pattern(s) remain. Run /rag rebuild to re-apply.`, "info");
    return;
  }

  if (config.excludePatterns.includes(expression)) {
    ctx.ui.notify(`Already excluded: ${expression}`, "warning");
    return;
  }
  config.excludePatterns.push(expression);
  saveConfig(config);
  ctx.ui.notify(`✅ Added exclude: ${expression} · ${config.excludePatterns.length} pattern(s) total. Run /rag rebuild to re-apply.`, "info");
}

// ─── /rag find ──────────────────────────────────────────────────────────────

async function handleFindSubcommand(parts: string[], ctx: RagCommandContext) {
  const glob = parts.slice(1).join(" ").trim();
  if (!glob) {
    ctx.ui.notify("Usage: /rag find <glob>   e.g. *.html, page*, foo.js, src/*.ts", "warning");
    return;
  }

  const indexedPaths = await withDb(database => getIndexedPaths(database));
  const cwd = process.cwd();
  const globMatcher = ignore().add([glob]);

  const matches: string[] = [];
  for (const indexedPath of indexedPaths) {
    const pathRelativeToCwd = relative(cwd, indexedPath);
    const matchCandidate = pathRelativeToCwd && !pathRelativeToCwd.startsWith("..")
      ? pathRelativeToCwd
      : basename(indexedPath);
    if (globMatcher.ignores(matchCandidate)) matches.push(indexedPath);
  }
  matches.sort();

  if (!matches.length) {
    ctx.ui.notify(`No indexed files match: ${glob}`, "warning");
    return;
  }
  const theme = ctx.ui.theme;
  const lines: string[] = [
    theme.bold(`🔍 ${matches.length} indexed file${matches.length === 1 ? "" : "s"} matching "${glob}"`),
    "",
  ];
  for (const matchedPath of matches) lines.push(theme.fg("success", matchedPath));
  ctx.ui.setWidget("rag-find", lines);
}

// ─── /rag help ──────────────────────────────────────────────────────────────

function handleHelpSubcommand(ctx: RagCommandContext) {
  const padRight = (text: string, width: number) => text + " ".repeat(Math.max(0, width - text.length));
  const commandDescriptions: [string, string][] = [
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
  const USAGE_COLUMN_WIDTH = 36;
  const theme = ctx.ui.theme;
  const lines: string[] = [theme.bold("pi-local-rag commands"), ""];
  for (const [usage, description] of commandDescriptions) {
    lines.push("  " + theme.fg("success", padRight(usage, USAGE_COLUMN_WIDTH)) + "  " + theme.fg("dim", description));
  }
  ctx.ui.setWidget("rag-help", lines);
}

// ─── bare /rag status widget ────────────────────────────────────────────────

async function renderStatusWidget(ctx: RagCommandContext) {
  const config = loadConfig();
  const { stats, indexedPaths } = await withDb(database => ({
    stats: getIndexStats(database),
    indexedPaths: getIndexedPaths(database),
  }));
  const totalVectors = stats.embeddedCount + stats.embeddedCodeCount;
  const vectorCoverage = stats.totalChunks ? Math.round((totalVectors / stats.totalChunks) * 100) : 0;

  const theme = ctx.ui.theme;
  const label = (text: string) => theme.fg("dim", text.padEnd(18));
  const value = (text: string | number) => theme.fg("success", String(text));
  const ragDir = getRagDir();
  const scope = storeScope(ragDir);
  const lines: string[] = [
    theme.bold("🔍 pi-local-rag"),
    "",
    "  " + label("Files indexed:")  + value(stats.totalFiles),
    "  " + label("Chunks:")         + value(stats.totalChunks),
    "  " + label("Vectors:")        + value(`${stats.embeddedCount} text · ${stats.embeddedCodeCount} code`) + "  " + theme.fg("dim", `(${vectorCoverage}% coverage)`),
    "  " + label("Total tokens:")   + value(stats.totalTokens.toLocaleString()),
    "  " + label("Text model:")     + theme.fg("dim", stats.embeddingModel || "none"),
    "  " + label("Code model:")     + theme.fg("dim", stats.codeEmbeddingModel || "none"),
    "  " + label("Last build:")     + (stats.lastBuild || theme.fg("dim", "never")),
    "  " + label("Storage:")        + theme.fg("dim", `${ragDir} (${scope})`),
    "",
    "  " + label("RAG injection:")  +
      (config.ragEnabled ? theme.fg("success", "enabled") : theme.fg("warning", "disabled")) +
      theme.fg("dim", `  topK=${config.ragTopK}  threshold=${config.ragScoreThreshold}  alpha=${config.ragAlpha}`),
  ];

  if (stats.totalFiles) {
    lines.push("", "  " + theme.bold("File types:"));
    const fileCountByExtension: Record<string, number> = {};
    for (const indexedPath of indexedPaths) {
      const extension = extname(indexedPath);
      fileCountByExtension[extension] = (fileCountByExtension[extension] || 0) + 1;
    }
    for (const [extension, count] of Object.entries(fileCountByExtension).sort((a, b) => b[1] - a[1]).slice(0, 8)) {
      const groupLabel = classifyFile(extension) === "code" ? theme.fg("accent", " code") : theme.fg("muted", " text");
      lines.push("    " + theme.fg("muted", extension) + "  " + theme.fg("dim", String(count)) + groupLabel);
    }
  }

  lines.push("", "  " + theme.bold("Tracked paths:"));
  if (config.trackedPaths.length) {
    for (const trackedPath of config.trackedPaths) lines.push("    " + theme.fg("muted", trackedPath));
  } else {
    lines.push("    " + theme.fg("dim", "(none — run /rag index <path> to track)"));
  }

  lines.push("", "  " + theme.bold("Exclude patterns:"));
  if (config.excludePatterns.length) {
    for (const pattern of config.excludePatterns) lines.push("    " + theme.fg("muted", pattern));
  } else {
    lines.push("    " + theme.fg("dim", "(none — add with /rag exclude <pattern>)"));
  }

  ctx.ui.setWidget("rag-status", lines);
}
