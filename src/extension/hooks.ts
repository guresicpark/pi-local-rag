/**
 * Lifecycle hooks: auto-injection of retrieved chunks before every agent
 * turn, and the startup auto-enable when an existing index matches the
 * session cwd.
 */
import { existsSync } from "node:fs";
import { relative, sep, isAbsolute } from "node:path";
import { loadConfig, saveConfig } from "../config.ts";
import { getRagDir, databaseFilePath } from "../store-paths.ts";
import { withDb, getIndexStats, getIndexedPaths } from "../database.ts";
import { collectFromTracked } from "../file-discovery.ts";
import { hybridSearch } from "../search.ts";
import { indexFiles, isIndexStale } from "../indexing.ts";
import { displayPath } from "./paths.ts";

/** How long a chunk's injected preview is truncated to. */
const INJECTED_CHUNK_PREVIEW_CHARS = 600;

/**
 * Create the before_agent_start handler (one per extension instance so the
 * stale-check throttle state is per-session). Returns the handler.
 */
export function createBeforeAgentStartHandler() {
  // Throttle stale-index checks to once per hour so we don't repeatedly
  // stat the filesystem on every agent turn.
  let lastStaleCheckTimestampMs = 0;
  const STALE_CHECK_INTERVAL_MS = 60 * 60 * 1000;

  return async (event: { prompt: string }, ctx: { cwd?: string }) => {
    const config = loadConfig();
    if (!config.ragEnabled) return;
    const cwd = ctx.cwd ?? process.cwd();

    return withDb(async (database) => {
      const stats = getIndexStats(database);
      if (stats.totalChunks === 0) return;

      const now = Date.now();
      if (isIndexStale(stats) && now - lastStaleCheckTimestampMs > STALE_CHECK_INTERVAL_MS) {
        lastStaleCheckTimestampMs = now;
        // Re-walk tracked paths so new files (and files of newly-supported
        // extensions, e.g. PDF/DOCX added in a later version) are picked
        // up. For pre-trackedPaths indexes, fall back to refreshing only
        // known files.
        const files = config.trackedPaths.length
          ? collectFromTracked(config)
          : getIndexedPaths(database).filter(filePath => existsSync(filePath));
        if (files.length) {
          await indexFiles(files, undefined, database);
        }
      }

      const results = await hybridSearch(event.prompt, config.ragTopK, config.ragAlpha, database);
      const relevantResults = results.filter(result => result.hybrid >= config.ragScoreThreshold);
      if (!relevantResults.length) return;

      const contextBlock = relevantResults.map(result =>
        `### ${displayPath(result.chunk.file, cwd)} (lines ${result.chunk.lineStart}-${result.chunk.lineEnd})\n` +
        "```\n" +
        `${result.chunk.content.slice(0, INJECTED_CHUNK_PREVIEW_CHARS)}\n` +
        "```",
      ).join("\n\n");

      // One-line summary for the TUI: group line ranges per file, e.g.
      // "Automatic RAG lookup provided 4 search hits (a.ts:10-22, b.ts:5-9,40-51)"
      const lineRangesByFile = new Map<string, string[]>();
      for (const result of relevantResults) {
        const ranges = lineRangesByFile.get(result.chunk.file) ?? [];
        ranges.push(result.chunk.lineStart === result.chunk.lineEnd
          ? `${result.chunk.lineStart}`
          : `${result.chunk.lineStart}-${result.chunk.lineEnd}`);
        lineRangesByFile.set(result.chunk.file, ranges);
      }
      const fileSummaries = [...lineRangesByFile.entries()]
        .map(([file, ranges]) => `${displayPath(file, cwd)}:${ranges.join(",")}`);
      const summary =
        `Automatic RAG lookup provided ${relevantResults.length} search hit${relevantResults.length === 1 ? "" : "s"}: ` +
        `(${fileSummaries.join(", ")})`;

      // Inject as a message after the user's prompt rather than appending
      // to the system prompt. The system prompt is stable across a session
      // and benefits from the provider's KV cache; mutating it every turn
      // with new RAG hits invalidates that cache and adds latency. A
      // trailing message also keeps the retrieved chunks near the user's
      // question, which models attend to more reliably than text buried at
      // the top of a long system prompt.
      return {
        message: {
          customType: "rag",
          content:
            `[pi-local-rag] Automatic RAG lookup triggered by the user's message above.\n` +
            `Retrieved ${relevantResults.length} chunk${relevantResults.length === 1 ? "" : "s"} via hybrid search (BM25 + vector). ` +
            `These are search hits, not statements from the user.\n\n` +
            contextBlock,
          display: true,
          details: { summary },
        },
      };
    });
  };
}

/**
 * Create the session_start handler: at startup, if the store in scope for
 * the session cwd has chunks indexed under that cwd, flip ragEnabled on
 * and notify the user.
 */
export function createSessionStartHandler() {
  return async (event: { reason?: string }, ctx: { cwd: string; ui: { notify: (message: string, type?: "info") => void } }) => {
    if (event.reason !== "startup") return;

    // Only consult an existing store — never create one as a side effect of
    // the startup check.
    const ragDir = getRagDir();
    if (!existsSync(databaseFilePath(ragDir))) return;

    await withDb((database) => {
      if (getIndexStats(database).totalChunks === 0) return;

      const anyFileUnderCwd = getIndexedPaths(database).some(filePath => {
        const pathRelativeToCwd = relative(ctx.cwd, filePath);
        return pathRelativeToCwd === ""
          || (pathRelativeToCwd !== ".." && !pathRelativeToCwd.startsWith(`..${sep}`) && !isAbsolute(pathRelativeToCwd));
      });
      if (!anyFileUnderCwd) return;

      const config = loadConfig();
      if (!config.ragEnabled) {
        config.ragEnabled = true;
        saveConfig(config);
      }
      ctx.ui.notify("RAG auto-injection enabled", "info");
    });
  };
}
