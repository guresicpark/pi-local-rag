/**
 * Model-facing tools: rag_index, rag_query, rag_status, rag_kb_read. Thin
 * wrappers around the same core functions the /rag commands use, returning
 * plain-text results for the model.
 */
import { existsSync } from "node:fs";
import { resolve, extname } from "node:path";
import { Type } from "typebox";
import type { ExtensionAPI } from "@earendil-works/pi-coding-agent";
import { getRagDir } from "../store-paths.ts";
import { loadConfig, saveConfig, resolveExtensions } from "../config.ts";
import { getIndexStats, withDb, getIndexedPaths } from "../database.ts";
import { collectFiles } from "../file-discovery.ts";
import { hybridSearch } from "../search.ts";
import { indexFiles } from "../indexing.ts";
import { resolveNote, readNote, buildIndexedFiles } from "../kb-reader.ts";
import { extractText } from "../text-extraction.ts";
import { BINARY_DOC_EXTS } from "../constants.ts";
import { storeScope, displayPath } from "./paths.ts";

/** Default note-like extensions when the effective allowlist is empty. */
const RAG_KB_READ_FALLBACK_EXTS = [".md", ".txt"];
/** Read cap for rag_kb_read (post-UTF8 bytes). Mirrors the pi-knowledge-search default. */
const RAG_KB_READ_MAX_BYTES = 64 * 1024;

/** Register all four RAG tools on the extension API. */
export function registerRagTools(pi: Pick<ExtensionAPI, "registerTool">) {
  pi.registerTool({
    name: "rag_index",
    label: "RAG index",
    description:
      "Index a file or directory into the local pi-local-rag pipeline. Chunks text files (including PDF and DOCX), generates embeddings, stores for hybrid BM25+vector search.",
    parameters: Type.Object({
      path: Type.String({ description: "File or directory path to index" }),
    }),
    execute: async (_toolCallId, params) => {
      if (!existsSync(params.path)) {
        return { content: [{ type: "text" as const, text: `Path not found: ${params.path}` }], details: undefined };
      }
      // Anchor a project-local store at cwd if there isn't one in scope yet.
      getRagDir({ createIfMissing: true });
      const config = loadConfig();
      const absolutePath = resolve(params.path);
      if (!config.trackedPaths.includes(absolutePath)) {
        config.trackedPaths.push(absolutePath);
        saveConfig(config);
      }
      const filesToIndex = collectFiles(absolutePath, undefined, config.excludePatterns);
      if (!filesToIndex.length) {
        return { content: [{ type: "text" as const, text: `No indexable files found in: ${params.path}` }], details: undefined };
      }
      const { result, enabledNow } = await withDb(async (database) => {
        const result = await indexFiles(filesToIndex, {}, database);
        // Enable auto-injection now that chunks exist (default is off).
        return { result, enabledNow: !config.ragEnabled && getIndexStats(database).totalChunks > 0 };
      });
      process.stderr.write("\n");
      if (enabledNow) {
        config.ragEnabled = true;
        saveConfig(config);
      }
      return {
        content: [{
          type: "text" as const,
          text:
            `Indexed ${result.indexed} files (${result.chunks} chunks, embeddings generated). ` +
            `${result.skipped} unchanged. ${(result.durationMs / 1000).toFixed(1)}s` +
            `${enabledNow ? " · RAG auto-injection enabled" : ""}`,
        }],
        details: undefined,
      };
    },
  });

  pi.registerTool({
    name: "rag_query",
    label: "RAG query",
    description:
      "Search the local pi-local-rag index using hybrid BM25+vector search. Returns relevant chunks with file paths, line numbers, and relevance scores.",
    parameters: Type.Object({
      query: Type.String({ description: "Search query" }),
      limit: Type.Optional(Type.Number({ description: "Max results (default 10)" })),
    }),
    execute: async (_toolCallId, params, _signal, _onUpdate, ctx) => {
      const config = loadConfig();
      const outcome = await withDb(async (database) => {
        if (!getIndexStats(database).totalChunks) return { empty: true as const };
        const results = await hybridSearch(params.query, params.limit ?? 10, config.ragAlpha, database);
        return { empty: false as const, results };
      });
      if (outcome.empty) {
        return { content: [{ type: "text" as const, text: "pi-local-rag index is empty. Run rag_index first." }], details: undefined };
      }
      if (!outcome.results.length) {
        return { content: [{ type: "text" as const, text: `No results for: ${params.query}` }], details: undefined };
      }
      const resultText = JSON.stringify(outcome.results.map(result => ({
        file: displayPath(result.chunk.file, ctx?.cwd ?? process.cwd()),
        lines: `${result.chunk.lineStart}-${result.chunk.lineEnd}`,
        tokens: result.chunk.tokens,
        scores: { bm25: result.bm25.toFixed(3), vector: result.vector.toFixed(3), hybrid: result.hybrid.toFixed(3) },
        preview: result.chunk.content.slice(0, 300),
      })), null, 2);
      return { content: [{ type: "text" as const, text: resultText }], details: undefined };
    },
  });

  pi.registerTool({
    name: "rag_status",
    label: "RAG status",
    description:
      "Show pi-local-rag index statistics: file count, chunk count, vector coverage, embedding model, RAG config.",
    parameters: Type.Object({}),
    execute: async () => {
      const config = loadConfig();
      const ragDir = getRagDir();
      const stats = await withDb(database => getIndexStats(database));
      const totalVectors = stats.embeddedCount + stats.embeddedCodeCount;
      const statusText = JSON.stringify({
        files: stats.totalFiles,
        chunks: stats.totalChunks,
        vectorsEmbedded: {
          text: stats.embeddedCount,
          code: stats.embeddedCodeCount,
        },
        vectorCoverage: stats.totalChunks ? `${Math.round((totalVectors / stats.totalChunks) * 100)}%` : "0%",
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
      return { content: [{ type: "text" as const, text: statusText }], details: undefined };
    },
  });

  pi.registerTool({
    name: "rag_kb_read",
    label: "KB Read",
    description:
      "Read an indexed file from the pi-local-rag knowledge base by name, relative path, or [[wikilink]]. Resolves fuzzy references without needing an absolute path — use this when you know a file's name but not its full path on disk. PDF/DOCX/HTML files are decoded to text.",
    promptGuidelines: [
      "Use rag_kb_read when a file is referenced by name or [[wikilink]] — don't run find/grep first.",
      "Use the standard `read` tool for non-indexed files or when you already have an absolute path.",
    ],
    parameters: Type.Object({
      name: Type.String({
        description:
          "Note reference: filename, basename, relative path, or [[wikilink]]. Examples: 'docs/hybrid-search', 'Hybrid search.md', '[[Hybrid search]]', '[[docs/hybrid-search|alias]]'.",
      }),
      max_bytes: Type.Optional(
        Type.Number({
          description: "Truncate output to at most this many bytes (default 65536).",
        }),
      ),
    }),
    execute: async (_toolCallId, params, _signal, _onUpdate, ctx) => {
      const config = loadConfig();
      const cwd = ctx?.cwd ?? process.cwd();

      const outcome = await withDb(async (database) => {
        if (!getIndexStats(database).totalChunks) return { empty: true as const };
        const paths = getIndexedPaths(database);
        const fileExtensions = [...resolveExtensions(config)];
        const indexedFiles = buildIndexedFiles({
          paths,
          roots: config.trackedPaths,
          cwd,
        });
        const result = resolveNote(params.name, indexedFiles, {
          fileExtensions: fileExtensions.length ? fileExtensions : RAG_KB_READ_FALLBACK_EXTS,
          cwd,
        });
        return { empty: false as const, result };
      });
      if (outcome.empty) {
        return {
          content: [{ type: "text" as const, text: "pi-local-rag index is empty. Run rag_index first." }],
          details: undefined,
        };
      }

      const result = outcome.result;
      if (result.matches.length === 0) {
        return {
          content: [{
            type: "text" as const,
            text: `No indexed file matched "${result.normalizedRef}". Try rag_query with a topic query to find related files.`,
          }],
          details: undefined,
        };
      }

      if (!result.unique && result.matches.length > 1) {
        const home = process.env.HOME || "";
        const listed = result.matches
          .map((m, i) => {
            const display = home && m.absPath.startsWith(home) ? m.absPath.replace(home, "~") : m.absPath;
            return `${i + 1}. ${display}  _(${m.reason})_`;
          })
          .join("\n");
        return {
          content: [{
            type: "text" as const,
            text:
              `"${result.normalizedRef}" is ambiguous. ${result.matches.length} candidates:\n\n${listed}\n\n` +
              `Call rag_kb_read again with a more specific path (e.g. the exact relative path) to disambiguate.`,
          }],
          details: { candidates: result.matches.map((m) => m.absPath) },
        };
      }

      const match = result.matches[0];
      let note;
      try {
        note = await readKbFile(match.absPath, params.max_bytes);
      } catch (err: unknown) {
        const msg = err instanceof Error ? err.message : String(err);
        return {
          content: [{ type: "text" as const, text: `Failed to read ${match.absPath}: ${msg}` }],
          details: undefined,
        };
      }

      const home = process.env.HOME || "";
      const display = home && note.path.startsWith(home) ? note.path.replace(home, "~") : note.path;
      const truncNote = note.truncated
        ? `\n\n_(truncated: showing first ${note.content.length} of ${note.totalBytes} bytes)_`
        : "";
      const section = result.subheading ? ` — section "${result.subheading}"` : "";
      // When a single low-confidence match slips through (fuzzy substring), flag
      // the reason so the agent can decide whether to trust the result or refine
      // the reference. High-confidence tiers are resolved silently.
      const fuzzyNote = !result.unique
        ? `\n\n_(fuzzy match via ${match.reason} — if this isn't the file you meant, re-run rag_kb_read with a more specific path)_`
        : "";
      const header = `# ${display}${section}${truncNote}${fuzzyNote}\n\n`;

      return {
        content: [{ type: "text" as const, text: header + note.content }],
        details: {
          resolvedPath: match.absPath,
          truncated: note.truncated,
        },
      };
    },
  });
}

/**
 * Read an indexed file for rag_kb_read. Binary documents (PDF/DOCX) and HTML go
 * through extractText() so the model gets decoded content instead of raw
 * bytes; plain text files are read with the UTF-8-safe readNote().
 */
async function readKbFile(absPath: string, maxBytes?: number): Promise<{
  path: string;
  content: string;
  truncated: boolean;
  totalBytes: number;
}> {
  const cap = maxBytes ?? RAG_KB_READ_MAX_BYTES;
  const extension = extname(absPath).toLowerCase();
  if (BINARY_DOC_EXTS.has(extension) || extension === ".html" || extension === ".htm") {
    const { text } = await extractText(absPath);
    const truncated = text.length > cap;
    return {
      path: absPath,
      content: truncated ? text.slice(0, cap) : text,
      truncated,
      totalBytes: text.length,
    };
  }
  return readNote(absPath, { maxBytes: cap });
}
