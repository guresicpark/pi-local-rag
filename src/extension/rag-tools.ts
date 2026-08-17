/**
 * Model-facing tools: rag_index, rag_query, rag_status. Thin wrappers
 * around the same core functions the /rag commands use, returning
 * plain-text results for the model.
 */
import { existsSync } from "node:fs";
import { resolve } from "node:path";
import { Type } from "typebox";
import type { ExtensionAPI } from "@earendil-works/pi-coding-agent";
import { getRagDir } from "../store-paths.ts";
import { loadConfig, saveConfig } from "../config.ts";
import { getIndexStats, withDb } from "../database.ts";
import { collectFiles } from "../file-discovery.ts";
import { hybridSearch } from "../search.ts";
import { indexFiles } from "../indexing.ts";
import { storeScope, displayPath } from "./paths.ts";

/** Register all three RAG tools on the extension API. */
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
}
