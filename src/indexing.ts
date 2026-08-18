/**
 * The indexing pipeline: parallel file reads/chunking (phase 1), then
 * per-model batched embedding with incremental transactional writes
 * (phases 2+3).
 *
 * Unchanged files (hash match + already embedded) are skipped before any
 * chunking CPU is spent. Chunk texts are released as soon as their vectors
 * are durable, keeping heap usage roughly flat across the run instead of
 * tracking the whole corpus.
 */
import { basename } from "node:path";
import type { DatabaseSync } from "node:sqlite";
import { getDbConn, type IndexStats } from "./database.ts";
import { EMBEDDING_MODEL, CODE_EMBEDDING_MODEL, CODE_EMBED_SCHEME, type EmbedGroup } from "./constants.ts";
import { embedBatchFor } from "./embedding.ts";
import { chunkText } from "./chunking.ts";
import { extractText } from "./text-extraction.ts";
import { sha256 } from "./hashing.ts";
import { classifyFile, loadConfig } from "./config.ts";
import { yieldToEventLoop, writeProgressLineToStderr } from "./runtime-utils.ts";
import * as sqlRepository from "./repository.ts";

/** Progress callbacks the indexer reports to (all optional). */
export interface ProgressCallbacks {
  /** Fired per processed file (including skipped ones). */
  onFile?: (current: number, total: number, filename: string, skipped: number) => void;
  /**
   * Fired once when the scan phase finishes: `toProcess` files need
   * (re)indexing out of `total` — the earliest point the new/changed
   * count is known, since it requires reading + hashing every file.
   */
  onScanComplete?: (toProcess: number, total: number) => void;
  onChunk?: (fileChunk: number, totalChunks: number, filename: string) => void;
  /**
   * Fired after each cross-file embed micro-batch completes, per embedding
   * group — `done`/`total` are scoped to `group` ("code" → jina model,
   * "text" → nomic model), so the TUI can render one progress line per
   * model instead of a single aggregate bar.
   */
  onEmbed?: (done: number, total: number, group: EmbedGroup) => void;
  /**
   * Fired right before a group's model is downloaded/loaded (once per
   * model per process) — covers the silent multi-minute cold-start
   * download.
   */
  onModelLoad?: (group: EmbedGroup, model: string) => void;
  onSave?: () => void;
}

/** True when the index was built more than `maxAgeMs` ago (0/blank = never stale). */
export function isIndexStale(index: IndexStats, maxAgeMs = 24 * 60 * 60 * 1000): boolean {
  if (!index.lastBuild) return false;
  return Date.now() - new Date(index.lastBuild).getTime() > maxAgeMs;
}

/** How many files the read/chunk phase processes concurrently. */
const READ_CONCURRENCY = 32;
/** How many drained files between event-loop yields in the write loop. */
const YIELD_INTERVAL = 64;
/** How many chunk texts go into one cross-file embedding batch. */
const EMBED_BATCH_TARGET = 256;

// Stderr progress is suppressed while a caller supplies TUI callbacks, so
// the two progress mechanisms don't fight over the same screen.
let suppressStderrProgress = false;

function stderrProgress(message: string) {
  if (suppressStderrProgress) return;
  writeProgressLineToStderr(message);
}

/**
 * A chunk with its content hash filled in (chunk ids are
 * `fileHash-chunkOrdinal-lineStart`). The ordinal is required for
 * uniqueness: a pathologically long line is split into MAX_LINE_CHARS
 * segments that share the source line's number, so several chunks of a
 * minified/CSV file can legitimately start on the same line — ids keyed by
 * lineStart alone collide and fail the chunks.id UNIQUE constraint.
 */
interface HashedChunk {
  content: string;
  lineStart: number;
  lineEnd: number;
  hash: string;
}

/**
 * Per-file work state for the embed/write phase. `rawChunks` slots are
 * nulled as soon as a chunk is durable, releasing its content (and with
 * the file's last chunk, the parent text the slices pin); `pending` hits
 * 0 → the file's row is upserted.
 */
interface FileWork {
  filePath: string;
  hash: string;
  size: number;
  group: EmbedGroup;
  /** Cached sha256(filePath) — the chunk-id prefix. */
  chunkIdPrefix: string;
  rawChunks: (HashedChunk | null)[];
  pending: number;
}

/** Outcome summary of one indexFiles() run. */
export interface IndexRunResult {
  indexed: number;
  chunks: number;
  chunksByGroup: Record<EmbedGroup, number>;
  skipped: number;
  durationMs: number;
}

/**
 * Index `paths`: read + chunk each file (skipping unchanged ones), embed
 * the chunks with their group's model, and write chunks + vectors + file
 * rows in per-slice transactions. `force` re-indexes even unchanged files.
 */
export async function indexFiles(
  paths: string[],
  progress?: ProgressCallbacks,
  databaseOverride?: DatabaseSync,
  force?: boolean,
): Promise<IndexRunResult> {
  const hasUiCallbacks = !!progress;
  if (hasUiCallbacks) suppressStderrProgress = true;
  const database = databaseOverride ?? getDbConn();
  const config = loadConfig();
  const startedAtMs = Date.now();
  const totalPaths = paths.length;

  try {
    if (totalPaths === 0) {
      return {
        indexed: 0,
        chunks: 0,
        chunksByGroup: { code: 0, text: 0 },
        skipped: 0,
        durationMs: Date.now() - startedAtMs,
      };
    }

    // ── Phase 1: parallel read + chunk; DB operations on the main thread ──

    // Producer-side skip cache: {path → stored hash/embedded} for every
    // requested path, loaded once before the workers start. Producers
    // consult it right after extractText (hash already in hand) and skip
    // chunkText for unchanged + already-embedded files — the skip decision
    // must happen before the chunking CPU is spent, not after.
    const existingFiles = force ? undefined : sqlRepository.getFilesByPaths(database, paths);

    /** One producer's output: extracted file identity + chunked rows (null when skipped). */
    interface ReadResult {
      filePath: string;
      hash: string;
      size: number;
      rawChunks: { content: string; lineStart: number; lineEnd: number }[] | null;
    }

    const readResultsQueue: ReadResult[] = [];
    let producersAllDone = false;
    let readErrorCount = 0;
    let resolveReadWait: (() => void) | null = null;
    const notifyReadAvailable = () => { resolveReadWait?.(); resolveReadWait = null; };
    const waitForReadAvailable = () => new Promise<void>(resolve => { resolveReadWait = resolve; });

    const workerCount = Math.min(READ_CONCURRENCY, totalPaths);
    let nextPathIndex = 0;
    let finishedWorkers = 0;
    const producers: Promise<void>[] = [];
    for (let workerSlot = 0; workerSlot < workerCount; workerSlot++) {
      producers.push((async () => {
        while (true) {
          const pathIndex = nextPathIndex++;
          if (pathIndex >= totalPaths) {
            finishedWorkers++;
            if (finishedWorkers >= workerCount) {
              producersAllDone = true;
              notifyReadAvailable();
            }
            return;
          }
          try {
            const { text, hash, size } = await extractText(paths[pathIndex]);
            const existing = existingFiles?.get(paths[pathIndex]);
            const unchanged = existing !== undefined && existing.hash === hash && !!existing.embedded;
            readResultsQueue.push({
              filePath: paths[pathIndex],
              hash,
              size,
              rawChunks: unchanged ? null : chunkText(text),
            });
            notifyReadAvailable();
          } catch {
            readErrorCount++;
            stderrProgress(`[${pathIndex + 1}/${totalPaths}] ERROR ${basename(paths[pathIndex])}: not found or unreadable`);
          }
        }
      })());
    }

    const filesToIndex: FileWork[] = [];
    let skippedCount = 0;
    let processedCount = 0;
    let nextYieldAt = 0;

    // Consume finished reads: skip unchanged files, else replace their
    // old chunks/vectors in the DB and queue the fresh work for embedding.
    const drainReadResults = () => {
      while (readResultsQueue.length > 0) {
        const readResult = readResultsQueue.shift()!;
        processedCount++;
        const fileName = basename(readResult.filePath);

        // Skip decision was made in the producer (hash match + embedded):
        // rawChunks is null and no chunking was spent on this file.
        if (readResult.rawChunks === null) {
          skippedCount++;
          progress?.onFile?.(processedCount, totalPaths, fileName, skippedCount);
          continue;
        }

        sqlRepository.deleteVectorsForFile(database, readResult.filePath);
        sqlRepository.deleteCodeVectorsForFile(database, readResult.filePath);
        sqlRepository.deleteChunksForFile(database, readResult.filePath);

        // Annotate chunks with their hash in place — a `{ ...chunk, hash }`
        // spread map would briefly double the chunk-object count per file.
        const rawChunks = readResult.rawChunks as HashedChunk[];
        for (const chunk of rawChunks) chunk.hash = sha256(chunk.content);
        stderrProgress(`[${processedCount}/${totalPaths}] chunked ${fileName} (${rawChunks.length} chunks)`);
        progress?.onFile?.(processedCount, totalPaths, fileName, skippedCount);

        filesToIndex.push({
          filePath: readResult.filePath,
          hash: readResult.hash,
          size: readResult.size,
          group: classifyFile(readResult.filePath, config),
          chunkIdPrefix: sha256(readResult.filePath),
          rawChunks,
          pending: rawChunks.length,
        });
      }
    };

    const maybeYieldToUi = async () => {
      if (processedCount >= nextYieldAt) {
        nextYieldAt = processedCount + YIELD_INTERVAL;
        await yieldToEventLoop();
      }
    };

    while (!producersAllDone || readResultsQueue.length > 0) {
      drainReadResults();
      if (!producersAllDone) await waitForReadAvailable();
      await maybeYieldToUi();
    }
    drainReadResults();
    await yieldToEventLoop();

    skippedCount += readErrorCount;
    progress?.onScanComplete?.(filesToIndex.length, totalPaths);

    // ── Phases 2+3: embed in cross-file groups, per model; write each ──
    // ── slice to the DB as soon as its vectors arrive ──────────────────

    // Interleaving embed with insert — instead of accumulating every
    // chunk's text plus one 768-dim Float32Array per chunk for the whole
    // corpus until a single giant end-of-run transaction — lets each
    // slice's strings and vectors be collected right after they hit the
    // database, and keeps every transaction (and therefore the WAL) small.
    // Code chunks go through the jina pipeline, prose chunks through
    // nomic — each with its own progress counter.
    const chunkRefsByGroup: Record<EmbedGroup, { fileWork: FileWork; chunkIndex: number }[]> = { code: [], text: [] };
    for (const fileWork of filesToIndex) {
      for (let chunkIndex = 0; chunkIndex < fileWork.rawChunks.length; chunkIndex++) {
        chunkRefsByGroup[fileWork.group].push({ fileWork, chunkIndex });
      }
    }
    // Sort by content length so each group embeds similar-length texts
    // together — ONNX pads every text in a batch to the longest, so an
    // unsorted mix inflates the attention matrix for all its neighbors.
    // Vectors are written back by position, so embed order is free.
    for (const group of ["code", "text"] as const) {
      chunkRefsByGroup[group].sort((a, b) =>
        a.fileWork.rawChunks[a.chunkIndex]!.content.length - b.fileWork.rawChunks[b.chunkIndex]!.content.length,
      );
    }

    let writtenChunkCount = 0;
    const chunksWrittenByGroup: Record<EmbedGroup, number> = { code: 0, text: 0 };
    const indexedAtTimestamp = new Date().toISOString();

    // Files whose chunking produced nothing (fully blank files) never
    // complete through the embed loop — record them up front so later runs
    // can skip them.
    for (const fileWork of filesToIndex) {
      if (fileWork.rawChunks.length === 0) {
        sqlRepository.upsertFile(database, fileWork.filePath, fileWork.hash, 0, indexedAtTimestamp, fileWork.size, true);
      }
    }

    // Write one embed slice (chunks + vectors) in a single transaction.
    const insertSlice = (slice: { fileWork: FileWork; chunkIndex: number }[], vectors: Float32Array[]) => {
      sqlRepository.runInTransaction(database, () => {
        for (let slicePosition = 0; slicePosition < slice.length; slicePosition++) {
          const { fileWork, chunkIndex } = slice[slicePosition];
          const chunk = fileWork.rawChunks[chunkIndex]!;
          const insertResult = sqlRepository.insertChunk(database, {
            id: `${fileWork.chunkIdPrefix}-${chunkIndex}-${chunk.lineStart}`,
            filePath: fileWork.filePath,
            content: chunk.content,
            lineStart: chunk.lineStart,
            lineEnd: chunk.lineEnd,
            hash: chunk.hash,
            indexedAt: indexedAtTimestamp,
            tokens: Math.ceil(chunk.content.length / 4),
          });
          const vector = vectors[slicePosition];
          if (vector) {
            if (fileWork.group === "code") {
              sqlRepository.insertCodeVector(database, Number(insertResult.lastInsertRowid), vector);
            } else {
              sqlRepository.insertVector(database, Number(insertResult.lastInsertRowid), vector);
            }
          }
          writtenChunkCount++;
        }
      });
    };

    for (const group of ["code", "text"] as const) {
      const groupChunkRefs = chunkRefsByGroup[group];
      const groupTotal = groupChunkRefs.length;
      for (let sliceStart = 0; sliceStart < groupChunkRefs.length; sliceStart += EMBED_BATCH_TARGET) {
        const slice = groupChunkRefs.slice(sliceStart, sliceStart + EMBED_BATCH_TARGET);
        const sliceTexts = slice.map(ref => ref.fileWork.rawChunks[ref.chunkIndex]!.content);
        // Fire before the batch too, so the TUI flips to the "Embedding"
        // widget (covering first-run model download) instead of the stale
        // 100% screen.
        progress?.onEmbed?.(sliceStart, groupTotal, group);
        stderrProgress(`Embedding(${group}) ${sliceStart + 1}…${sliceStart + slice.length}/${groupTotal} chunks`);
        // Forward embedBatchFor's per-batch (BATCH_SIZE) progress so the
        // TUI updates every 64 chunks instead of once per 256-chunk slice.
        const vectors = await embedBatchFor(
          group,
          sliceTexts,
          {
            onProgress: done => progress?.onEmbed?.(sliceStart + done, groupTotal, group),
            onModelLoad: model => progress?.onModelLoad?.(group, model),
          },
          slice.map(ref => ref.fileWork.filePath),
        );
        insertSlice(slice, vectors);

        // The slice is durable — drop its chunk contents (which pin their
        // parent file text) and close out any file whose last chunk just
        // landed. This is what keeps heap usage roughly flat across the
        // run instead of tracking the whole corpus.
        for (let slicePosition = 0; slicePosition < slice.length; slicePosition++) {
          const ref = slice[slicePosition];
          ref.fileWork.rawChunks[ref.chunkIndex] = null;
          chunksWrittenByGroup[ref.fileWork.group]++;
          if (--ref.fileWork.pending === 0) {
            // rawChunks keeps its slot count as the file's chunk total.
            sqlRepository.upsertFile(
              database,
              ref.fileWork.filePath,
              ref.fileWork.hash,
              ref.fileWork.rawChunks.length,
              indexedAtTimestamp,
              ref.fileWork.size,
              true,
            );
          }
        }
        progress?.onEmbed?.(sliceStart + slice.length, groupTotal, group);
        // Yield so the TUI can render the progress update before the next batch.
        await yieldToEventLoop();
      }
    }

    if (!hasUiCallbacks) process.stderr.write("\r\x1b[2K");
    progress?.onSave?.();
    sqlRepository.setMetadata(database, sqlRepository.MetadataKey.LastBuild, new Date().toISOString());
    sqlRepository.setMetadata(database, sqlRepository.MetadataKey.EmbeddingModel, EMBEDDING_MODEL);
    sqlRepository.setMetadata(database, sqlRepository.MetadataKey.EmbeddingCodeModel, CODE_EMBEDDING_MODEL);
    sqlRepository.setMetadata(database, sqlRepository.MetadataKey.EmbeddingCodeScheme, CODE_EMBED_SCHEME);

    return {
      indexed: filesToIndex.length,
      chunks: writtenChunkCount,
      chunksByGroup: chunksWrittenByGroup,
      skipped: skippedCount,
      durationMs: Date.now() - startedAtMs,
    };
  } finally {
    if (hasUiCallbacks) suppressStderrProgress = false;
  }
}
