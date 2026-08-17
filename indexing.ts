import { basename } from "node:path";
import Database from "better-sqlite3";
import { getDbConn, type IndexStats } from "./db.ts";
import { EMBEDDING_MODEL, CODE_EMBEDDING_MODEL, CODE_EMBED_SCHEME, type EmbedGroup } from "./constants.ts";
import { embedBatchFor } from "./embed.ts";
import { chunkText, extractText, sha256 } from "./chunking.ts";
import { classifyFile, loadConfig } from "./config.ts";
import * as repo from "./repository.ts";

export interface ProgressCallbacks {
  onFile?: (current: number, total: number, filename: string, skipped: number) => void;
  onChunk?: (fileChunk: number, totalChunks: number, filename: string) => void;
  /** Fires after each cross-file embed micro-batch completes, per embedding
   *  group — `done`/`total` are scoped to `group` ("code" → jina model,
   *  "text" → nomic model), so the TUI can render one progress line per
   *  model instead of a single aggregate bar. */
  onEmbed?: (done: number, total: number, group: EmbedGroup) => void;
  /** Fires right before a group's model is downloaded/loaded (once per model
   *  per process) — covers the silent multi-minute cold-start download. */
  onModelLoad?: (group: EmbedGroup, model: string) => void;
  onSave?: () => void;
}

export function isIndexStale(index: IndexStats, maxAgeMs = 24 * 60 * 60 * 1000): boolean {
  if (!index.lastBuild) return false;
  return Date.now() - new Date(index.lastBuild).getTime() > maxAgeMs;
}

const yield_ = () => new Promise<void>(r => setTimeout(r, 0));

let _suppressStderr = false;

function stderrProgress(msg: string) {
  if (_suppressStderr) return;
  process.stderr.write(`\r\x1b[2K${msg}`);
}

interface HashedChunk { content: string; lineStart: number; lineEnd: number; hash: string }

interface FileWork {
  fp: string;
  hash: string;
  size: number;
  group: EmbedGroup;
  /** Cached sha256(fp) — the chunk-id prefix, previously recomputed per chunk. */
  idPrefix: string;
  /** Slots are nulled as soon as a chunk is durable, releasing its content
   *  (and with the file's last chunk, the parent text the slices pinned). */
  rawChunks: (HashedChunk | null)[];
  /** Chunks not yet written; hits 0 → upsertFile. */
  pending: number;
}

export async function indexFiles(
  paths: string[],
  progress?: ProgressCallbacks,
  _db?: Database.Database,
  force?: boolean,
): Promise<{ indexed: number; chunks: number; chunksByGroup: Record<EmbedGroup, number>; skipped: number; durationMs: number }> {
  const hadCallbacks = !!progress;
  if (hadCallbacks) _suppressStderr = true;
  const database = _db ?? getDbConn();
  const config = loadConfig();
  const startMs = Date.now();
  const total = paths.length;

  try {
    if (total === 0) {
      return { indexed: 0, chunks: 0, chunksByGroup: { code: 0, text: 0 }, skipped: 0, durationMs: Date.now() - startMs };
    }

    // Phase 1: parallel read + chunk; DB ops on main thread
    const CONCURRENCY = 32;
    const YIELD_INTERVAL = 64;

    // Producer-side skip cache: {path → stored hash/embedded} for every
    // requested path, loaded once before the workers start. Producers
    // consult it right after extractText (hash already in hand) and skip
    // chunkText for unchanged + already-embedded files — previously the
    // skip decision happened only in drainReads, after the chunking CPU had
    // already been spent and discarded on every re-index.
    const existingFiles = force ? undefined : repo.getFilesByPaths(database, paths);

    interface ReadResult { fp: string; hash: string; size: number; raw: { content: string; lineStart: number; lineEnd: number }[] | null }

    const readQueue: ReadResult[] = [];
    let readQueueDone = false;
    let readErrorCount = 0;
    let resolveRead: (() => void) | null = null;
    const notifyRead = () => { resolveRead?.(); resolveRead = null; };
    const waitRead = () => new Promise<void>(r => { resolveRead = r; });

    const workerCount = Math.min(CONCURRENCY, paths.length);
    let pathsIdx = 0;
    let producersDone = 0;
    const producers: Promise<void>[] = [];
    for (let w = 0; w < workerCount; w++) {
      producers.push((async () => {
        while (true) {
          const i = pathsIdx++;
          if (i >= paths.length) { producersDone++; if (producersDone >= workerCount) { readQueueDone = true; notifyRead(); } return; }
          try {
            const { text, hash, size } = await extractText(paths[i]);
            const existing = existingFiles?.get(paths[i]);
            const unchanged = existing !== undefined && existing.hash === hash && !!existing.embedded;
            readQueue.push({ fp: paths[i], hash, size, raw: unchanged ? null : chunkText(text) });
            notifyRead();
          } catch {
            readErrorCount++;
            stderrProgress(`[${i + 1}/${total}] ERROR ${basename(paths[i])}: not found or unreadable`);
          }
        }
      })());
    }

    const toIndex: FileWork[] = [];
    let skipped = 0;
    let processedCount = 0;
    let nextYieldAt = 0;

    const drainReads = () => {
      while (readQueue.length > 0) {
        const r = readQueue.shift()!;
        processedCount++;
        const name = basename(r.fp);

        // Skip decision was made in the producer (hash match + embedded):
        // raw is null and no chunking was spent on this file.
        if (r.raw === null) {
          skipped++;
          progress?.onFile?.(processedCount, total, name, skipped);
          continue;
        }

        repo.deleteVectorsForFile(database, r.fp);
        repo.deleteCodeVectorsForFile(database, r.fp);
        repo.deleteChunksForFile(database, r.fp);

        // Annotate chunks with their hash in place — a `{ ...c, hash }`
        // spread map briefly doubled the chunk-object count per file.
        const rawChunks = r.raw as HashedChunk[];
        for (const c of rawChunks) c.hash = sha256(c.content);
        stderrProgress(`[${processedCount}/${total}] chunked ${name} (${rawChunks.length} chunks)`);
        progress?.onFile?.(processedCount, total, name, skipped);

        toIndex.push({
          fp: r.fp, hash: r.hash, size: r.size,
          group: classifyFile(r.fp, config), idPrefix: sha256(r.fp),
          rawChunks, pending: rawChunks.length,
        });
      }
    };

    const maybeYield = async () => {
      if (processedCount >= nextYieldAt) {
        nextYieldAt = processedCount + YIELD_INTERVAL;
        await yield_();
      }
    };

    while (!readQueueDone || readQueue.length > 0) {
      drainReads();
      if (!readQueueDone) await waitRead();
      await maybeYield();
    }
    drainReads();
    await yield_();

    skipped += readErrorCount;

    // Phase 2+3: embed in cross-file groups, per embedding model, and write
    // each slice to the DB as soon as its vectors arrive. Interleaving embed
    // with insert — instead of accumulating every chunk's text plus one
    // 768-dim Float32Array per chunk for the whole corpus until a single
    // giant end-of-run transaction — lets each slice's strings and vectors
    // be collected right after they hit the database, and keeps every
    // transaction (and therefore the WAL) small. Code chunks go through the
    // jina pipeline, prose chunks through nomic — each with its own
    // progress counter.
    const EMBED_GROUP_TARGET = 256;
    const groupPairs: Record<EmbedGroup, { fw: FileWork; ci: number }[]> = { code: [], text: [] };
    for (const fw of toIndex) {
      for (let j = 0; j < fw.rawChunks.length; j++) groupPairs[fw.group].push({ fw, ci: j });
    }
    // Sort by content length so each group embeds similar-length texts
    // together — ONNX pads every text in a batch to the longest, so an
    // unsored mix inflates the attention matrix for all its neighbors.
    // Vectors are written back by position, so embed order is free.
    for (const g of ["code", "text"] as const) {
      groupPairs[g].sort((a, b) => a.fw.rawChunks[a.ci]!.content.length - b.fw.rawChunks[b.ci]!.content.length);
    }

    let chunked = 0;
    const chunksByGroup: Record<EmbedGroup, number> = { code: 0, text: 0 };
    const indexedAt = new Date().toISOString();

    // Files whose chunking produced nothing (fully blank files) never
    // complete through the embed loop — record them up front so later runs
    // can skip them.
    for (const fw of toIndex) {
      if (fw.rawChunks.length === 0) repo.upsertFile(database, fw.fp, fw.hash, 0, indexedAt, fw.size, true);
    }

    const insertSlice = database.transaction((entries: { fw: FileWork; ci: number }[], vectors: Float32Array[]) => {
      for (let k = 0; k < entries.length; k++) {
        const { fw, ci } = entries[k];
        const c = fw.rawChunks[ci]!;
        const chunkResult = repo.insertChunk(database, {
          id: `${fw.idPrefix}-${c.lineStart}`,
          filePath: fw.fp, content: c.content,
          lineStart: c.lineStart, lineEnd: c.lineEnd, hash: c.hash,
          indexedAt, tokens: Math.ceil(c.content.length / 4),
        });
        if (vectors[k]) {
          if (fw.group === "code") repo.insertCodeVector(database, Number(chunkResult.lastInsertRowid), vectors[k]);
          else repo.insertVector(database, Number(chunkResult.lastInsertRowid), vectors[k]);
        }
        chunked++;
      }
    });

    for (const g of ["code", "text"] as const) {
      const pairs = groupPairs[g];
      const groupTotal = pairs.length;
      for (let p = 0; p < pairs.length; p += EMBED_GROUP_TARGET) {
        const slice = pairs.slice(p, p + EMBED_GROUP_TARGET);
        const texts = slice.map(x => x.fw.rawChunks[x.ci]!.content);
        // Fire before the batch too, so the TUI flips to the "Embedding"
        // widget (covering first-run model download) instead of the stale
        // 100% screen.
        progress?.onEmbed?.(p, groupTotal, g);
        stderrProgress(`Embedding(${g}) ${p + 1}…${p + slice.length}/${groupTotal} chunks`);
        // Forward embedBatchFor's per-batch (BATCH_SIZE) progress so the TUI
        // updates every 64 chunks instead of once per 256-chunk group.
        const vectors = await embedBatchFor(g, texts, {
          onProgress: done => progress?.onEmbed?.(p + done, groupTotal, g),
          onModelLoad: model => progress?.onModelLoad?.(g, model),
        }, slice.map(x => x.fw.fp));
        insertSlice(slice, vectors);
        // The slice is durable — drop its chunk contents (which pin their
        // parent file text) and close out any file whose last chunk just
        // landed. This is what keeps heap usage roughly flat across the
        // run instead of tracking the whole corpus.
        for (let vi = 0; vi < slice.length; vi++) {
          const x = slice[vi];
          x.fw.rawChunks[x.ci] = null;
          chunksByGroup[x.fw.group]++;
          if (--x.fw.pending === 0) {
            // rawChunks keeps its slot count as the file's chunk total.
            repo.upsertFile(database, x.fw.fp, x.fw.hash, x.fw.rawChunks.length, indexedAt, x.fw.size, true);
          }
        }
        progress?.onEmbed?.(p + slice.length, groupTotal, g);
        // Yield so the TUI can render the progress update before the next batch.
        await yield_();
      }
    }

    if (!hadCallbacks) process.stderr.write(`\r\x1b[2K`);
    progress?.onSave?.();
    repo.setMetadata(database, repo.MetadataKey.LastBuild, new Date().toISOString());
    repo.setMetadata(database, repo.MetadataKey.EmbeddingModel, EMBEDDING_MODEL);
    repo.setMetadata(database, repo.MetadataKey.EmbeddingCodeModel, CODE_EMBEDDING_MODEL);
    repo.setMetadata(database, repo.MetadataKey.EmbeddingCodeScheme, CODE_EMBED_SCHEME);

    return { indexed: toIndex.length, chunks: chunked, chunksByGroup, skipped, durationMs: Date.now() - startMs };
  } finally {
    if (hadCallbacks) _suppressStderr = false;
  }
}
