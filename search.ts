import Database from "better-sqlite3";
import { embedQueryFor } from "./embed.ts";
import { getDbConn } from "./db.ts";
import { Chunk } from "./db.ts";
import * as repo from "./repository.ts";

export interface ScoredChunk {
  chunk: Chunk;
  bm25: number;
  vector: number;
  hybrid: number;
}

export function cosineSimilarity(a: number[], b: number[]): number {
  if (a.length !== b.length) return 0;
  let dot = 0, normA = 0, normB = 0;
  for (let i = 0; i < a.length; i++) {
    dot += a[i] * b[i];
    normA += a[i] * a[i];
    normB += b[i] * b[i];
  }
  const denom = Math.sqrt(normA) * Math.sqrt(normB);
  return denom === 0 ? 0 : dot / denom;
}

export function normalize(scores: number[]): number[] {
  const max = Math.max(...scores);
  const min = Math.min(...scores);
  const range = max - min;
  if (range === 0) return scores.map(() => 0);
  return scores.map(s => (s - min) / range);
}

function l2ToCosine(l2Dist: number): number {
  return 1 - (l2Dist * l2Dist) / 2;
}

/**
 * Hybrid search using SQLite FTS5 (BM25) + sqlite-vec (vector).
 */
export async function hybridSearch(
  query: string,
  limit = 10,
  alpha = 0.4,
  _db?: Database.Database
): Promise<ScoredChunk[]> {
  const database = _db ?? getDbConn();

  // Fast existence check — LIMIT 1 avoids full table scan
  if (!repo.hasAnyChunks(database)) return [];

  // BM25 via FTS5 — cap candidates to avoid scanning entire index
  const ftsQuery = query.split(/\s+/).map(t => `"${t.replace(/"/g, '""')}"`).join(" ");
  const ftsLimit = Math.max(limit * 20, 200);
  const ftsResults = repo.searchFts(database, ftsQuery, ftsLimit);

  // Vector via sqlite-vec — two independent spaces: prose chunks live in
  // chunks_vec (nomic), code chunks in chunks_vec_code (jina). Each model's
  // query is only built when its table actually has vectors (avoids loading
  // a model the store can't use). Distances are normalized per-space before
  // merging, since raw scores across different embedding spaces are
  // meaningless.
  const vecLimit = Math.max(limit * 10, 100);
  const [textVecResults, codeVecResults] = await Promise.all([
    (async () =>
      repo.getEmbeddedCount(database)
        ? repo.searchVectors(database, await embedQueryFor("text", query), vecLimit)
        : [])(),
    (async () =>
      repo.getCodeEmbeddedCount(database)
        ? repo.searchCodeVectors(database, await embedQueryFor("code", query), vecLimit)
        : [])(),
  ]);

  const ftsRowIds = new Set(ftsResults.map(r => r.rowid));
  const vecRowIds = new Set([...textVecResults, ...codeVecResults].map(r => r.rowid));
  const allRowIds: Set<number> = new Set([...ftsRowIds, ...vecRowIds]);

  if (allRowIds.size === 0) return [];

  const chunks = repo.getChunksByRowids(database, Array.from(allRowIds));

  const chunkMap = new Map<number, typeof chunks[0]>();
  for (const c of chunks) chunkMap.set(c.rowid, c);

  const bm25Scores = ftsResults.map(r => r.bm25_score);
  const hasBm25 = bm25Scores.length > 0;

  // Normalize BM25
  const bm25NormMap = new Map<number, number>();
  if (hasBm25) {
    const bm25Max = Math.max(...bm25Scores);
    const bm25Min = Math.min(...bm25Scores);
    const bm25Range = bm25Max - bm25Min;
    if (bm25Range === 0) {
      for (const r of ftsResults) {
        bm25NormMap.set(r.rowid, 1);
      }
    } else {
      for (const r of ftsResults) {
        bm25NormMap.set(r.rowid, (r.bm25_score - bm25Min) / bm25Range);
      }
    }
  }

  // Both models produce unit-normalized embeddings, so raw cosine similarity
  // is on a shared absolute scale across the two spaces — no per-space
  // min-max (that would pin the top chunk of each space at 1.0 and create
  // structural cross-space ties, e.g. a lone prose chunk always tying the
  // best code chunk). Clamped at 0 to keep hybrid scores non-negative.
  const vecNormMap = new Map<number, number>();
  for (const r of textVecResults) vecNormMap.set(r.rowid, Math.max(0, l2ToCosine(r.distance)));
  for (const r of codeVecResults) vecNormMap.set(r.rowid, Math.max(0, l2ToCosine(r.distance)));
  const hasVectors = vecNormMap.size > 0;

  // Build scored results
  const terms = query.toLowerCase().split(/\s+/).filter(t => t.length > 1);
  const scored: ScoredChunk[] = [];

  for (const rowid of allRowIds) {
    const c = chunkMap.get(rowid);
    if (!c) continue;

    const bm25Norm = bm25NormMap.get(rowid) ?? 0;
    const vecNorm = vecNormMap.get(rowid) ?? 0;

      let bm25Final = bm25Norm;
      // Boost when the first meaningful query term appears in the file path.
      // Guard on terms[0]: an empty/short query makes includes("") always true,
      // which would spuriously boost every result.
      if (terms[0] && c.file_path.toLowerCase().includes(terms[0])) {
        bm25Final = Math.min(1, bm25Final * 1.5);
      }

    const hybrid = hasVectors
      ? alpha * bm25Final + (1 - alpha) * vecNorm
      : bm25Final;

    scored.push({
      chunk: {
        id: c.id, file: c.file_path, content: c.chunk_content,
        lineStart: c.line_start, lineEnd: c.line_end,
        hash: c.chunk_hash, indexed: c.indexed_at, tokens: c.tokens,
      },
      bm25: bm25Final, vector: vecNorm, hybrid,
    });
  }

  return scored
    .filter(s => s.hybrid > 0)
    .sort((a, b) => b.hybrid - a.hybrid)
    .slice(0, limit);
}
