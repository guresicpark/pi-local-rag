/**
 * Hybrid search: SQLite FTS5 (BM25) + sqlite-vec (vector KNN) over two
 * independent embedding spaces (prose/nomic and code/jina), merged with a
 * per-query blend weight.
 */
import type { DatabaseSync } from "node:sqlite";
import { embedQueryFor } from "./embedding.ts";
import { getDbConn, type Chunk } from "./database.ts";
import * as sqlRepository from "./repository.ts";

/** A chunk plus its individual and blended relevance scores. */
export interface ScoredChunk {
  chunk: Chunk;
  bm25: number;
  vector: number;
  hybrid: number;
}

/** Cosine similarity between two vectors; 0 for length mismatch or zero vectors. */
export function cosineSimilarity(vectorA: number[], vectorB: number[]): number {
  if (vectorA.length !== vectorB.length) return 0;
  let dotProduct = 0;
  let normA = 0;
  let normB = 0;
  for (let i = 0; i < vectorA.length; i++) {
    dotProduct += vectorA[i] * vectorB[i];
    normA += vectorA[i] * vectorA[i];
    normB += vectorB[i] * vectorB[i];
  }
  const denominator = Math.sqrt(normA) * Math.sqrt(normB);
  return denominator === 0 ? 0 : dotProduct / denominator;
}

/**
 * Min-max normalize scores into [0, 1] (order preserved). All-equal input
 * maps to all zeros. Implemented with a single loop — the spread-based
 * Math.max(...scores) form risks blowing the argument-limit on large
 * candidate sets.
 */
export function normalize(scores: number[]): number[] {
  let max = -Infinity;
  let min = Infinity;
  for (const score of scores) {
    if (score > max) max = score;
    if (score < min) min = score;
  }
  const range = max - min;
  if (range === 0) return scores.map(() => 0);
  return scores.map(score => (score - min) / range);
}

/**
 * Convert a sqlite-vec L2 distance into a cosine similarity. Both models
 * produce unit-normalized embeddings, for which
 *   ||a − b||² = 2 − 2·cos(θ)  →  cos(θ) = 1 − ||a − b||²/2.
 */
function l2DistanceToCosine(l2Distance: number): number {
  return 1 - (l2Distance * l2Distance) / 2;
}

/**
 * Run hybrid search over the index: BM25 via FTS5 plus vector KNN over
 * both embedding spaces, blended as `alpha * bm25 + (1 - alpha) * vector`
 * (alpha = 1 → pure BM25 when no vectors exist).
 */
export async function hybridSearch(
  query: string,
  limit = 10,
  alpha = 0.4,
  databaseOverride?: DatabaseSync,
): Promise<ScoredChunk[]> {
  const database = databaseOverride ?? getDbConn();

  // Fast existence check — LIMIT 1 avoids a full table scan.
  if (!sqlRepository.hasAnyChunks(database)) return [];

  // BM25 via FTS5. Terms are quoted (with "" escaping) so punctuation in
  // the query can't produce an FTS syntax error; candidates are capped to
  // avoid scanning the entire index.
  const ftsQuery = query
    .split(/\s+/)
    .map(term => `"${term.replace(/"/g, '""')}"`)
    .join(" ");
  const ftsCandidateLimit = Math.max(limit * 20, 200);
  const ftsResults = sqlRepository.searchFts(database, ftsQuery, ftsCandidateLimit);

  // Vector search via sqlite-vec — two independent spaces: prose chunks
  // live in chunks_vec (nomic), code chunks in chunks_vec_code (jina).
  // Each model's query is only built when its table actually has vectors
  // (avoids loading a model the store can't use).
  const vectorCandidateLimit = Math.max(limit * 10, 100);
  const [textVectorResults, codeVectorResults] = await Promise.all([
    (async () =>
      sqlRepository.getEmbeddedCount(database)
        ? sqlRepository.searchVectors(database, await embedQueryFor("text", query), vectorCandidateLimit)
        : [])(),
    (async () =>
      sqlRepository.getCodeEmbeddedCount(database)
        ? sqlRepository.searchCodeVectors(database, await embedQueryFor("code", query), vectorCandidateLimit)
        : [])(),
  ]);

  // Union of candidate rowids from all three sources, then hydrate once.
  const candidateRowIds = new Set<number>([
    ...ftsResults.map(match => match.rowid),
    ...textVectorResults.map(match => match.rowid),
    ...codeVectorResults.map(match => match.rowid),
  ]);
  if (candidateRowIds.size === 0) return [];

  const chunkRows = sqlRepository.getChunksByRowids(database, Array.from(candidateRowIds));
  const chunkByRowid = new Map<number, typeof chunkRows[0]>();
  for (const chunkRow of chunkRows) chunkByRowid.set(chunkRow.rowid, chunkRow);

  // Min-max normalize BM25 scores across the FTS candidate set (constant 1
  // when every candidate ties, so ties stay rankable rather than zeroed).
  const bm25NormalizedByRowid = new Map<number, number>();
  if (ftsResults.length > 0) {
    let bm25Max = -Infinity;
    let bm25Min = Infinity;
    for (const match of ftsResults) {
      if (match.bm25_score > bm25Max) bm25Max = match.bm25_score;
      if (match.bm25_score < bm25Min) bm25Min = match.bm25_score;
    }
    const bm25Range = bm25Max - bm25Min;
    for (const match of ftsResults) {
      bm25NormalizedByRowid.set(
        match.rowid,
        bm25Range === 0 ? 1 : (match.bm25_score - bm25Min) / bm25Range,
      );
    }
  }

  // Both models produce unit-normalized embeddings, so raw cosine similarity
  // is on a shared absolute scale across the two spaces — no per-space
  // min-max (that would pin the top chunk of each space at 1.0 and create
  // structural cross-space ties, e.g. a lone prose chunk always tying the
  // best code chunk). Clamped at 0 to keep hybrid scores non-negative.
  const vectorSimilarityByRowid = new Map<number, number>();
  for (const match of textVectorResults) {
    vectorSimilarityByRowid.set(match.rowid, Math.max(0, l2DistanceToCosine(match.distance)));
  }
  for (const match of codeVectorResults) {
    vectorSimilarityByRowid.set(match.rowid, Math.max(0, l2DistanceToCosine(match.distance)));
  }
  const hasAnyVectors = vectorSimilarityByRowid.size > 0;

  // Score every candidate: normalized BM25 (with a filename-boost) blended
  // with the vector similarity.
  const meaningfulQueryTerms = query.toLowerCase().split(/\s+/).filter(term => term.length > 1);
  const scoredResults: ScoredChunk[] = [];

  for (const rowid of candidateRowIds) {
    const chunkRow = chunkByRowid.get(rowid);
    if (!chunkRow) continue;

    const bm25Normalized = bm25NormalizedByRowid.get(rowid) ?? 0;
    const vectorSimilarity = vectorSimilarityByRowid.get(rowid) ?? 0;

    // Boost when the first meaningful query term appears in the file path.
    // Guarded on terms[0]: an empty/short query makes includes("") always
    // true, which would spuriously boost every result.
    let bm25Final = bm25Normalized;
    const firstQueryTerm = meaningfulQueryTerms[0];
    if (firstQueryTerm && chunkRow.file_path.toLowerCase().includes(firstQueryTerm)) {
      bm25Final = Math.min(1, bm25Final * 1.5);
    }

    const hybridScore = hasAnyVectors
      ? alpha * bm25Final + (1 - alpha) * vectorSimilarity
      : bm25Final;

    scoredResults.push({
      chunk: {
        id: chunkRow.id,
        file: chunkRow.file_path,
        content: chunkRow.chunk_content,
        lineStart: chunkRow.line_start,
        lineEnd: chunkRow.line_end,
        hash: chunkRow.chunk_hash,
        indexed: chunkRow.indexed_at,
        tokens: chunkRow.tokens,
      },
      bm25: bm25Final,
      vector: vectorSimilarity,
      hybrid: hybridScore,
    });
  }

  return scoredResults
    .filter(scored => scored.hybrid > 0)
    .sort((a, b) => b.hybrid - a.hybrid)
    .slice(0, limit);
}
