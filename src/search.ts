/**
 * Hybrid search: SQLite FTS5 (BM25) + sqlite-vec (vector KNN) over two
 * independent embedding spaces (prose/nomic and code/jina), merged with a
 * per-query blend weight. Each space's query is embedded only when that
 * space has stored vectors; results are a fixed quota split — 5 code hits
 * + 5 prose hits when both groups qualify, 5 of the single group
 * otherwise.
 */
import type { DatabaseSync } from "node:sqlite";
import { embedQueryFor } from "./embedding.ts";
import { getDbConn, type Chunk } from "./database.ts";
import * as sqlRepository from "./repository.ts";

/**
 * Which engine surfaced a chunk as a candidate: BM25 (FTS5 keyword),
 * nomic (prose vector space), or jina-code (code vector space).
 */
export type RetrievalSource = "bm25" | "nomic" | "jina-code";

/** A chunk plus its individual and blended relevance scores. */
export interface ScoredChunk {
  chunk: Chunk;
  bm25: number;
  vector: number;
  hybrid: number;
  /** Every engine that returned this chunk as a candidate. */
  sources: RetrievalSource[];
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
 * Result-selection quotas for the dual-space policy: 5 code hits + 5 prose
 * hits when both groups have qualifying hits; 5 of the single group
 * otherwise. Exported for tests.
 */
export const CODE_RESULT_QUOTA = 5;
export const PROSE_RESULT_QUOTA = 5;
export const SINGLE_GROUP_RESULT_QUOTA = 5;

/**
 * Run hybrid search over the index: BM25 via FTS5 plus vector KNN over
 * both embedding spaces, blended as `alpha * bm25 + (1 - alpha) * vector`
 * (alpha = 1 → pure BM25 when no vectors exist). Each model embeds the
 * query only when its own vector space has stored vectors — a space with
 * no vectors is skipped entirely (no query embedding, no KNN).
 *
 * Result selection is a fixed quota split: both groups present → up to 5
 * jina-code hits, then up to 5 nomic/bm25 hits (code first); a single
 * group → up to 5 of that group. `limit` caps each group's quota, not the
 * combined total.
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
  // Each model's query is only embedded when its table actually has
  // vectors (skips loading a model the store can't use); when both
  // spaces are populated, both queries are embedded in parallel and both
  // spaces are searched.
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

  // Attribute each candidate to the engine(s) that surfaced it, so
  // consumers (e.g. the auto-injection summary) can report provenance.
  const sourcesByRowid = new Map<number, RetrievalSource[]>();
  const attributeSource = (rowid: number, source: RetrievalSource) => {
    const attributed = sourcesByRowid.get(rowid) ?? [];
    if (!attributed.includes(source)) attributed.push(source);
    sourcesByRowid.set(rowid, attributed);
  };
  for (const match of ftsResults) attributeSource(match.rowid, "bm25");
  for (const match of textVectorResults) attributeSource(match.rowid, "nomic");
  for (const match of codeVectorResults) attributeSource(match.rowid, "jina-code");

  const chunkRows = sqlRepository.getChunksByRowids(database, Array.from(candidateRowIds));
  const chunkByRowid = new Map<number, typeof chunkRows[0]>();
  for (const chunkRow of chunkRows) chunkByRowid.set(chunkRow.rowid, chunkRow);

  // Min-max normalize BM25 scores across the FTS candidate set (constant 1
  // when every candidate ties, so ties stay rankable rather than zeroed).
  // FTS5's bm25() is lower-is-better (more negative = better match), so
  // scores are negated first — the normalization and the hybrid blend both
  // assume bigger-is-better.
  const bm25NormalizedByRowid = new Map<number, number>();
  if (ftsResults.length > 0) {
    let bm25Max = -Infinity;
    let bm25Min = Infinity;
    for (const match of ftsResults) {
      const bm25Score = -match.bm25_score;
      if (bm25Score > bm25Max) bm25Max = bm25Score;
      if (bm25Score < bm25Min) bm25Min = bm25Score;
    }
    const bm25Range = bm25Max - bm25Min;
    for (const match of ftsResults) {
      const bm25Score = -match.bm25_score;
      bm25NormalizedByRowid.set(
        match.rowid,
        bm25Range === 0 ? 1 : (bm25Score - bm25Min) / bm25Range,
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
      sources: sourcesByRowid.get(rowid) ?? [],
    });
  }

  // Result selection is a fixed per-group quota split: when both code and
  // prose hits qualify (hybrid > 0), the result is up to 5 jina-code hits
  // followed by up to 5 nomic/bm25 hits (code group first — a code hit
  // always outranks a prose hit); when only one group has hits, that group
  // fills up to 5 slots. Chunks found only by BM25 rank with the prose
  // group. Within a group, order is by hybrid score.
  //
  // `limit` caps each group's quota (never the combined total — capping
  // the total at the default topK of 5 would crowd prose back out, the
  // exact failure the quota split exists to prevent).
  const isCodeHit = (scored: ScoredChunk) => scored.sources.includes("jina-code");
  const ranked = scoredResults
    .filter(scored => scored.hybrid > 0)
    .sort((a, b) => b.hybrid - a.hybrid);
  const codeHits = ranked.filter(isCodeHit);
  const proseHits = ranked.filter(scored => !isCodeHit(scored));

  if (codeHits.length > 0 && proseHits.length > 0) {
    return [
      ...codeHits.slice(0, Math.min(CODE_RESULT_QUOTA, limit)),
      ...proseHits.slice(0, Math.min(PROSE_RESULT_QUOTA, limit)),
    ];
  }
  return (codeHits.length > 0 ? codeHits : proseHits)
    .slice(0, Math.min(SINGLE_GROUP_RESULT_QUOTA, limit));
}
