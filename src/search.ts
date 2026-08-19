/**
 * Hybrid search: SQLite FTS5 (BM25) + sqlite-vec (vector KNN) over two
 * independent embedding spaces (prose/nomic and code/jina), merged with a
 * per-query blend weight. Each space's query is embedded only when that
 * space has stored vectors; results are a ratio-based quota split — the
 * total (7 when both spaces store vectors, else 5, capped by `limit`) is
 * divided between code and prose in proportion to each space's stored
 * vector count, integer quotas with a minimum of 1 per group when both
 * qualify.
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
 * Total number of result slots when at most one embedding space has
 * stored vectors (or a single group fills the result alone). `limit` can
 * only shrink this, never grow it.
 */
export const RESULT_TOTAL_QUOTA = 5;

/**
 * Result-slot total for stores with vectors in both embedding spaces —
 * a mixed corpus has two groups to fill, so more hits feed the
 * ratio-based quota split. Same rule as RESULT_TOTAL_QUOTA otherwise:
 * `limit` can only shrink it.
 */
export const RESULT_TOTAL_QUOTA_DUAL_SPACE = 7;

/**
 * Split `total` result slots between the code and prose groups in
 * proportion to the store's stored vector counts (the corpus
 * composition ratio). Integer quotas that sum exactly to `total`, each
 * at least 1 — except `total < 2`, where both minimums can't hold and
 * the single slot goes to the code group (code-first policy). Exported
 * for tests.
 */
export function splitResultQuotas(
  total: number,
  codeVectorCount: number,
  proseVectorCount: number,
): { codeQuota: number; proseQuota: number } {
  if (total < 2) return { codeQuota: total, proseQuota: 0 };
  const vectorTotal = codeVectorCount + proseVectorCount;
  const codeShare = vectorTotal > 0 ? codeVectorCount / vectorTotal : 0.5;
  const codeQuota = Math.min(total - 1, Math.max(1, Math.round(total * codeShare)));
  return { codeQuota, proseQuota: total - codeQuota };
}

/**
 * Run hybrid search over the index: BM25 via FTS5 plus vector KNN over
 * both embedding spaces, blended as `alpha * bm25 + (1 - alpha) * vector`
 * (alpha = 1 → pure BM25 when no vectors exist). Each model embeds the
 * query only when its own vector space has stored vectors — a space with
 * no vectors is skipped entirely (no query embedding, no KNN).
 *
 * Result selection is a ratio-based quota split: the result totals
 * RESULT_TOTAL_QUOTA_DUAL_SPACE (7) when both spaces store vectors,
 * else RESULT_TOTAL_QUOTA (5) — either way capped by `limit` — and is
 * split between code and prose proportionally to each space's stored
 * vector count — integer quotas summing exactly to the total, at least
 * 1 per group when both qualify, code group first. A group that can't
 * fill its quota yields the slack to the other group's next-best hits;
 * when only one group qualifies it takes the whole total.
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
  // spaces are searched. The per-space counts also drive the ratio-based
  // result quota split below.
  const vectorCandidateLimit = Math.max(limit * 10, 100);
  const proseVectorCount = sqlRepository.getEmbeddedCount(database);
  const codeVectorCount = sqlRepository.getCodeEmbeddedCount(database);
  const [textVectorResults, codeVectorResults] = await Promise.all([
    (async () =>
      proseVectorCount
        ? sqlRepository.searchVectors(database, await embedQueryFor("text", query), vectorCandidateLimit)
        : [])(),
    (async () =>
      codeVectorCount
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

  // Result selection is a ratio-based quota split: the result totals
  // RESULT_TOTAL_QUOTA_DUAL_SPACE (7) when both spaces store vectors,
  // else RESULT_TOTAL_QUOTA (5) — capped by `limit` either way —
  // divided between the code
  // and prose groups in proportion to the store's per-space vector
  // counts — integer quotas summing exactly to the total, at least 1 per
  // group when both qualify, code group first (a code hit outranks a
  // prose hit at equal quota rank). A group with fewer qualifying hits
  // than its quota yields the slack to the other group's next-best hits
  // (best hybrid first), so the total stays filled. Chunks found only by
  // BM25 rank with the prose group. Within a group, order is by hybrid
  // score. When only one group qualifies, it takes the whole total.
  const isCodeHit = (scored: ScoredChunk) => scored.sources.includes("jina-code");
  const ranked = scoredResults
    .filter(scored => scored.hybrid > 0)
    .sort((a, b) => b.hybrid - a.hybrid);
  const codeHits = ranked.filter(isCodeHit);
  const proseHits = ranked.filter(scored => !isCodeHit(scored));

  const bothSpacesHaveVectors = codeVectorCount > 0 && proseVectorCount > 0;
  const total = Math.min(
    limit,
    bothSpacesHaveVectors ? RESULT_TOTAL_QUOTA_DUAL_SPACE : RESULT_TOTAL_QUOTA,
  );

  if (codeHits.length > 0 && proseHits.length > 0) {
    const { codeQuota, proseQuota } = splitResultQuotas(total, codeVectorCount, proseVectorCount);
    const codePrimary = codeHits.slice(0, codeQuota);
    const prosePrimary = proseHits.slice(0, proseQuota);
    const shortfall = total - (codePrimary.length + prosePrimary.length);
    let codeExtra: ScoredChunk[] = [];
    let proseExtra: ScoredChunk[] = [];
    if (shortfall > 0) {
      const filler = [
        ...codeHits.slice(codePrimary.length),
        ...proseHits.slice(prosePrimary.length),
      ]
        .sort((a, b) => b.hybrid - a.hybrid)
        .slice(0, shortfall);
      codeExtra = filler.filter(isCodeHit);
      proseExtra = filler.filter(scored => !isCodeHit(scored));
    }
    return [...codePrimary, ...codeExtra, ...prosePrimary, ...proseExtra];
  }
  return (codeHits.length > 0 ? codeHits : proseHits).slice(0, total);
}
