/**
 * Embedding tests — exercise the real local ONNX pipelines.
 *
 * Two models are fetched from HuggingFace by Transformers.js on first use:
 * nomic-ai/nomic-embed-text-v1.5 (~111 MB quantized, text group) and
 * jinaai/jina-embeddings-v2-base-code (~170 MB quantized, code group).
 * Subsequent runs read from the Transformers.js cache
 * (~/.cache/huggingface/...). No fixture data is bundled with the repo.
 *
 * This file lives separately from __tests__/index.test.ts because that file
 * mocks @huggingface/transformers at module scope (to keep the SQLite-era
 * hybridSearch tests fast and deterministic). The mock returns a constant
 * 768-dim vector, which trivially fails normalization + semantic-similarity
 * checks. Splitting matches the upstream fork's layout — kallewoof's
 * __tests__/embedding.test.ts has no mock; __tests__/index.test.ts does.
 *
 * Set SKIP_EMBEDDING_TESTS=1 to skip (e.g. in offline CI).
 */
import { describe, it, expect, afterEach } from "vitest";
import Database from "better-sqlite3";
import { load as loadVec } from "sqlite-vec";
import {
  embed, embedQueryFor, cosineSimilarity, hybridSearch, sha256, initSchema,
} from "../index.ts";

const skip = process.env.SKIP_EMBEDDING_TESTS === "1";
// Generous: the first call includes the ~111 MB model download + ONNX
// session init; later calls reuse the cached pipeline.
const EMBED_TIMEOUT = 300_000;

// Close the cached DB singleton after every test so it can't leak into the next test
afterEach(async () => {
  const { closeDbConn } = await import("../db.ts");
  closeDbConn();
});

describe("embed (real ONNX)", () => {
  it.skipIf(skip)("returns a 768-dim unit-normalized vector for a single string", async () => {
    const v = await embed("hello world");
    expect(Array.isArray(v)).toBe(true);
    expect(v.length).toBe(768);
    const norm = Math.sqrt(v.reduce((s, x) => s + x * x, 0));
    expect(Math.abs(norm - 1)).toBeLessThan(1e-3);
    expect(v.some(x => x !== 0)).toBe(true);
  }, EMBED_TIMEOUT);

  it.skipIf(skip)("deterministic — same input produces same output", async () => {
    const a = await embed("the quick brown fox jumps over the lazy dog");
    const b = await embed("the quick brown fox jumps over the lazy dog");
    expect(a.length).toBe(b.length);
    for (let i = 0; i < a.length; i++) {
      expect(Math.abs(a[i] - b[i])).toBeLessThan(1e-6);
    }
  }, EMBED_TIMEOUT);

  it.skipIf(skip)("semantic similarity — related sentences are closer than unrelated ones", async () => {
    const cat = await embed("A cat sits on the windowsill watching birds.");
    const kitten = await embed("A small kitten is looking at sparrows through the window.");
    const finance = await embed("Quarterly revenue exceeded analyst expectations by twelve percent.");
    const simRelated = cosineSimilarity(cat, kitten);
    const simUnrelated = cosineSimilarity(cat, finance);
    expect(simRelated).toBeGreaterThan(simUnrelated + 0.1);
    expect(simRelated).toBeGreaterThan(0.5);
  }, EMBED_TIMEOUT);

  it.skipIf(skip)("hybridSearch: vector path retrieves semantically relevant chunks even without keyword overlap", async () => {
    // Build an in-memory DB (matches createTestDb from index.test.ts but
    // populates with REAL embeddings — this test specifically validates the
    // semantic vector path end-to-end through sqlite-vec.
    const chunks = [
      { content: "Photosynthesis is how plants convert sunlight into chemical energy.", file: "plants.md" },
      { content: "The team shipped a new dashboard for analytics reporting.", file: "shipping.md" },
      { content: "We pickled cucumbers in a vinegar brine with dill and garlic.", file: "recipe.md" },
    ];
    const vectors = await Promise.all(chunks.map(c => embed(c.content)));

    const db = new Database(":memory:");
    db.pragma("journal_mode = WAL");
    loadVec(db);
    initSchema(db);

    const insChunk = db.prepare(`
      INSERT INTO chunks(id, file_path, chunk_content, line_start, line_end, chunk_hash, indexed_at, tokens)
      VALUES (?, ?, ?, ?, ?, ?, ?, ?)
    `);
    const insVec = db.prepare("INSERT INTO chunks_vec(rowid, embedding) VALUES (CAST(? AS INTEGER), ?)");
    for (let i = 0; i < chunks.length; i++) {
      const c = chunks[i];
      const r = insChunk.run(
        `${c.file}-1`, c.file, c.content, 1, 1, sha256(c.content),
        "2026-05-15T00:00:00Z", Math.ceil(c.content.length / 4),
      );
      const f = new Float32Array(vectors[i]);
      insVec.run(Number(r.lastInsertRowid), Buffer.from(f.buffer, f.byteOffset, f.byteLength));
    }

    const results = await hybridSearch(
      "How do leaves produce food from light?",
      3, 0, db,
    );
    db.close();

    expect(results.length).toBeGreaterThan(0);
    expect(results[0].chunk.file).toBe("plants.md");
  }, EMBED_TIMEOUT);
});

describe("embedQueryFor('code') (real ONNX, jina-embeddings-v2-base-code)", () => {
  it.skipIf(skip)("returns a 768-dim unit-normalized vector for a single string", async () => {
    const v = await embedQueryFor("code", "export function parseJson(text: string): unknown {}");
    expect(Array.isArray(v)).toBe(true);
    expect(v.length).toBe(768);
    const norm = Math.sqrt(v.reduce((s, x) => s + x * x, 0));
    expect(Math.abs(norm - 1)).toBeLessThan(1e-3);
    expect(v.some(x => x !== 0)).toBe(true);
  }, EMBED_TIMEOUT);

  it.skipIf(skip)("code relevance — related snippets are closer than unrelated prose", async () => {
    const auth = await embedQueryFor("code", "export async function authenticate(user: string, password: string): Promise<Token> {");
    const authVariant = await embedQueryFor("code", "function login(credentials: UserPass) { return verifyPassword(credentials); }");
    const prose = await embedQueryFor("code", "The quarterly marketing report highlights customer retention improvements.");
    const simRelated = cosineSimilarity(auth, authVariant);
    const simUnrelated = cosineSimilarity(auth, prose);
    expect(simRelated).toBeGreaterThan(simUnrelated);
  }, EMBED_TIMEOUT);

  it.skipIf(skip)("dual-space hybridSearch retrieves from both the code and the text vector tables", async () => {
    const textChunks = [
      { content: "Photosynthesis is how plants convert sunlight into chemical energy.", file: "plants.md" },
    ];
    const codeChunks = [
      { content: "export function authenticate(user: string, password: string): Promise<Token> { return verify(user, password); }", file: "auth.ts" },
      { content: "export function renderTemplate(html: string, ctx: Context): string { return interpolate(html, ctx); }", file: "render.ts" },
    ];

    // Text chunks → nomic space (chunks_vec); code chunks → jina space
    // (chunks_vec_code). Mirrors what indexFiles does per extension group.
    const db = new Database(":memory:");
    db.pragma("journal_mode = WAL");
    loadVec(db);
    initSchema(db);

    const insChunk = db.prepare(`
      INSERT INTO chunks(id, file_path, chunk_content, line_start, line_end, chunk_hash, indexed_at, tokens)
      VALUES (?, ?, ?, ?, ?, ?, ?, ?)
    `);
    const insVec = db.prepare("INSERT INTO chunks_vec(rowid, embedding) VALUES (CAST(? AS INTEGER), ?)");
    const insCodeVec = db.prepare("INSERT INTO chunks_vec_code(rowid, embedding) VALUES (CAST(? AS INTEGER), ?)");
    const insert = async (c: { content: string; file: string }, vector: number[], codeVec: boolean) => {
      const r = insChunk.run(
        `${c.file}-1`, c.file, c.content, 1, 1, sha256(c.content),
        "2026-05-15T00:00:00Z", Math.ceil(c.content.length / 4),
      );
      const f = new Float32Array(vector);
      (codeVec ? insCodeVec : insVec).run(Number(r.lastInsertRowid), Buffer.from(f.buffer, f.byteOffset, f.byteLength));
    };

    // Embed docs with the same embedding call the indexer uses for each
    // group: nomic search_document: prefix for prose, jina (no prefix) for code.
    const { embedBatchFor } = await import("../embed.ts");
    const textVectors = await embedBatchFor("text", textChunks.map(c => c.content));
    const codeVectors = await embedBatchFor("code", codeChunks.map(c => c.content));

    for (let i = 0; i < textChunks.length; i++) await insert(textChunks[i], textVectors[i], false);
    for (let i = 0; i < codeChunks.length; i++) await insert(codeChunks[i], codeVectors[i], true);

    // Pure-vector search (alpha=0): the code query must hit the jina space,
    // the prose query the nomic space.
    const codeHit = await hybridSearch("how to verify a user login and password", 3, 0, db);
    expect(codeHit.length).toBeGreaterThan(0);
    expect(codeHit[0].chunk.file).toBe("auth.ts");

    const proseHit = await hybridSearch("How do leaves produce food from light?", 3, 0, db);
    expect(proseHit.length).toBeGreaterThan(0);
    expect(proseHit[0].chunk.file).toBe("plants.md");

    db.close();
  }, EMBED_TIMEOUT);
});
