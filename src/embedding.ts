/**
 * Dual ONNX embedding pipelines via Transformers.js: one pipeline per
 * embedding group — nomic-embed-text-v1.5 for prose, jina-embeddings-v2-
 * base-code for source code. Model downloads are cached in a shared
 * HuggingFace cache directory so they happen once per machine.
 */
import { existsSync } from "node:fs";
import { join, basename } from "node:path";
import { homedir } from "node:os";
import {
  EMBEDDING_MODEL,
  QUERY_PREFIX,
  DOC_PREFIX,
  CODE_EMBEDDING_MODEL,
  CODE_QUERY_PREFIX,
  CODE_DOC_PREFIX,
  type EmbedGroup,
} from "./constants.ts";
import { yieldToEventLoop } from "./runtime-utils.ts";

/** Everything the pipeline needs to know about one embedding model. */
interface ModelSpec {
  id: string;
  queryPrefix: string;
  docPrefix: string;
}

/** Per-group model registry: text → nomic, code → jina. */
export const EMBED_MODELS: Record<EmbedGroup, ModelSpec> = {
  text: { id: EMBEDDING_MODEL, queryPrefix: QUERY_PREFIX, docPrefix: DOC_PREFIX },
  code: { id: CODE_EMBEDDING_MODEL, queryPrefix: CODE_QUERY_PREFIX, docPrefix: CODE_DOC_PREFIX },
};

/**
 * Normalize a query before embedding: trim + collapse runs of whitespace to
 * a single space. Queries come straight from the user's prompt and can
 * carry stray newlines/indentation that dilute the embedding (and,
 * downstream, the FTS tokenization).
 */
export function cleanQuery(text: string): string {
  return text.replace(/\s+/g, " ").trim();
}

/**
 * The exact input text a model expects for a QUERY, encoding each model's
 * special capability:
 *
 * - text (nomic-embed-text-v1.5): asymmetric retrieval — queries get the
 *   `search_query:` task prefix, documents `search_document:`. These two
 *   are deliberately different so nomic can place short questions and long
 *   passages in aligned-but-distinct regions of its space.
 * - code (jina-embeddings-v2-base-code): symmetric, no task prefix (jina v2
 *   added none — those arrived with v3). Its code-search training maps
 *   natural language straight to code, so the raw question is already the
 *   ideal query.
 */
export function buildQueryInput(group: EmbedGroup, text: string): string {
  return EMBED_MODELS[group].queryPrefix + cleanQuery(text);
}

/**
 * The exact input text a model expects for a DOCUMENT chunk, encoding each
 * model's special capability:
 *
 * - text (nomic): `search_document:` prefix; prose is self-describing, so
 *   no extra context is injected.
 * - code (jina): no prefix, but the file basename is prepended as a context
 *   line. jina-code was trained on code-with-context pairs (docstring/
 *   question → code), and a bare ~50-line slice loses its file identity
 *   otherwise; prepending the basename anchors filename-oriented queries
 *   ("what does auth.ts do?") into the vector space without touching the
 *   stored chunk content or FTS text.
 */
export function buildDocumentInput(group: EmbedGroup, text: string, filename?: string): string {
  const modelSpec = EMBED_MODELS[group];
  const content = group === "code" && filename ? `${basename(filename)}\n${text}` : text;
  return modelSpec.docPrefix + content;
}

/**
 * Persistent HuggingFace model cache directory.
 *
 * Transformers.js in Node defaults `env.cacheDir` to `./.cache` relative to
 * the process cwd — the ~111 MB nomic download would be repeated for every
 * project the agent runs in. Pin it to one shared location, honoring the
 * standard HF env vars with a pi-local-rag-specific override.
 *
 * Priority: PI_RAG_MODEL_CACHE > TRANSFORMERS_CACHE > HF_HOME/transformers >
 * ~/.cache/huggingface/transformers.
 */
export function resolveModelCacheDir(): string {
  if (process.env.PI_RAG_MODEL_CACHE) return process.env.PI_RAG_MODEL_CACHE;
  if (process.env.TRANSFORMERS_CACHE) return process.env.TRANSFORMERS_CACHE;
  if (process.env.HF_HOME) return join(process.env.HF_HOME, "transformers");
  return join(homedir(), ".cache", "huggingface", "transformers");
}

/**
 * Whether a group's q8 ONNX weights are already present in the local model
 * cache. Transformers.js stores files under
 * `<cacheDir>/<org>/<model>/onnx/model_quantized.onnx` (q8 dtype adds the
 * `_quantized` suffix) — if the weights file exists, loading the pipeline
 * is fast local I/O; if not, first use triggers the ~111 MB / ~170 MB
 * download.
 */
function isModelCachedInHfCache(modelId: string): boolean {
  return existsSync(join(resolveModelCacheDir(), modelId, "onnx", "model_quantized.onnx"));
}

/**
 * One pipeline per group — nomic for prose, jina for code. The load promise
 * is cached (not the resolved pipeline) so concurrent first calls share a
 * single download; a failed load is evicted so the next call retries.
 */
const pipelineLoadPromises = new Map<EmbedGroup, Promise<unknown>>();

/** Get (or lazily start loading) the embedding pipeline for one group. */
export async function getEmbedder(group: EmbedGroup = "text"): Promise<any> {
  const inFlightLoad = pipelineLoadPromises.get(group);
  if (inFlightLoad) return inFlightLoad;
  const modelSpec = EMBED_MODELS[group];
  const loadPromise = (async () => {
    const { pipeline, env } = await import("@huggingface/transformers");
    env.cacheDir = resolveModelCacheDir();
    // q8 = quantized ONNX weights (~111 MB nomic, ~170 MB jina-code, vs
    // ~547/~650 MB fp32) — keeps first-run downloads reasonable with
    // negligible quality loss.
    return pipeline("feature-extraction", modelSpec.id, { dtype: "q8" });
  })();
  pipelineLoadPromises.set(group, loadPromise);
  loadPromise.catch(() => pipelineLoadPromises.delete(group));
  return loadPromise;
}

/** Embed a search query with a specific group's model (returns number[]). */
export async function embedQueryFor(group: EmbedGroup, text: string): Promise<number[]> {
  const embedder = await getEmbedder(group);
  const output = await embedder(buildQueryInput(group, text), { pooling: "mean", normalize: true });
  return Array.from(output.data as Float32Array);
}

/** Embed a search query with the text/prose model (nomic `search_query:` prefix). */
export async function embed(text: string): Promise<number[]> {
  return embedQueryFor("text", text);
}

/**
 * Default number of texts per single ONNX forward pass. Kept moderate: a
 * batch is padded to its longest member, so oversized batches of long code
 * chunks turn into one multi-minute pass with no progress tick until it
 * completes (the run looks stuck). 16 bounds per-pass wall time while still
 * amortizing tokenizer/session overhead.
 */
export const BATCH_SIZE = 16;

/** Progress/callback hooks for batch embedding. */
export interface EmbedBatchOpts {
  /** Fired after each ONNX batch with the cumulative count. */
  onProgress?: (done: number, total: number) => void;
  /**
   * Fired right before a group's model is downloaded — only when the ONNX
   * weights are not already in the local HF cache — lets the UI explain a
   * multi-minute cold-start instead of looking stuck.
   */
  onModelLoad?: (modelId: string) => void;
}

/**
 * Embed `texts` with a specific group's model using true batched ONNX
 * inference — one forward pass per BATCH_SIZE texts (~BATCH_SIZE× speedup
 * on CPU). The output Tensor has dims [batchSize, dim]; sliced per-text.
 *
 * Vectors come back as Float32Array (4 B/element) — half the memory of a
 * boxed number[], and zero-copy into sqlite-vec's float blob at insert
 * time. For a 14k-chunk store that's ~100 MB of retained JS heap → ~44 MB
 * of external backing stores held between the embed phase and the DB write.
 *
 * `fileNames` (parallel to `texts`) supplies the filename-context header
 * for code chunks; `onProgress` fires after each batch so the TUI can
 * render a smooth progress bar; `onModelLoad` fires once per model.
 */
export async function embedBatchFor(
  group: EmbedGroup,
  texts: string[],
  opts: EmbedBatchOpts = {},
  fileNames?: (string | undefined)[],
): Promise<Float32Array[]> {
  if (texts.length === 0) return [];

  // Notify only on a true cold start: pipeline not yet loaded in this
  // process AND weights missing from the on-disk cache. A warm disk cache
  // loads in seconds, so the "downloading" notice would be noise.
  if (!pipelineLoadPromises.has(group) && !isModelCachedInHfCache(EMBED_MODELS[group].id)) {
    opts.onModelLoad?.(EMBED_MODELS[group].id);
  }

  const embedder = await getEmbedder(group);
  const resultVectors: Float32Array[] = new Array(texts.length);

  for (let batchStart = 0; batchStart < texts.length; batchStart += BATCH_SIZE) {
    const batch = texts.slice(batchStart, batchStart + BATCH_SIZE);
    // Pass the whole batch in a single forward pass — the model returns a
    // Tensor with dims [batchSize, dim]. Documents get the group's doc
    // prefix (nomic task instruction; jina v2 uses none) plus, for code
    // chunks, the file basename as a context header.
    const output = await embedder(
      batch.map((text, indexInBatch) =>
        buildDocumentInput(group, text, fileNames?.[batchStart + indexInBatch]),
      ),
      { pooling: "mean", normalize: true },
    );
    const flattened = output.data as Float32Array;
    const dimensionsPerVector = flattened.length / batch.length; // 768 for both current models

    for (let vectorIndex = 0; vectorIndex < batch.length; vectorIndex++) {
      // slice(), not subarray() — subarray would alias the tensor's backing
      // store, pinning the whole [batch × dim] buffer while any one of its
      // vectors is still unwritten.
      resultVectors[batchStart + vectorIndex] =
        flattened.slice(vectorIndex * dimensionsPerVector, (vectorIndex + 1) * dimensionsPerVector);
    }

    opts.onProgress?.(Math.min(batchStart + batch.length, texts.length), texts.length);
    // Yield after each batch so the TUI can re-render before the next pass.
    await yieldToEventLoop();
  }

  return resultVectors;
}
