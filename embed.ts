import { existsSync } from "node:fs";
import { join } from "node:path";
import { homedir } from "node:os";
import {
  EMBEDDING_MODEL, QUERY_PREFIX, DOC_PREFIX,
  CODE_EMBEDDING_MODEL, CODE_QUERY_PREFIX, CODE_DOC_PREFIX,
  type EmbedGroup,
} from "./constants.ts";

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
 * `_quantized` suffix) — if the weights file exists, loading the pipeline is
 * fast local I/O; if not, first use triggers the ~111 MB / ~170 MB download.
 */
function isModelCached(modelId: string): boolean {
  return existsSync(join(resolveModelCacheDir(), modelId, "onnx", "model_quantized.onnx"));
}

// One pipeline per group — nomic for prose, jina for code. The load promise
// is cached (not the resolved pipeline) so concurrent first calls share a
// single download; a failed load is evicted so the next call retries.
const _pipelines = new Map<EmbedGroup, Promise<any>>();

export async function getEmbedder(group: EmbedGroup = "text"): Promise<any> {
  const cached = _pipelines.get(group);
  if (cached) return cached;
  const spec = EMBED_MODELS[group];
  const load = (async () => {
    const { pipeline, env } = await import("@huggingface/transformers");
    env.cacheDir = resolveModelCacheDir();
    // q8 = quantized ONNX weights (~111 MB nomic, ~170 MB jina-code, vs
    // ~547/~650 MB fp32) — keeps first-run downloads reasonable with
    // negligible quality loss.
    return pipeline("feature-extraction", spec.id, { dtype: "q8" });
  })();
  _pipelines.set(group, load);
  load.catch(() => _pipelines.delete(group));
  return load;
}

/** Embed a search query with a specific group's model. */
export async function embedQueryFor(group: EmbedGroup, text: string): Promise<number[]> {
  const spec = EMBED_MODELS[group];
  const embedder = await getEmbedder(group);
  const output = await embedder(spec.queryPrefix + text, { pooling: "mean", normalize: true });
  return Array.from(output.data as Float32Array);
}

/** Embed a search query with the text/prose model (nomic `search_query:` prefix). */
export async function embed(text: string): Promise<number[]> {
  return embedQueryFor("text", text);
}

/**
 * Yield to the event loop so the TUI can render progress updates.
 * ONNX inference is synchronous from the event loop's perspective;
 * without this, the UI freezes during embedding.
 */
const yield_ = () => new Promise<void>(r => setTimeout(r, 0));

/** Default batch size for a single ONNX forward pass. */
export const BATCH_SIZE = 64;

export interface EmbedBatchOpts {
  onProgress?: (done: number, total: number) => void;
  /** Fired right before a group's model is downloaded — only when the ONNX
   *  weights are not already in the local HF cache — lets the UI explain a
   *  multi-minute cold-start instead of looking stuck. */
  onModelLoad?: (modelId: string) => void;
}

/**
 * Embed `texts` with a specific group's model using true batched ONNX
 * inference — one forward pass per BATCH_SIZE texts (~BATCH_SIZE× speedup
 * on CPU). The output Tensor has dims [batchSize, dim]; sliced per-text.
 *
 * `onProgress` fires after each batch with the cumulative count so the TUI
 * can render a smooth progress bar; `onModelLoad` fires once per model.
 */
export async function embedBatchFor(
  group: EmbedGroup,
  texts: string[],
  opts: EmbedBatchOpts = {},
): Promise<number[][]> {
  if (texts.length === 0) return [];
  // Notify only on a true cold start: pipeline not yet loaded in this
  // process AND weights missing from the on-disk cache. A warm disk cache
  // loads in seconds, so the "downloading" notice would be noise.
  if (!_pipelines.has(group) && !isModelCached(EMBED_MODELS[group].id)) {
    opts.onModelLoad?.(EMBED_MODELS[group].id);
  }
  const spec = EMBED_MODELS[group];
  const embedder = await getEmbedder(group);
  const results: number[][] = new Array(texts.length);

  for (let start = 0; start < texts.length; start += BATCH_SIZE) {
    const batch = texts.slice(start, start + BATCH_SIZE);
    // Pass the whole batch in a single forward pass — the model returns a
    // Tensor with dims [batchSize, dim]. Documents get the group's doc
    // prefix (nomic task instruction; jina v2 uses none).
    const output = await embedder(batch.map(t => spec.docPrefix + t), { pooling: "mean", normalize: true });
    const flat = output.data as Float32Array;
    const dim = flat.length / batch.length; // 768 for both current models

    for (let j = 0; j < batch.length; j++) {
      results[start + j] = Array.from(flat.subarray(j * dim, (j + 1) * dim));
    }

    opts.onProgress?.(Math.min(start + batch.length, texts.length), texts.length);
    // Yield after each batch so the TUI can re-render before the next pass.
    await yield_();
  }

  return results;
}

/** Back-compat: batch-embed with the text/prose model. */
export async function embedBatch(
  texts: string[],
  onProgress?: (i: number, total: number) => void,
): Promise<number[][]> {
  return embedBatchFor("text", texts, { onProgress });
}
