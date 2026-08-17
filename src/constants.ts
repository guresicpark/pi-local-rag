/**
 * Shared constants for the whole pi-local-rag pipeline.
 *
 * Everything that other modules must agree on lives here: embedding model
 * identities, task prefixes, file-extension groups, size limits, chunking
 * caps, and the ANSI color escapes used by stderr progress lines and TUI
 * widgets in callers that lack access to `ctx.ui.theme`.
 */

// ── ANSI color escapes ───────────────────────────────────────────────────────

/** Reset all text attributes. */
export const ANSI_RESET = "\x1b[0m";
/** Bold text. */
export const ANSI_BOLD = "\x1b[1m";
/** Dim (faint) text. */
export const ANSI_DIM = "\x1b[2m";
/** Green text — used for success values in progress widgets. */
export const ANSI_GREEN = "\x1b[32m";
/** Cyan text — used for progress-bar fills. */
export const ANSI_CYAN = "\x1b[36m";

// ── Embedding groups ─────────────────────────────────────────────────────────

/**
 * Embedding group: code files go to the dedicated code model, everything
 * else (prose, markup, data/config, PDF/DOCX) to the text model.
 */
export type EmbedGroup = "code" | "text";

// ── Text/prose model: nomic-ai/nomic-embed-text-v1.5 ─────────────────────────

/** HuggingFace id of the text/prose embedding model. */
export const EMBEDDING_MODEL = "nomic-ai/nomic-embed-text-v1.5";
/** Dimensionality of the text-model embedding space. */
export const VECTOR_DIM = 768;
/**
 * nomic-embed-text-v1.5 requires task-type prefixes on inputs (see the model
 * card). Queries and indexed documents must use different prefixes so the
 * model can place short questions and long passages in aligned-but-distinct
 * regions of its space (asymmetric retrieval).
 */
export const QUERY_PREFIX = "search_query: ";
export const DOC_PREFIX = "search_document: ";

// ── Code model: jina-embeddings-v2-base-code ─────────────────────────────────

/**
 * HuggingFace id of the code embedding model. The official repo ships the
 * ONNX weights itself — tagged transformers.js-compatible, including the
 * quantized (q8) file. 768-dim, trained on code + docstring pairs.
 */
export const CODE_EMBEDDING_MODEL = "jinaai/jina-embeddings-v2-base-code";
/** Dimensionality of the code-model embedding space. */
export const CODE_VECTOR_DIM = 768;
/**
 * jina v2 models take NO task prefixes (those arrived with v3), so the
 * prefix constants are empty strings. Its code-search training maps natural
 * language straight onto code.
 */
export const CODE_QUERY_PREFIX = "";
export const CODE_DOC_PREFIX = "";

/**
 * Version of the code-document embedding *scheme* — i.e. how a chunk's text
 * is prepared before it goes into the code model (not which model). Bumping
 * this invalidates existing code vectors so they are re-embedded with the
 * new scheme, the same way a model-id change does. Prose vectors (nomic)
 * are unaffected.
 */
export const CODE_EMBED_SCHEME = "file-context-v1";

// ── File-extension groups ────────────────────────────────────────────────────

/** Code extensions → embedded with CODE_EMBEDDING_MODEL (jina). */
export const DEFAULT_CODE_EXTS = [
  ".ts", ".tsx", ".js", ".jsx", ".mjs", ".cjs",
  ".py", ".rs", ".go", ".java", ".kt", ".kts", ".scala",
  ".c", ".cc", ".cpp", ".cxx", ".h", ".hpp", ".hxx",
  ".cs", ".fs", ".vb",
  ".swift", ".m", ".mm",
  ".rb", ".php", ".pl", ".lua", ".dart", ".ex", ".exs", ".erl", ".clj", ".cljs", ".edn",
  ".vue", ".svelte", ".astro", ".twig",
  ".css", ".scss", ".sass", ".less",
  ".sh", ".bash", ".zsh", ".fish", ".ps1",
  ".sql", ".graphql", ".gql", ".proto",
  ".tf", ".hcl",
];

/**
 * Text/prose + data/config extensions → embedded with EMBEDDING_MODEL
 * (nomic). HTML/HTM land here too: extractText() converts them to markdown
 * prose before chunking. PDF/DOCX (BINARY_DOC_EXTS) are also text-group.
 */
export const DEFAULT_DOC_EXTS = [
  ".md", ".mdx", ".txt", ".rst",
  ".html", ".htm",
  ".json", ".jsonc", ".yaml", ".yml", ".toml", ".ini", ".xml", ".csv", ".tsv",
  ".env", ".gitignore", ".dockerfile",
];

/**
 * Union of both groups — the full default allowlist. Callers use this to
 * answer "is this file indexable at all".
 */
export const DEFAULT_TEXT_EXTS = [...DEFAULT_CODE_EXTS, ...DEFAULT_DOC_EXTS];

/**
 * Binary document extensions routed through dedicated extraction libraries
 * (unpdf for PDFs, mammoth for DOCX) instead of being read as UTF-8. They
 * belong to the text embedding group.
 */
export const BINARY_DOC_EXTS = new Set([".pdf", ".docx"]);

// ── Size limits ──────────────────────────────────────────────────────────────

/** Plain-text files larger than this are skipped during discovery. */
export const TEXT_MAX_BYTES = 500_000;
/** Binary documents (PDF/DOCX) larger than this are skipped during discovery. */
export const BINARY_DOC_MAX_BYTES = 10_000_000;

// ── Chunk-size caps ──────────────────────────────────────────────────────────

/**
 * Chunk-size caps. chunkText() limits line COUNT, not length — a single
 * minified/CSV/base64 line up to TEXT_MAX_BYTES would become one giant
 * chunk that gets tokenized to the tokenizer's 8192-token ceiling, and the
 * whole ONNX batch would be padded to it ([batch, heads, seq, seq] attention
 * → tens of GB). Capping keeps a batch at ~1k tokens padded worst case.
 */
export const MAX_LINE_CHARS = 1000;
export const MAX_CHUNK_CHARS = 4000;

// ── Directory traversal ──────────────────────────────────────────────────────

/** Directory names never descended into during file discovery. */
export const SKIP_DIRS = new Set([
  "node_modules", ".git", ".next", "dist", "build", "__pycache__", ".venv", "venv", ".cache",
]);
