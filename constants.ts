// ANSI color escapes — used by stderr progress lines and TUI widgets in
// callers that don't have access to ctx.ui.theme.
export const RST = "\x1b[0m", B = "\x1b[1m", D = "\x1b[2m";
export const GREEN = "\x1b[32m", YELLOW = "\x1b[33m", CYAN = "\x1b[36m", RED = "\x1b[31m", MAGENTA = "\x1b[35m";

/** Embedding group: code files go to the dedicated code model, everything
 *  else (prose, markup, data/config, PDF/DOCX) to the text model. */
export type EmbedGroup = "code" | "text";

// ── Text/prose model: nomic-ai/nomic-embed-text-v1.5 ──
export const EMBEDDING_MODEL = "nomic-ai/nomic-embed-text-v1.5";
export const VECTOR_DIM = 768;
// nomic-embed-text-v1.5 requires task-type prefixes on inputs (see model
// card). Queries and indexed documents must use different prefixes.
export const QUERY_PREFIX = "search_query: ";
export const DOC_PREFIX = "search_document: ";

// ── Code model: jina-embeddings-v2-base-code (official repo ships the ONNX
// weights itself — tagged transformers.js-compatible, incl. the q8 file) ──
// 768-dim, trained on code + docstring pairs. jina v2 models take NO task
// prefixes (those arrived with v3), so the prefix constants are "".
export const CODE_EMBEDDING_MODEL = "jinaai/jina-embeddings-v2-base-code";
export const CODE_VECTOR_DIM = 768;
export const CODE_QUERY_PREFIX = "";
export const CODE_DOC_PREFIX = "";

// Version of the code-document embedding *scheme* — i.e. how a chunk's text is
// prepared before it goes into the code model (not which model). Bumping this
// invalidates existing code vectors so they're re-embedded with the new scheme,
// the same way a model-id change does. Prose vectors (nomic) are unaffected.
export const CODE_EMBED_SCHEME = "file-context-v1";

// ── File-extension groups ──

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

/** Text/prose + data/config extensions → embedded with EMBEDDING_MODEL
 *  (nomic). HTML/HTM land here too: extractText() converts them to markdown
 *  prose before chunking. PDF/DOCX (BINARY_DOC_EXTS) are also text-group. */
export const DEFAULT_DOC_EXTS = [
  ".md", ".mdx", ".txt", ".rst",
  ".html", ".htm",
  ".json", ".jsonc", ".yaml", ".yml", ".toml", ".ini", ".xml", ".csv", ".tsv",
  ".env", ".gitignore", ".dockerfile",
];

/** Union of both groups — the full default allowlist. Legacy name kept:
 *  callers use this for "is this file indexable at all". */
export const DEFAULT_TEXT_EXTS = [...DEFAULT_CODE_EXTS, ...DEFAULT_DOC_EXTS];

export const BINARY_DOC_EXTS = new Set([".pdf", ".docx"]); // → text group

export const TEXT_MAX_BYTES = 500_000;
export const BINARY_DOC_MAX_BYTES = 10_000_000;

// Chunk-size caps. chunkText() limits line COUNT, not length — a single
// minified/CSV/base64 line up to TEXT_MAX_BYTES becomes one giant chunk that
// gets tokenized to the tokenizer's 8192-token ceiling, and the whole ONNX
// batch is padded to it ([batch, heads, seq, seq] attention → tens of GB).
// Capping keeps a batch at ~1k tokens padded worst case.
export const MAX_LINE_CHARS = 1000;
export const MAX_CHUNK_CHARS = 4000;

export const SKIP_DIRS = new Set([
  "node_modules", ".git", ".next", "dist", "build", "__pycache__", ".venv", "venv", ".cache",
]);
