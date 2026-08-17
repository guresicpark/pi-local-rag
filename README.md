# pi-local-rag

Local hybrid RAG pipeline for the [Pi coding agent](https://github.com/badlogic/pi-mono). Index your local files and search them with BM25 + vector similarity — **zero cloud dependency, works fully offline**.

> **Fork notice**: this is a fork of [vahidkowsari/pi-local-rag](https://github.com/vahidkowsari/pi-local-rag), diverged at upstream `v0.4.1`. See [What differs from upstream](#what-differs-from-upstream) below.

## What differs from upstream

**Embedding models**

- Switched from `Xenova/all-MiniLM-L6-v2` (384-dim, ~23 MB) to `nomic-ai/nomic-embed-text-v1.5` (768-dim, q8, ~111 MB) with the required `search_query:` / `search_document:` task prefixes; existing stores auto-migrate
- Dual-model embedding: code files → `jinaai/jina-embeddings-v2-base-code` (768-dim, ~170 MB), prose/data/docs → nomic; vectors in separate `sqlite-vec` tables, hybrid search embeds the query with both models and ranks by absolute cosine similarity (shared scale across spaces)
- `/rag ext` commands now take an optional `[code|text]` group; new `extraCodeExtensions` config field

**Reliability & security fixes**

- Chunk-size caps (`MAX_LINE_CHARS=1000`, `MAX_CHUNK_CHARS≈4000`) — one minified/base64 line could previously produce a ~500 KB chunk whose 8192-token padding blew up the ONNX attention batch and froze indexing
- ~5× faster chunking — `chunkText()` (the CPU hot spot of indexing's read phase) scans line boundaries via `indexOf` instead of `split` and, when no line exceeds `MAX_LINE_CHARS`, emits chunks as O(1) substring slices of the source text (zero copy, no join); the minified/long-line path keeps segment splitting with allocation-free whitespace checks. Byte-identical output
- O(1) chunk-assembly — the per-chunk char-cap check reads the joined length straight off the line-offset array (`starts[e] − starts[i] − 1`, binary search only when a window exceeds `MAX_CHUNK_CHARS`) instead of re-summing up to 50 line lengths per chunk; blank-line detection exits on the first non-whitespace char without a call. Chunking is now memchr-scan-bound (~650 MB/s)
- Sentinel line-offset terminator — `scanLines()` writes one extra slot (`starts[lineCount] = text.length + 1`) so the assembly loops read `starts[end]` / `starts[j + 1]` uniformly and drop the per-chunk "last line" ternaries; byte-identical output, ~5–15% faster on realistic corpora
- Lower indexing memory — line offsets live in exactly-sized `Int32Array`s (4 B/line vs 8-byte tagged SMIs in a `number[]`); `scanLines()` counts lines in a first memchr pass then allocates precisely (no 2× growth tail, no dead backing stores), and the long-line path describes rows as `[start, end)` offsets instead of a per-row string. Embedding vectors stay packed as `Float32Array` end-to-end instead of being boxed to `number[]` at embed time and converted back to float32 at insert time (~102 MB → ~44 MB retained for a 14k-chunk store; 2–6.6× less transient GC garbage per file)
- Near-flat heap while indexing — embedding is interleaved with the DB writes: each 256-chunk batch is committed in its own transaction as soon as its vectors arrive, then its chunk texts and vectors are released immediately (previously the whole corpus's texts *plus* one vector per chunk accumulated in the heap until a single end-of-run transaction, and one big transaction piled up in the WAL). PDF/DOCX hashing streams the buffer in 64 KiB windows instead of materializing a full string copy of each multi-megabyte document; stored hashes stay compatible so skip-on-rebuild caches survive upgrades
- Skip-before-chunk on re-index — `{path → hash/embedded}` state is bulk-loaded up front and checked right after `extractText()`, so unchanged files never enter `chunkText()`; previously every file was fully chunked before the skip decision discarded the work
- `/rag clear` was a no-op upstream (never wiped the SQLite index) — now factory-resets the entire store directory and regenerates fresh defaults
- All 17 `npm audit` vulnerabilities fixed via dependency `overrides` (`adm-zip`, `onnxruntime-node`, `sharp`)
- HF model cache pinned to a shared global directory (no per-project re-downloads)
- Singleton DB connection scoped to every request path — a new `withDb()` helper opens the shared connection, runs the handler, and closes it in a `finally` (also on throw); the `/rag` command handlers and the `rag_*` tools previously opened the connection via `getDbConn()` and never closed it, leaking the open handle (and WAL file descriptor) across turns

**UX**

- Live per-batch embedding progress during `/rag index|rebuild|refresh`, with one progress line per model and a `⏳ Loading …` notification before first-run model downloads (only when weights are actually missing from the disk cache)
- Auto-injected RAG context is visible in chat, collapsed to a single summary line
- RAG injection defaults to **off** and auto-enables only once the store has chunks — at session start, or immediately after `/rag index`
- Bare `/rag` toggles the stats widget on repeat calls (`/rag status` subcommand removed)

## Features

- **Hybrid BM25 + vector search** — SQLite FTS5 for keyword scoring, [`sqlite-vec`](https://github.com/asg017/sqlite-vec) for 768-dim cosine NN, blended at retrieval time
- **Dual local ONNX embeddings** — code files are embedded by `jinaai/jina-embeddings-v2-base-code` (~170 MB quantized, trained on code + docstrings); everything else (prose, Markdown, data/config, PDF/DOCX/HTML) by `nomic-ai/nomic-embed-text-v1.5` (~111 MB quantized). Vectors live in separate `sqlite-vec` tables and hybrid search queries **both** spaces, ranking by absolute cosine similarity — all fully offline after first download
- **Many file formats** — text, source code, Markdown, JSON, YAML, plus PDF (with optional OCR fallback for scanned docs), DOCX, HTML (auto-converted to Markdown)
- **Per-project storage** — walks up from cwd looking for `.pi/rag/`; falls back to `~/.pi/rag/` global store
- **Tracked paths + exclude patterns** — `/rag index <path>` remembers what to keep current; gitignore-style `/rag exclude` for `dist/`, `*.log`, etc.
- **Auto-refresh** — stale index (>24 h) silently refreshed before the next agent turn; manual `/rag refresh` for on-demand incremental updates
- **Auto-injection** — relevant chunks appended after the user prompt before every agent turn (KV-cache friendly); off by default, auto-enables once the store has chunks
- **3 AI tools** — `rag_index`, `rag_query`, `rag_status` for the agent to call directly

## Install

```bash
pi install npm:pi-local-rag
```

Or via git:

```bash
pi install git:github.com/vahidkowsari/pi-local-rag
```

Optional: install `pdftoppm` (poppler) + `tesseract` with `eng`/`jpn` traineddata to enable OCR fallback for image-only PDFs.

```bash
# macOS
brew install poppler tesseract tesseract-lang

# Debian/Ubuntu
apt install poppler-utils tesseract-ocr tesseract-ocr-eng tesseract-ocr-jpn
```

The OCR fallback is silent when these tools aren't installed (logs one stderr hint on the first image-only PDF encountered).

## Commands

| Command | Description |
|---|---|
| `/rag index <path>` | Index a file or directory (chunks → embeds → stores); adds the path to tracked paths |
| `/rag search <query>` | Hybrid BM25 + vector search over the index |
| `/rag find <glob>` | List indexed files matching a glob (e.g. `*.ts`, `src/*`) |
| `/rag` | Show index stats, active config, tracked paths, exclude patterns, storage scope (run again to hide) |
| `/rag rebuild [--force]` | Re-walk tracked paths and re-embed all files. `--force` wipes the DB and bypasses the hash-cache check |
| `/rag refresh` | Incremental refresh — only new/changed files (same code path as the 24 h auto-refresh) |
| `/rag clear` | Factory-reset the store: delete every file in the active store dir (`.pi/rag/` or `~/.pi/rag/`) and regenerate fresh defaults (default `config.json` + empty `rag.db`). Tracked paths and custom config are wiped |
| `/rag exclude <pattern>` | Add a gitignore-style exclude pattern; `/rag exclude -<pattern>` to remove; no arg to list |
| `/rag ext list` | Show the extension groups and which model embeds each |
| `/rag ext add <.ext> [code\|text]` | Add an extension to a group (group inferred from the extension by default) |
| `/rag ext remove <.ext>` \| `reset` | Remove an extension / restore defaults |
| `/rag on` \| `off` | Toggle auto-injection |
| `/rag help` | Show all subcommands |

Tab-completion is available for every subcommand.

## Example session

```text
$ /rag index ~/code/my-app
Found 412 files to index
Indexing  ████████████████████████  100%
file:    src/server/handlers/payments.ts
done:    412 embedded · 0 unchanged
⏳ Loading code embedding model: jinaai/jina-embeddings-v2-base-code — first run downloads it (this can take a few minutes)
Embedding  ████████████████████████  100%
code  1610/1610  jina-code
text   237/237   nomic
✅ Indexed 412 files (1,847 chunks: 1610 code · 237 text) · 0 unchanged · 38.4s · tracking 1 path(s) · project store

$ /rag
🔍 pi-local-rag

  Files indexed:    412
  Chunks:           1847
  Vectors:          237 text · 1610 code  (100% coverage)
  Total tokens:     438,219
  Text model:       nomic-ai/nomic-embed-text-v1.5
  Code model:       jinaai/jina-embeddings-v2-base-code
  Last build:       2026-05-26T20:14:03.221Z
  Storage:          /Users/you/code/my-app/.pi/rag (project)

  RAG injection:    enabled  topK=5  threshold=0.1  alpha=0.4

  File types:
    .ts    231  code
    .tsx   118  code
    .md     34  text
    .json   18  text
    .yaml    7  text

  Tracked paths:
    /Users/you/code/my-app

  Exclude patterns:
    (none — add with /rag exclude <pattern>)

$ /rag search "stripe webhook signature verification"
🔍 4 results for "stripe webhook signature verification"  hybrid BM25+vector

payments.ts:142-187  score=0.92
  export async function verifyStripeWebhook(req: Request) {
    const sig = req.headers.get("stripe-signature");
    if (!sig) throw new Error("missing signature header");

webhooks.md:1-23  score=0.71
  # Webhook signing
  All inbound webhooks are verified against the shared secret stored in
  STRIPE_WEBHOOK_SECRET. Stripe signs each request with a t= timestamp...

$ /rag exclude dist/
✅ Added exclude: dist/ · 1 pattern(s) total. Run /rag rebuild to re-apply.

$ /rag find *.html
🔍 12 indexed files matching "*.html"
src/docs/install.html
src/docs/quickstart.html
...

$ /rag rebuild
Scanning tracked paths...
Discovered 3 new files
Rebuilding 415 files...
Rebuilding  ████████████████████████  100%
Embedding   ████████████████████████  100%  1847/1847 chunks
✅ Rebuilt: 3 re-indexed · 412 unchanged · 0 deleted · 1850 chunks · 14.2s
```

> Output above is approximate — actual colors, spacing, and widget layout depend on your terminal theme and the Pi agent's UI.

## AI Tools

The extension registers three tools the agent can call directly:

- **`rag_index`** — Index a path into the pipeline (also adds it to tracked paths)
- **`rag_query`** — Hybrid BM25 + vector search; returns file paths + line numbers + previews + scores
- **`rag_status`** — Index stats, RAG config, storage path + scope

## How It Works

1. **Index** — files are chunked (~50 lines each, broken at blank lines where possible), embedded by group (code extensions → `jinaai/jina-embeddings-v2-base-code`, everything else → `nomic-ai/nomic-embed-text-v1.5`; both 768-dim), and stored in SQLite. PDF/DOCX go through `pdf-parse`/`mammoth`; HTML is converted to Markdown via `turndown`; scanned PDFs fall back to OCR (`pdftoppm` + `tesseract`) when the system tools are installed.
2. **Search** — FTS5 `bm25()` + `sqlite-vec` cosine NN over **both** vector tables (unit-normalized embeddings put both models on a shared cosine scale), blended: `alpha × BM25 + (1-alpha) × cosine` (default `alpha=0.4`). Filename matches on the first query term get a 1.5× boost.
3. **Auto-inject** — before every agent turn, the user's prompt is searched against the index and relevant chunks are appended after the prompt as a hidden `customType: "rag"` message (KV-cache friendly — the system prompt is unchanged across turns).
4. **Auto-refresh** — if the index is older than 24 h, the `before_agent_start` hook re-walks tracked paths and re-indexes new/changed files in the background. Throttled to one stale check per hour.

## Storage

Index data lives in `rag.db` (SQLite, WAL mode, with FTS5 + sqlite-vec extensions loaded). Three resolution rules:

1. **`$PI_RAG_DIR`** — explicit override, wins over everything
2. **Walk-up** from `process.cwd()` looking for an existing `.pi/rag/` directory (stopping before `$HOME`)
3. **Global** fallback at `~/.pi/rag/`

`/rag index <path>` creates a project store at the current cwd if no parent store is in scope. Bare `/rag` shows the resolved path and whether it's project-local or global.

Legacy `~/.pi/lens/` directories are renamed to `~/.pi/rag/` on first run; legacy `index.json` files are migrated into `rag.db` and removed.

## Configuration

Auto-injection is **off by default**. It turns itself on only when the store actually has indexed chunks — at session start or right after `/rag index` — and `/rag on` / `/rag off` toggle it manually. Config lives in `<ragDir>/config.json`:

| Setting | Default | Description |
|---|---|---|
| `ragEnabled` | `false` | Auto-inject context before each turn (auto-enabled once the store has chunks) |
| `ragTopK` | `5` | Max chunks to inject |
| `ragScoreThreshold` | `0.1` | Min hybrid score to include |
| `ragAlpha` | `0.4` | BM25/vector blend (0 = pure vector, 1 = pure BM25) |
| `extraExtensions` | `[]` | Extra text-group extensions to index beyond the defaults |
| `extraCodeExtensions` | `[]` | Extra code-group extensions (embedded by the code model) |
| `excludeExtensions` | `[]` | Default extensions to skip (applies to both groups) |
| `trackedPaths` | `[]` | Absolute paths that `/rag rebuild`/`refresh` re-walk |
| `excludePatterns` | `[]` | Gitignore-style patterns applied when walking tracked paths |

## Testing

```bash
npm test                          # full suite (downloads ~111 MB nomic + ~170 MB jina-code on first run)
SKIP_EMBEDDING_TESTS=1 npm test   # skip the real-ONNX semantic tests
```

OCR end-to-end test is skipped when `tesseract` isn't installed.
