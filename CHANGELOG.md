# Changelog

## Unreleased

- **`/rag clear` factory reset**: in addition to clearing the index, `/rag clear` now deletes every file in the active store directory (`.pi/rag/` project store or `~/.pi/rag/` global — `rag.db`, WAL/SHM sidecars, `config.json`, legacy `index.json`, anything else) and regenerates fresh defaults: a default `config.json` and an empty schema-initialized `rag.db`. Tracked paths, exclude patterns, and custom config are no longer preserved across a clear. New `resetStore()` export (db.ts / package root); `clearIndex()` remains available for index-only wipes.
- **Dual embedding models (code vs. prose)**: file extensions are split into two groups. Code files (`.ts`, `.py`, `.rs`, `.css`, `.tf`, … — `DEFAULT_CODE_EXTS`) are embedded by `jinaai/jina-embeddings-v2-base-code` (768-dim, q8 ONNX, ~170 MB, no task prefixes); everything else (`.md`, `.txt`, `.html`, `.json`, `.yaml`, PDF/DOCX, … — `DEFAULT_DOC_EXTS`) keeps `nomic-ai/nomic-embed-text-v1.5`. Vectors live in separate `sqlite-vec` tables (`chunks_vec` text / `chunks_vec_code` code); `hybridSearch` embeds the query with both models and only loads a model when its table has vectors. Migration on first open clears code-classified files (prose chunks and their nomic vectors survive) — run `/rag rebuild` to re-embed code files with the code model.
- **Vector scoring change**: vector scores are now raw cosine similarity (clamped at 0) instead of candidate-set min-max. Both models emit unit-normalized vectors, so cosine is on a shared absolute scale across the two spaces; min-max would have pinned the top chunk of *each* space at 1.0, making a weak match in a small space tie the best match in the other space.
- **Per-model progress + notifications**: `onEmbed` progress is now scoped per group (`code`/`text`), and the TUI renders one line per model (`code n/n jina-code`, `text n/n nomic`) under a combined bar. A new `onModelLoad` callback fires a `⏳ Loading … embedding model` notification right before each model's first-run download so the cold-start doesn't look stuck. `/rag index|rebuild|refresh` share one progress-callbacks factory (was triplicated).
- **`/rag ext` groups**: `add` takes an optional `[code|text]` group (inferred from the extension by default), `list` shows both groups with their models, `reset` restores both. New `extraCodeExtensions` config field; generic `extraExtensions` route to the text group unless the extension is a code default.
- **`.twig` support**: added to the code group (`DEFAULT_CODE_EXTS`) — a markup-with-logic template language, consistent with `.vue`/`.svelte`/`.astro` already routing there. Picked up on the next rebuild/24 h auto-refresh.
- **Status surface**: bare `/rag` and the `rag_status` tool now report both models and per-group vector counts; index/rebuild/refresh completion notifications include the code/text chunk split.
- **Fixes**: session_start notification message aligned with tests (drops the "sesseion" typo); pre-existing typecheck breakage from test imports of `MAX_LINE_CHARS`/`MAX_CHUNK_CHARS` (now re-exported from the package root); `collectFromTracked`/`collectFromTrackedAsync` accept `Pick<RagConfig, "trackedPaths" | "excludePatterns">`.
- **Tests**: extension-group resolution + `classifyFile`; one-shot dual-model schema migration (code files cleared, prose vectors survive, migration doesn't fire on hand-seeded stores without model metadata); real-ONNX jina tests (dim/norm, code relevance, dual-space hybridSearch) gated by `SKIP_EMBEDDING_TESTS`.
- **Chunk-size caps (indexing freeze fix)**: `chunkText` previously capped line *count* (50) but not length — one minified/CSV/base64 line could become a ~500 KB chunk, tokenized to the tokenizer's 8192-token ceiling, and the whole 64-text ONNX batch padded to it (`[batch, heads, seq, seq]` attention → tens of GB → indexing appears stuck, e.g. at "chunks: 384/3793"). Lines are now hard-split at `MAX_LINE_CHARS` (1000, segments keep their source line number) and chunks capped at `MAX_CHUNK_CHARS` (4000 ≈ 1k tokens). Embedding groups are also sorted by content length so batches pad to similar-size texts instead of one monster inflating its neighbors. Re-index with `/rag rebuild --force` to re-chunk existing stores.
- **Embedding model swap**: `Xenova/all-MiniLM-L6-v2` (384-dim, ~23 MB) → `nomic-ai/nomic-embed-text-v1.5` (768-dim, q8-quantized ONNX, ~111 MB download). Queries and documents are prefixed with the nomic `search_query:` / `search_document:` task instructions as required by the model card. Existing stores are auto-migrated on first open: the stale `chunks_vec` table and indexed content are dropped (tracked paths survive in `config.json`) — run `/rag rebuild` to re-index.

## 0.4.1

- **Docs refresh**: README rewritten for 0.4.0 feature set — SQLite/FTS5/sqlite-vec storage, PDF/DOCX/HTML extraction, OCR fallback, per-project store, tracked paths + exclude patterns, 24 h auto-refresh, trailing-message auto-injection. Commands table expanded with `/rag find`, `/rag refresh`, `/rag rebuild --force`, `/rag exclude`, `/rag help`. Optional OCR install instructions (`brew install poppler tesseract tesseract-lang` / `apt install poppler-utils tesseract-ocr ...`). New "Testing" section noting `SKIP_EMBEDDING_TESTS` and the tesseract-absent OCR skip.
- **`package.json`**: description rewritten to mention SQLite + sqlite-vec + PDF/DOCX/HTML + OCR + per-project storage. Keywords += `sqlite`, `fts5`, `sqlite-vec`, `pdf`, `docx`, `ocr`.
- **`.gitignore`**: ignore `.pi/` so local RAG stores don't leak into commits.

## 0.4.0

- **SQLite storage** (replaces JSON): index now lives in a `rag.db` file using `better-sqlite3` + FTS5 virtual table for BM25 full-text search + `sqlite-vec` for vector similarity. Automatic one-shot migration from legacy `index.json` on first run, no data loss. WAL mode enabled for safe concurrent reads.
- **Per-project RAG store**: walks up from `process.cwd()` looking for `.pi/rag/`; falls back to `~/.pi/rag/` global store. First `/rag index` in a directory with no parent store creates one at cwd. Override with `$PI_RAG_DIR`.
- **Tracked paths + gitignore-style exclude patterns**: `trackedPaths` and `excludePatterns` in config; `/rag rebuild` re-walks tracked paths so new files are picked up automatically.
- **24h auto-refresh**: `before_agent_start` hook checks index age; re-indexes stale tracked paths in the background. `/rag refresh` command triggers manually. Configurable via `ragAutoRefresh`.
- **`/rag rebuild --force`**: wipes DB and re-embeds everything from scratch; fixes progress bar freezing during rebuild.
- **PDF + DOCX indexing**: `pdf-parse` for text PDFs, `mammoth` for DOCX files.
- **OCR fallback for image-only PDFs**: `pdftoppm` + `tesseract` pipeline for scanned documents (optional system deps).
- **HTML → Markdown via `turndown`** before chunking — cleaner chunks for web content.
- **`/rag find <glob>`**: list indexed files matching a glob pattern.
- **`/rag help`**: show all available subcommands.
- **`/rag` autocompletions**: working tab-completions for all subcommands.
- **Batched ONNX embeddings** (perf): `embedBatch()` now passes up to 64 texts per ONNX forward pass instead of 1-at-a-time (~64× fewer forward passes; ~219 passes for 13,955 chunks vs 13,955 previously).
- **Parallel file reads** (perf): Phase 1 of `indexFiles()` reads/chunks up to 32 files concurrently so I/O latency hides behind CPU work.
- **RAG context injected at end of prompt** (perf): avoids KV cache invalidation on models that support prefix caching.
- **Modular split**: `index.ts` refactored into 9 focused modules (`chunking.ts`, `embed.ts`, `indexing.ts`, `search.ts`, `store.ts`, `db.ts`, `config.ts`, `constants.ts`, `types/`).
- **104 tests** via vitest: covers chunking, math, BM25 search, SQLite storage round-trip, FTS5 triggers, vector normalization, PDF/DOCX/OCR extraction, per-project store resolution, 24h auto-refresh, and configurable extensions.
- **Fix**: tool definitions now include required `label` and `AgentToolResult.details` fields.
- **Fix**: silence `pdfjs` worker warnings in TUI.
- **Fix**: FTS5 query escaping for single quotes; split into individual terms.

## Unreleased

- **Configurable file extensions** (closes #9): expanded the default list to cover commonly-missing languages (`.cs`, `.tsx`, `.jsx`, `.kt`, `.swift`, `.rb`, `.php`, `.lua`, `.dart`, `.vue`, `.svelte`, `.scala`, `.scss`, `.tf`, `.hcl`, `.mdx`, …) and added `extraExtensions` / `excludeExtensions` to `RagConfig` plus a `/rag ext list|add|remove|reset` subcommand so users can extend the allowlist without forking. Includes 6 new tests for `normalizeExt` and `resolveExtensions`.
- **Test suite** (38 tests, no dev dependencies — uses `node --test` + `--experimental-strip-types`): covers cosine/normalize math, chunking, file collection against real tmp dirs, BM25 search ranking + phrase boost, storage I/O round-trip + legacy `~/.pi/lens` → `~/.pi/rag` migration, and live embedding/semantic-search against the real ONNX model. The model (`Xenova/all-MiniLM-L6-v2`, ~23 MB) is fetched from HuggingFace on the first run; set `SKIP_EMBEDDING_TESTS=1` to opt out in offline CI. Run with `npm test`.
- **Storage paths overridable via env**: `PI_RAG_DIR` and `PI_RAG_LEGACY_DIR` let the index live somewhere other than `~/.pi/rag` (useful for project-local indexes and isolated tests).

## 0.3.0

- **Renamed `/lens` → `/rag`**: all commands now use `/rag index|search|status|rebuild|clear|on|off`
- **Renamed tools**: `lens_index` → `rag_index`, `lens_query` → `rag_query`, `lens_status` → `rag_status`
- **Storage migrated**: index data moved from `~/.pi/lens/` to `~/.pi/rag/` — existing index is automatically migrated on first run, no data loss
- **`/rag on|off`**: simplified toggle (previously `/lens rag on|off`)
- **No file limit**: removed the 500-file cap — indexes all files in a directory
- **Live progress**: `setWidget` + `setStatus` updates during `/rag index` and `/rag rebuild`; stderr overwrite line for tool mode (`rag_index`)
- **Event loop yield**: `await setTimeout(0)` between files so the TUI can re-render and the agent doesn't appear hung

## 0.2.0

- **Hybrid RAG**: BM25 + local vector embeddings via `@huggingface/transformers` (Transformers.js)
- **Auto-injection**: `before_agent_start` hook injects relevant chunks into every LLM prompt
- **Embedding model**: `Xenova/all-MiniLM-L6-v2` (384-dim, ~23MB, downloads once, runs fully offline)
- **Score transparency**: search results now show `bm25`, `vector`, and `hybrid` scores
- **`/lens rag on|off`**: toggle auto-injection at runtime *(renamed to `/rag on|off` in 0.3.0)*
- **`/lens status`**: now shows vector coverage % *(renamed to `/rag status` in 0.3.0)*
- **Config file**: `~/.pi/lens/config.json` for `ragEnabled`, `ragTopK`, `ragScoreThreshold`, `ragAlpha` *(moved to `~/.pi/rag/` in 0.3.0)*
- Bumped to `dependencies` for `@huggingface/transformers`

## 0.1.0

- Initial release
- BM25 keyword search over local files
- Tools: `lens_index`, `lens_query`, `lens_status` *(renamed to `rag_*` in 0.3.0)*
- Commands: `/lens index|search|status|rebuild|clear|context` *(renamed to `/rag` in 0.3.0)*
