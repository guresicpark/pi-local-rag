import { existsSync, readFileSync, readdirSync, statSync, mkdtempSync, rmSync, writeFileSync, promises as fsPromises } from "node:fs";
import { extname, basename, join, relative } from "node:path";
import { tmpdir } from "node:os";
import { createHash } from "node:crypto";
import { spawnSync } from "node:child_process";
import ignore from "ignore";
import { BINARY_DOC_EXTS, TEXT_MAX_BYTES, BINARY_DOC_MAX_BYTES, MAX_LINE_CHARS, MAX_CHUNK_CHARS, SKIP_DIRS } from "./constants.ts";
import { loadConfig, resolveExtensions, type RagConfig } from "./config.ts";

const yield_ = () => new Promise<void>(r => setTimeout(r, 0));

function stderrProgress(msg: string) { process.stderr.write(`\r\x1b[2K${msg}`); }

export function sha256(data: string): string {
  return createHash("sha256").update(data).digest("hex").slice(0, 12);
}

/** sha256 over a Buffer's bytes, streamed in 64 KiB latin-1 windows.
 *  update() UTF-8-encodes each window exactly as the old one-shot
 *  `sha256(buf.toString("binary"))` encoded the whole string, and SHA-256
 *  digests concatenated updates identically — so stored hashes stay
 *  compatible — while the transient string stays bounded at 64 KiB instead
 *  of materializing a full copy of a multi-megabyte PDF/DOCX (times the 32
 *  concurrent indexing producers). */
export function sha256Buf(buf: Buffer): string {
  const h = createHash("sha256");
  for (let i = 0; i < buf.length; i += 65536) {
    h.update(buf.subarray(i, Math.min(i + 65536, buf.length)).toString("binary"));
  }
  return h.digest("hex").slice(0, 12);
}

/** True for the exact character set String.prototype.trim removes
 *  (WhiteSpace + LineTerminator per spec). */
function isTrimWs(c: number): boolean {
  return c === 32 || (c >= 9 && c <= 13) || c === 0xa0 || c === 0xfeff || c === 0x2028 || c === 0x2029;
}

/** True when [from, to) of `s` contains no non-whitespace character —
 *  like `s.trim() === ""` on the slice, with no allocation and an exit at
 *  the first non-whitespace char (usually char 0 for code lines). */
function isBlankRange(s: string, from: number, to: number): boolean {
  for (let i = from; i < to; i++) if (!isTrimWs(s.charCodeAt(i))) return false;
  return true;
}

/** `s.slice(from, to).trim().length > n` without allocating anything. */
function trimLenGTRange(s: string, from: number, to: number, n: number): boolean {
  let f = from;
  let l = to - 1;
  while (f <= l && isTrimWs(s.charCodeAt(f))) f++;
  if (f > l) return false;
  while (l > f && isTrimWs(s.charCodeAt(l))) l--;
  return l - f + 1 > n;
}

function trimLenGT(s: string, n: number): boolean {
  return trimLenGTRange(s, 0, s.length, n);
}

/**
 * Two indexOf("\n") scans recording line-start offsets — indexOf walks the
 * string with SIMD memchr and, unlike text.split("\n"), materializes no
 * per-line strings. Line j spans [starts[j], starts[j+1] - 1), the last
 * line to text.length.
 *
 * Offsets live in an Int32Array (4 B/line): a plain number[] stores each
 * offset as an 8-byte tagged SMI (Node builds don't enable pointer
 * compression) and its geometric growth leaves dead backing stores for the
 * GC. Files are capped at TEXT_MAX_BYTES, so offsets always fit int32.
 *
 * The array is sized exactly up front: a first memchr pass counts the lines
 * (and flags any over-long line), then a second pass records offsets into
 * that single, precisely-sized allocation. This drops the old 2× geometric
 * growth — which over-allocated the tail by up to 2× and left a dead
 * backing store for the GC at every doubling step — for zero garbage and no
 * over-allocation. The extra memchr pass is cheap (~650 MB/s) and roughly
 * offsets the memcpy the old grow steps paid. Kept as its own small
 * function so it optimizes cleanly, away from the branchier chunk-assembly
 * loops below.
 */
function scanLines(text: string): { starts: Int32Array; lineCount: number; hasLongLine: boolean } {
  // Pass 1: count lines + flag any line longer than MAX_LINE_CHARS. No
  // offsets are recorded — the buffer can't be sized until the count is
  // known, and a pure count pass keeps this loop free of array writes.
  let lineCount = 1;
  let hasLongLine = false;
  let prev = 0;
  let p = text.indexOf("\n");
  while (p !== -1) {
    if (p - prev > MAX_LINE_CHARS) hasLongLine = true;
    lineCount++;
    prev = p + 1;
    p = text.indexOf("\n", prev);
  }
  if (text.length - prev > MAX_LINE_CHARS) hasLongLine = true;

  // One allocation sized exactly for lineCount line starts plus the
  // sentinel slot at starts[lineCount].
  const starts = new Int32Array(lineCount + 1);
  let n = 1; // starts[0] = 0 via zero-initialization
  prev = 0;
  p = text.indexOf("\n");
  while (p !== -1) {
    prev = p + 1;
    starts[n++] = prev;
    p = text.indexOf("\n", prev);
  }
  // Sentinel: starts[lineCount] = text.length + 1, so the assembly loops can
  // read starts[end] / starts[j + 1] uniformly (no "last line" ternary) —
  // the last line then ends at starts[lineCount] - 1 === text.length.
  starts[lineCount] = text.length + 1;
  return { starts, lineCount, hasLongLine };
}

export function chunkText(text: string, maxLines = 50): { content: string; lineStart: number; lineEnd: number }[] {
  const chunks: { content: string; lineStart: number; lineEnd: number }[] = [];
  const { starts, lineCount, hasLongLine } = scanLines(text);

  if (!hasLongLine) {
    // Fast path (no line exceeds the cap — the overwhelmingly common case):
    // rows map 1:1 onto source lines and every chunk is a contiguous slice
    // of `text`, so content comes straight from substring() — in V8 an O(1)
    // sliced string sharing the parent, no bytes copied and no join needed.
    //
    // The joined length of rows [i, e) needs no separate prefix-sum array:
    // it is starts[e] - starts[i] - 1 (each line contributes its length
    // plus the newline, which lands exactly on the next line's start); a
    // sentinel at starts[lineCount] = text.length + 1 covers the last window
    // with no per-chunk "last line" ternary. So the char-cap check is O(1)
    // per chunk (a short binary search only when the window actually exceeds
    // MAX_CHUNK_CHARS) and no `endOf` closure survives in any hot loop. The
    // blank-line scan checks the first char inline and only calls
    // isBlankRange for rows that start with whitespace.
    let i = 0;
    while (i < lineCount) {
      let end = Math.min(i + maxLines, lineCount);
      for (let j = end - 1; j > i + 10 && j > end - 15; j--) {
        const s = starts[j];
        const e = starts[j + 1] - 1;
        if (s === e) { end = j + 1; break; }
        const c = text.charCodeAt(s);
        if ((c === 32 || (c >= 9 && c <= 13) || c === 0xa0 || c === 0xfeff || c === 0x2028 || c === 0x2029)
          && isBlankRange(text, s, e)) { end = j + 1; break; }
      }
      // Shrink the window further if the joined content would exceed
      // MAX_CHUNK_CHARS (≈1k tokens — the embedding model's sweet spot).
      const spanEnd = starts[end];
      if (spanEnd - starts[i] - 1 > MAX_CHUNK_CHARS) {
        const t = starts[i] + MAX_CHUNK_CHARS + 1;
        let lo = i + 1, hi = end;
        while (hi - lo > 1) {
          const mid = (lo + hi) >> 1;
          if (starts[mid] > t) hi = mid; else lo = mid;
        }
        end = hi - 1;
      }
      const from = starts[i];
      const to = starts[end] - 1;
      if (trimLenGTRange(text, from, to, 20)) {
        chunks.push({ content: text.substring(from, to), lineStart: i + 1, lineEnd: end });
      }
      i = end;
    }
    return chunks;
  }

  // Slow path — pathologically long lines (minified JS/JSON, CSV rows,
  // base64 blobs) are split into MAX_LINE_CHARS segments; segments share
  // the source line's number so references stay accurate. Chunk content is
  // no longer contiguous in `text` (segments of one line are joined by
  // injected newlines), so rows are described by [start, end) offsets into
  // `text` and joined on demand — no per-row strings are materialized.
  //
  // Every row is a substring of `text` (a whole line, or a MAX_LINE_CHARS
  // segment of a long line), so three parallel Int32Arrays — start, end,
  // source line number — replace the old string[] of row texts (one
  // sliced-string object per row) and are sized exactly up front from the
  // line offsets instead of growing.
  let totalRows = 0;
  for (let j = 0; j < lineCount; j++) {
    const len = starts[j + 1] - 1 - starts[j];
    totalRows += len <= MAX_LINE_CHARS ? 1 : Math.ceil(len / MAX_LINE_CHARS);
  }
  const rowStart = new Int32Array(totalRows);
  const rowEnd = new Int32Array(totalRows);
  const rowNums = new Int32Array(totalRows);
  let ri = 0;
  for (let j = 0; j < lineCount; j++) {
    const s = starts[j];
    const e = starts[j + 1] - 1;
    const num = j + 1;
    if (e - s <= MAX_LINE_CHARS) {
      rowStart[ri] = s; rowEnd[ri] = e; rowNums[ri] = num; ri++;
    } else {
      for (let k = s; k < e; k += MAX_LINE_CHARS) {
        rowStart[ri] = k; rowEnd[ri] = Math.min(k + MAX_LINE_CHARS, e); rowNums[ri] = num; ri++;
      }
    }
  }

  let i = 0;
  while (i < totalRows) {
    let end = Math.min(i + maxLines, totalRows);
    for (let j = end - 1; j > i + 10 && j > end - 15; j--) {
      if (isBlankRange(text, rowStart[j], rowEnd[j])) { end = j + 1; break; }
    }
    let len = rowEnd[i] - rowStart[i];
    for (let j = i + 1; j < end; j++) {
      len += 1 + rowEnd[j] - rowStart[j];
      if (len > MAX_CHUNK_CHARS) { end = j; break; }
    }
    const parts: string[] = new Array(end - i);
    for (let j = i; j < end; j++) parts[j - i] = text.substring(rowStart[j], rowEnd[j]);
    const chunk = parts.join("\n");
    if (trimLenGT(chunk, 20)) {
      chunks.push({ content: chunk, lineStart: rowNums[i], lineEnd: rowNums[end - 1] });
    }
    i = end;
  }
  return chunks;
}

/** Shared extension/size + exclude-pattern filter for the sync and async
 *  walkers — keeps `acceptable` and `isExcluded` defined once instead of
 *  being copy-pasted into collectFiles and collectFilesAsync. */
function makeFileFilter(allowed: Set<string>, ig: ReturnType<typeof ignore> | null, root: string) {
  function acceptable(fp: string, size: number): boolean {
    const ext = extname(fp).toLowerCase();
    if (allowed.has(ext)) return size < TEXT_MAX_BYTES;
    if (BINARY_DOC_EXTS.has(ext)) return size < BINARY_DOC_MAX_BYTES;
    return false;
  }

  function isExcluded(absPath: string): boolean {
    if (!ig) return false;
    const rel = relative(root, absPath);
    if (!rel || rel.startsWith("..")) return false;
    return ig.ignores(rel);
  }

  return { acceptable, isExcluded };
}

export function collectFiles(
  dirPath: string,
  exts?: Set<string>,
  excludePatterns: string[] = [],
): string[] {
  const allowed = exts ?? resolveExtensions(loadConfig());
  const ig = excludePatterns.length ? ignore().add(excludePatterns) : null;
  const files: string[] = [];
  const root = dirPath;
  const { acceptable, isExcluded } = makeFileFilter(allowed, ig, root);

  try {
    const stat = statSync(dirPath);
    if (stat.isFile()) {
      if (!acceptable(dirPath, stat.size)) return [];
      if (ig && ig.ignores(basename(dirPath))) return [];
      return [dirPath];
    }
  } catch { return []; }

  function walk(dir: string) {
    try {
      for (const entry of readdirSync(dir, { withFileTypes: true })) {
        const fp = join(dir, entry.name);
        if (entry.isDirectory()) {
          if (SKIP_DIRS.has(entry.name) || entry.name.startsWith(".")) continue;
          if (isExcluded(fp)) continue;
          walk(fp);
        } else {
          const ext = extname(entry.name).toLowerCase();
          if (!allowed.has(ext) && !BINARY_DOC_EXTS.has(ext)) continue;
          if (isExcluded(fp)) continue;
          try {
            if (acceptable(fp, statSync(fp).size)) files.push(fp);
          } catch {}
        }
      }
    } catch {}
  }
  walk(root);
  return files;
}

export function collectFromTracked(cfg: Pick<RagConfig, "trackedPaths" | "excludePatterns">): string[] {
  const out = new Set<string>();
  for (const p of cfg.trackedPaths) {
    if (!existsSync(p)) continue;
    for (const f of collectFiles(p, undefined, cfg.excludePatterns)) out.add(f);
  }
  return [...out];
}

/**
 * Async variant of collectFiles that uses fs.promises and yields to the event
 * loop between directories. Required for /rag rebuild on large trackedPaths
 * (45k+ files) — the synchronous walk pegs the event loop long enough that
 * the TUI freezes before reaching the embed phase. Adapted from
 * theli-ua/pi-local-rag@8432a15.
 */
export async function collectFilesAsync(
  dirPath: string,
  exts?: Set<string>,
  excludePatterns: string[] = [],
): Promise<string[]> {
  const allowed = exts ?? resolveExtensions(loadConfig());
  const ig = excludePatterns.length ? ignore().add(excludePatterns) : null;
  const files: string[] = [];
  const root = dirPath;
  const { acceptable, isExcluded } = makeFileFilter(allowed, ig, root);

  try {
    const st = await fsPromises.stat(dirPath);
    if (st.isFile()) {
      if (!acceptable(dirPath, st.size)) return [];
      if (ig && ig.ignores(basename(dirPath))) return [];
      return [dirPath];
    }
  } catch { return []; }

  async function walk(dir: string): Promise<void> {
    let entries: import("node:fs").Dirent[];
    try {
      entries = await fsPromises.readdir(dir, { withFileTypes: true });
    } catch { return; }
    for (const entry of entries) {
      const fp = join(dir, entry.name);
      if (entry.isDirectory()) {
        if (SKIP_DIRS.has(entry.name) || entry.name.startsWith(".")) continue;
        if (isExcluded(fp)) continue;
        await walk(fp);
      } else {
        const ext = extname(entry.name).toLowerCase();
        if (!allowed.has(ext) && !BINARY_DOC_EXTS.has(ext)) continue;
        if (isExcluded(fp)) continue;
        try {
          const st = await fsPromises.stat(fp);
          if (acceptable(fp, st.size)) files.push(fp);
        } catch {}
      }
    }
    // Yield between directories so the event loop can process UI updates.
    await yield_();
  }

  await walk(root);
  return files;
}

export async function collectFromTrackedAsync(cfg: Pick<RagConfig, "trackedPaths" | "excludePatterns">): Promise<string[]> {
  const out = new Set<string>();
  for (const p of cfg.trackedPaths) {
    if (!existsSync(p)) continue;
    for (const f of await collectFilesAsync(p, undefined, cfg.excludePatterns)) out.add(f);
  }
  return [...out];
}

/** Returns true if `file` is matched by `excludePatterns` relative to any of `roots`. */
export function isExcludedByConfig(file: string, roots: string[], excludePatterns: string[]): boolean {
  if (!excludePatterns.length) return false;
  const ig = ignore().add(excludePatterns);
  for (const root of roots) {
    const rel = relative(root, file);
    if (!rel || rel.startsWith("..")) continue;
    if (ig.ignores(rel)) return true;
  }
  return false;
}

// pdfjs (bundled inside pdf-parse) routes warnings through console.log with a
// "Warning: " prefix. On real-world PDFs this fires thousands of times per
// document ("Ran out of space in font private use area", missing glyphs, …).
// The font warnings come from pdf.worker.js, which is a separate webpack
// bundle whose verbosity is not externally configurable (its setVerbosityLevel
// export exists only as a placeholder at the outer module level). Filtering
// console.log for the known pdfjs prefixes is the only reliable approach.
const PDFJS_LOG_PREFIX = /^(Warning|Info|Deprecated API usage):/;
async function withPdfjsSilenced<T>(fn: () => Promise<T>): Promise<T> {
  const origLog = console.log;
  console.log = (...args: unknown[]) => {
    const first = args[0];
    if (typeof first === "string" && PDFJS_LOG_PREFIX.test(first)) return;
    origLog(...args);
  };
  try {
    return await fn();
  } finally {
    console.log = origLog;
  }
}

// ─── OCR fallback for image-based PDFs ───────────────────────────────────────

type OcrTooling = { available: false } | { available: true; langs: string };
let _ocrTooling: OcrTooling | undefined;
let _ocrUnavailableLogged = false;

/** One-shot probe for system pdftoppm + tesseract. Caches the result. */
export function getOcrTooling(): OcrTooling {
  if (_ocrTooling) return _ocrTooling;
  const pdftoppm = spawnSync("pdftoppm", ["-v"]);
  const tess = spawnSync("tesseract", ["--list-langs"], { encoding: "utf-8" });
  if (pdftoppm.error || tess.error) return (_ocrTooling = { available: false });
  // tesseract prints langs on stderr in some builds, stdout in others.
  const out = `${tess.stdout || ""}\n${tess.stderr || ""}`;
  const have = new Set(out.split(/\r?\n/).map(s => s.trim()).filter(Boolean));
  const wanted = ["jpn", "eng"].filter(l => have.has(l));
  if (!wanted.length) return (_ocrTooling = { available: false });
  return (_ocrTooling = { available: true, langs: wanted.join("+") });
}

/** Render `buf` to PNGs via pdftoppm, OCR each page via tesseract, return concatenated text. */
async function ocrPdf(buf: Buffer, langs: string, label: string): Promise<string> {
  const MAX_PAGES = 200;
  const PER_PAGE_TIMEOUT_MS = 60_000;
  const dir = mkdtempSync(join(tmpdir(), "rag-ocr-"));
  try {
    const pdfPath = join(dir, "in.pdf");
    writeFileSync(pdfPath, buf);
    const render = spawnSync("pdftoppm", ["-png", "-r", "200", pdfPath, join(dir, "p")], { encoding: "utf-8" });
    if (render.status !== 0) return "";
    const pages = readdirSync(dir).filter(f => f.startsWith("p-") && f.endsWith(".png")).sort();
    const total = Math.min(pages.length, MAX_PAGES);
    if (pages.length > MAX_PAGES) {
      process.stderr.write(`\r\x1b[2K[rag] OCR ${label}: ${pages.length} pages, capping at ${MAX_PAGES}\n`);
    }
    const out: string[] = [];
    for (let i = 0; i < total; i++) {
      stderrProgress(`[OCR ${i + 1}/${total}] ${label}`);
      await yield_();
      const r = spawnSync("tesseract", [join(dir, pages[i]), "-", "-l", langs], {
        encoding: "utf-8",
        timeout: PER_PAGE_TIMEOUT_MS,
        maxBuffer: 16 * 1024 * 1024,
      });
      out.push(r.status === 0 ? (r.stdout ?? "") : "");
    }
    process.stderr.write(`\r\x1b[2K`);
    return out.join("\n\n");
  } finally {
    try { rmSync(dir, { recursive: true, force: true }); } catch {}
  }
}

/** True if `text` looks too sparse for `numpages` to be the real content of the document. */
export function isSparsePdfText(text: string, numpages: number): boolean {
  return text.trim().length < 50 * Math.max(1, numpages);
}

/**
 * Read and decode a file into UTF-8 text. PDF and DOCX are routed through
 * extraction libraries; everything else is read as plain UTF-8. Hash is
 * computed over the raw bytes for binaries (so the source file's identity
 * drives skip-on-rebuild) and over the decoded text for plain text files.
 */
export async function extractText(fp: string): Promise<{ text: string; hash: string; size: number }> {
  const ext = extname(fp).toLowerCase();
  if (ext === ".pdf") {
    const buf = readFileSync(fp);
    const { default: pdf } = await import("pdf-parse/lib/pdf-parse.js");
    const data = await withPdfjsSilenced(() => pdf(buf));
    let text = data.text;
    if (isSparsePdfText(text, data.numpages ?? 1)) {
      const tools = getOcrTooling();
      if (tools.available) {
        const ocr = await ocrPdf(buf, tools.langs, basename(fp));
        if (ocr.trim().length > text.trim().length) text = ocr;
      } else if (!_ocrUnavailableLogged) {
        _ocrUnavailableLogged = true;
        process.stderr.write(
          `\r\x1b[2K[rag] OCR unavailable: install pdftoppm + tesseract (with jpn/eng traineddata) to index image PDFs\n`
        );
      }
    }
    return { text, hash: sha256Buf(buf), size: buf.length };
  }
  if (ext === ".docx") {
    const buf = readFileSync(fp);
    const { default: mammoth } = await import("mammoth");
    const { value } = await mammoth.extractRawText({ buffer: buf });
    return { text: value, hash: sha256Buf(buf), size: buf.length };
  }
  if (ext === ".html" || ext === ".htm") {
    const { default: TurndownService } = await import("turndown");
    const raw = readFileSync(fp, "utf-8");
    const td = new TurndownService({
      headingStyle: "atx",
      codeBlockStyle: "fenced",
      blankReplacement: (_content, node) => node.tagName === "BR" ? "\n" : "",
    });
    td.remove(["script", "style"]);
    td.remove(["nav", "footer"]);
    const text = td.turndown(raw);
    return { text, hash: sha256(raw), size: raw.length };
  }
  const text = readFileSync(fp, "utf-8");
  return { text, hash: sha256(text), size: text.length };
}
