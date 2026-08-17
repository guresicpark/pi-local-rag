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

function isBlankRow(s: string): boolean {
  return isBlankRange(s, 0, s.length);
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
 * One indexOf("\n") scan recording line-start offsets — indexOf walks the
 * string with SIMD memchr and, unlike text.split("\n"), materializes no
 * per-line strings. Line j spans [starts[j], starts[j+1] - 1), the last
 * line to text.length.
 *
 * Offsets live in an Int32Array (4 B/line): a plain number[] stores each
 * offset as an 8-byte tagged SMI (Node builds don't enable pointer
 * compression) and its geometric growth leaves dead backing stores for the
 * GC. Files are capped at TEXT_MAX_BYTES, so offsets always fit int32.
 * Kept as its own small function so it optimizes cleanly, away from the
 * branchier chunk-assembly loops below.
 */
function scanLines(text: string): { starts: Int32Array; lineCount: number; hasLongLine: boolean } {
  let buf = new Int32Array(1024);
  let n = 1; // buf[0] = 0 via zero-initialization
  let hasLongLine = false;
  let prev = 0;
  let p = text.indexOf("\n");
  while (p !== -1) {
    if (p - prev > MAX_LINE_CHARS) hasLongLine = true;
    prev = p + 1;
    if (n === buf.length) {
      const grown = new Int32Array(buf.length * 2);
      grown.set(buf);
      buf = grown;
    }
    buf[n++] = prev;
    p = text.indexOf("\n", prev);
  }
  if (text.length - prev > MAX_LINE_CHARS) hasLongLine = true;
  return { starts: buf, lineCount: n, hasLongLine };
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
    // plus the newline, which lands exactly on the next line's start), with
    // text.length + 1 standing in for starts[lineCount] at the last window.
    // So the char-cap check is O(1) per chunk (a short binary search only
    // when the window actually exceeds MAX_CHUNK_CHARS) and no `endOf`
    // closure survives in any hot loop. The blank-line scan likewise
    // checks the first char inline and only calls isBlankRange for rows
    // that start with whitespace.
    let i = 0;
    while (i < lineCount) {
      let end = Math.min(i + maxLines, lineCount);
      for (let j = end - 1; j > i + 10 && j > end - 15; j--) {
        const s = starts[j];
        const e = j + 1 < lineCount ? starts[j + 1] - 1 : text.length;
        if (s === e) { end = j + 1; break; }
        const c = text.charCodeAt(s);
        if ((c === 32 || (c >= 9 && c <= 13) || c === 0xa0 || c === 0xfeff || c === 0x2028 || c === 0x2029)
          && isBlankRange(text, s, e)) { end = j + 1; break; }
      }
      // Shrink the window further if the joined content would exceed
      // MAX_CHUNK_CHARS (≈1k tokens — the embedding model's sweet spot).
      const spanEnd = end < lineCount ? starts[end] : text.length + 1;
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
      const to = end < lineCount ? starts[end] - 1 : text.length;
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
  // injected newlines), so rows are materialized and joined as before.
  const rowTexts: string[] = [];
  let rowNums = new Int32Array(1024); // 4 B/row, same rationale as `starts`
  const pushRow = (t: string, n: number) => {
    if (rowTexts.length === rowNums.length) {
      const grown = new Int32Array(rowNums.length * 2);
      grown.set(rowNums);
      rowNums = grown;
    }
    rowNums[rowTexts.length] = n;
    rowTexts.push(t);
  };
  for (let j = 0; j < lineCount; j++) {
    const s = starts[j];
    const e = j + 1 < lineCount ? starts[j + 1] - 1 : text.length;
    if (e - s <= MAX_LINE_CHARS) {
      pushRow(text.substring(s, e), j + 1);
    } else {
      for (let k = s; k < e; k += MAX_LINE_CHARS) {
        pushRow(text.substring(k, Math.min(k + MAX_LINE_CHARS, e)), j + 1);
      }
    }
  }

  let i = 0;
  while (i < rowTexts.length) {
    let end = Math.min(i + maxLines, rowTexts.length);
    for (let j = end - 1; j > i + 10 && j > end - 15; j--) {
      if (isBlankRow(rowTexts[j])) { end = j + 1; break; }
    }
    let len = rowTexts[i].length;
    for (let j = i + 1; j < end; j++) {
      len += 1 + rowTexts[j].length;
      if (len > MAX_CHUNK_CHARS) { end = j; break; }
    }
    const chunk = rowTexts.slice(i, end).join("\n");
    if (trimLenGT(chunk, 20)) {
      chunks.push({ content: chunk, lineStart: rowNums[i], lineEnd: rowNums[end - 1] });
    }
    i = end;
  }
  return chunks;
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
    return { text, hash: sha256(buf.toString("binary")), size: buf.length };
  }
  if (ext === ".docx") {
    const buf = readFileSync(fp);
    const { default: mammoth } = await import("mammoth");
    const { value } = await mammoth.extractRawText({ buffer: buf });
    return { text: value, hash: sha256(buf.toString("binary")), size: buf.length };
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
