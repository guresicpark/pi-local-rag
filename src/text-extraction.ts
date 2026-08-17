/**
 * Reading and decoding files into UTF-8 text for chunking.
 *
 * Routing by extension:
 *  - .pdf      → unpdf (a maintained pdfjs-dist wrapper), with an OCR
 *                fallback (pdftoppm + tesseract) for image-only PDFs
 *  - .docx     → mammoth raw-text extraction
 *  - .html/htm → turndown (HTML → markdown), stripping script/style/nav/footer
 *  - anything else → plain UTF-8
 *
 * Hashes are computed over the raw bytes for binaries (so the source
 * file's identity drives skip-on-rebuild) and over the decoded text for
 * plain-text files.
 */
import { readFileSync, readdirSync, mkdtempSync, rmSync, writeFileSync } from "node:fs";
import { extname, basename, join } from "node:path";
import { tmpdir } from "node:os";
import { spawnSync } from "node:child_process";
import { sha256, sha256Buffer } from "./hashing.ts";
import { yieldToEventLoop, writeProgressLineToStderr } from "./runtime-utils.ts";

// pdfjs (bundled inside the PDF stack) routes warnings through console.log
// with a "Warning: " prefix. On real-world PDFs this fires thousands of
// times per document ("Ran out of space in font private use area", missing
// glyphs, …), and the worker bundle's verbosity is not externally
// configurable. Filtering console.log for the known pdfjs prefixes is the
// only reliable approach.
const PDFJS_LOG_PREFIX_PATTERN = /^(Warning|Info|Deprecated API usage):/;

/**
 * Temporarily replace console.log with a filter that swallows pdfjs
 * noise, run `operation`, then restore the original.
 */
async function withPdfjsConsoleSilenced<T>(operation: () => Promise<T>): Promise<T> {
  const originalLog = console.log;
  console.log = (...args: unknown[]) => {
    const firstArgument = args[0];
    if (typeof firstArgument === "string" && PDFJS_LOG_PREFIX_PATTERN.test(firstArgument)) return;
    originalLog(...args);
  };
  try {
    return await operation();
  } finally {
    console.log = originalLog;
  }
}

// ─── OCR fallback for image-based PDFs ───────────────────────────────────────

/** Result of probing the system for OCR tooling (cached after first probe). */
type OcrTooling = { available: false } | { available: true; langs: string };

let cachedOcrTooling: OcrTooling | undefined;
let ocrUnavailableLogged = false;

/**
 * One-shot probe for system pdftoppm + tesseract (with English or Japanese
 * traineddata). The result is cached for the process lifetime.
 */
export function getOcrTooling(): OcrTooling {
  if (cachedOcrTooling) return cachedOcrTooling;
  const pdftoppmProbe = spawnSync("pdftoppm", ["-v"]);
  const tesseractProbe = spawnSync("tesseract", ["--list-langs"], { encoding: "utf-8" });
  if (pdftoppmProbe.error || tesseractProbe.error) return (cachedOcrTooling = { available: false });
  // tesseract prints its language list on stderr in some builds, stdout in
  // others — check both.
  const combinedOutput = `${tesseractProbe.stdout || ""}\n${tesseractProbe.stderr || ""}`;
  const installedLangs = new Set(combinedOutput.split(/\r?\n/).map(line => line.trim()).filter(Boolean));
  const wantedLangs = ["jpn", "eng"].filter(lang => installedLangs.has(lang));
  if (!wantedLangs.length) return (cachedOcrTooling = { available: false });
  return (cachedOcrTooling = { available: true, langs: wantedLangs.join("+") });
}

/**
 * Render `pdfBytes` to PNGs via pdftoppm, OCR each page via tesseract, and
 * return the concatenated text. Hard limits: 200 pages, 60 s per page.
 */
async function ocrPdf(pdfBytes: Buffer, langs: string, displayLabel: string): Promise<string> {
  const MAX_OCR_PAGES = 200;
  const OCR_PAGE_TIMEOUT_MS = 60_000;
  const workDirectory = mkdtempSync(join(tmpdir(), "rag-ocr-"));
  try {
    const pdfPath = join(workDirectory, "input.pdf");
    writeFileSync(pdfPath, pdfBytes);
    const renderResult = spawnSync(
      "pdftoppm",
      ["-png", "-r", "200", pdfPath, join(workDirectory, "page")],
      { encoding: "utf-8" },
    );
    if (renderResult.status !== 0) return "";
    const pageImages = readdirSync(workDirectory)
      .filter(fileName => fileName.startsWith("page-") && fileName.endsWith(".png"))
      .sort();
    const pageCount = Math.min(pageImages.length, MAX_OCR_PAGES);
    if (pageImages.length > MAX_OCR_PAGES) {
      process.stderr.write(
        `\r\x1b[2K[rag] OCR ${displayLabel}: ${pageImages.length} pages, capping at ${MAX_OCR_PAGES}\n`,
      );
    }
    const pageTexts: string[] = [];
    for (let pageIndex = 0; pageIndex < pageCount; pageIndex++) {
      writeProgressLineToStderr(`[OCR ${pageIndex + 1}/${pageCount}] ${displayLabel}`);
      await yieldToEventLoop();
      const ocrResult = spawnSync(
        "tesseract",
        [join(workDirectory, pageImages[pageIndex]), "-", "-l", langs],
        { encoding: "utf-8", timeout: OCR_PAGE_TIMEOUT_MS, maxBuffer: 16 * 1024 * 1024 },
      );
      pageTexts.push(ocrResult.status === 0 ? (ocrResult.stdout ?? "") : "");
    }
    process.stderr.write("\r\x1b[2K");
    return pageTexts.join("\n\n");
  } finally {
    try { rmSync(workDirectory, { recursive: true, force: true }); } catch { /* temp cleanup is best-effort */ }
  }
}

/**
 * True if `text` looks too sparse for `pageCount` to be the real content
 * of the document — the heuristic that decides whether to fall back to OCR
 * (fewer than 50 extracted characters per page).
 */
export function isSparsePdfText(text: string, pageCount: number): boolean {
  return text.trim().length < 50 * Math.max(1, pageCount);
}

/** The result of decoding one file: text plus identity + size bookkeeping. */
export interface ExtractedText {
  text: string;
  hash: string;
  size: number;
}

/**
 * Read and decode a file into UTF-8 text (see the module comment for the
 * per-extension routing). Throws for unreadable files — callers count the
 * failures as skips.
 */
export async function extractText(filePath: string): Promise<ExtractedText> {
  const extension = extname(filePath).toLowerCase();

  if (extension === ".pdf") {
    const rawBytes = readFileSync(filePath);
    const { getDocumentProxy, extractText: extractPdfText } = await import("unpdf");
    // unpdf's bundled pdfjs insists on a plain Uint8Array (it rejects
    // Buffer instances) — wrap the buffer in a zero-copy Uint8Array view.
    const pdfBytes = new Uint8Array(rawBytes.buffer, rawBytes.byteOffset, rawBytes.byteLength);
    const pdfDocument = await getDocumentProxy(pdfBytes);
    const { totalPages, text } = await withPdfjsConsoleSilenced(() =>
      extractPdfText(pdfDocument, { mergePages: true }),
    );
    let extractedText = text;
    if (isSparsePdfText(extractedText, totalPages ?? 1)) {
      const ocrTooling = getOcrTooling();
      if (ocrTooling.available) {
        const ocrText = await ocrPdf(rawBytes, ocrTooling.langs, basename(filePath));
        if (ocrText.trim().length > extractedText.trim().length) extractedText = ocrText;
      } else if (!ocrUnavailableLogged) {
        ocrUnavailableLogged = true;
        process.stderr.write(
          `\r\x1b[2K[rag] OCR unavailable: install pdftoppm + tesseract (with jpn/eng traineddata) to index image PDFs\n`,
        );
      }
    }
    return { text: extractedText, hash: sha256Buffer(rawBytes), size: rawBytes.length };
  }

  if (extension === ".docx") {
    const rawBytes = readFileSync(filePath);
    const { default: mammoth } = await import("mammoth");
    const { value: rawText } = await mammoth.extractRawText({ buffer: rawBytes });
    return { text: rawText, hash: sha256Buffer(rawBytes), size: rawBytes.length };
  }

  if (extension === ".html" || extension === ".htm") {
    const { default: TurndownService } = await import("turndown");
    const rawHtml = readFileSync(filePath, "utf-8");
    const htmlToMarkdown = new TurndownService({
      headingStyle: "atx",
      codeBlockStyle: "fenced",
      // <br> becomes a hard line break instead of vanishing.
      blankReplacement: (_content, node) => node.tagName === "BR" ? "\n" : "",
    });
    htmlToMarkdown.remove(["script", "style"]);
    htmlToMarkdown.remove(["nav", "footer"]);
    const markdown = htmlToMarkdown.turndown(rawHtml);
    return { text: markdown, hash: sha256(rawHtml), size: rawHtml.length };
  }

  const plainText = readFileSync(filePath, "utf-8");
  return { text: plainText, hash: sha256(plainText), size: plainText.length };
}
