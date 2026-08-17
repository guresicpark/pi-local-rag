/**
 * Line-oriented text chunking.
 *
 * Documents are split into chunks of at most `maxLines` source lines,
 * preferring blank-line boundaries near the window end, and hard-capped at
 * MAX_CHUNK_CHARS. Pathologically long single lines (minified JS/JSON, CSV
 * rows, base64 blobs) are pre-split into MAX_LINE_CHARS segments that share
 * the source line's number so line references stay accurate.
 *
 * The implementation is allocation-conscious by design:
 *  - line boundaries are recorded as offsets in an Int32Array (two memchr
 *    passes over the string) instead of materializing per-line strings;
 *  - the common no-long-line fast path slices content straight out of the
 *    parent string (O(1) in V8, no bytes copied, no join);
 *  - the slow path describes segments with three parallel Int32Arrays and
 *    joins once per chunk.
 */
import { MAX_LINE_CHARS, MAX_CHUNK_CHARS } from "./constants.ts";

/** Minimum trimmed content length for a chunk to be worth indexing. */
const MIN_CHUNK_CONTENT_CHARS = 20;

/**
 * True for the exact character set String.prototype.trim removes
 * (WhiteSpace + LineTerminator per spec).
 */
function isTrimWhitespaceCharCode(code: number): boolean {
  return code === 32 || (code >= 9 && code <= 13) || code === 0xa0 || code === 0xfeff || code === 0x2028 || code === 0x2029;
}

/**
 * True when [from, to) of `text` contains no non-whitespace character —
 * like `text.slice(from, to).trim() === ""`, with no allocation and an exit
 * at the first non-whitespace char (usually char 0 for code lines).
 */
function rangeIsBlank(text: string, from: number, to: number): boolean {
  for (let i = from; i < to; i++) {
    if (!isTrimWhitespaceCharCode(text.charCodeAt(i))) return false;
  }
  return true;
}

/**
 * `text.slice(from, to).trim().length > n` without allocating anything —
 * walks both ends past whitespace, then compares the surviving span.
 */
function trimmedRangeLengthExceeds(text: string, from: number, to: number, minLength: number): boolean {
  let firstNonWs = from;
  let lastNonWs = to - 1;
  while (firstNonWs <= lastNonWs && isTrimWhitespaceCharCode(text.charCodeAt(firstNonWs))) firstNonWs++;
  if (firstNonWs > lastNonWs) return false;
  while (lastNonWs > firstNonWs && isTrimWhitespaceCharCode(text.charCodeAt(lastNonWs))) lastNonWs--;
  return lastNonWs - firstNonWs + 1 > minLength;
}

/** `text.trim().length > n` without allocating anything. */
function trimmedLengthExceeds(text: string, minLength: number): boolean {
  return trimmedRangeLengthExceeds(text, 0, text.length, minLength);
}

/**
 * Record line-start offsets with two indexOf("\n") scans — indexOf walks
 * the string with SIMD memchr and, unlike text.split("\n"), materializes
 * no per-line strings. Line j spans [starts[j], starts[j+1] - 1); the last
 * line ends at text.length.
 *
 * Offsets live in an Int32Array (4 B/line): a plain number[] stores each
 * offset as an 8-byte tagged SMI (Node builds don't enable pointer
 * compression) and its geometric growth leaves dead backing stores for the
 * GC. Files are capped at TEXT_MAX_BYTES, so offsets always fit int32.
 *
 * The array is sized exactly up front: a first memchr pass counts the lines
 * (and flags any over-long line), then a second pass records offsets into
 * that single, precisely-sized allocation — zero garbage and no
 * over-allocation. The extra memchr pass is cheap (~650 MB/s) and roughly
 * offsets the memcpy the old grow steps paid. Kept as its own small
 * function so it optimizes cleanly, away from the branchier chunk-assembly
 * loops below.
 */
function scanLineStartOffsets(text: string): { lineStarts: Int32Array; lineCount: number; hasLongLine: boolean } {
  // Pass 1: count lines + flag any line longer than MAX_LINE_CHARS. No
  // offsets are recorded — the buffer can't be sized until the count is
  // known, and a pure count pass keeps this loop free of array writes.
  let lineCount = 1;
  let hasLongLine = false;
  let lineBegin = 0;
  let newlineIndex = text.indexOf("\n");
  while (newlineIndex !== -1) {
    if (newlineIndex - lineBegin > MAX_LINE_CHARS) hasLongLine = true;
    lineCount++;
    lineBegin = newlineIndex + 1;
    newlineIndex = text.indexOf("\n", lineBegin);
  }
  if (text.length - lineBegin > MAX_LINE_CHARS) hasLongLine = true;

  // Pass 2: one allocation sized exactly for lineCount line starts plus the
  // sentinel slot at lineStarts[lineCount]. lineStarts[0] = 0 comes free
  // from zero-initialization.
  const lineStarts = new Int32Array(lineCount + 1);
  let nextSlot = 1;
  lineBegin = 0;
  newlineIndex = text.indexOf("\n");
  while (newlineIndex !== -1) {
    lineBegin = newlineIndex + 1;
    lineStarts[nextSlot++] = lineBegin;
    newlineIndex = text.indexOf("\n", lineBegin);
  }
  // Sentinel: lineStarts[lineCount] = text.length + 1, so the assembly
  // loops can read lineStarts[end] / lineStarts[j + 1] uniformly (no "last
  // line" ternary) — the last line then ends at lineStarts[lineCount] - 1
  // === text.length.
  lineStarts[lineCount] = text.length + 1;
  return { lineStarts, lineCount, hasLongLine };
}

/** One chunk of a document: its text plus the 1-based source line span. */
export interface TextChunk {
  content: string;
  lineStart: number;
  lineEnd: number;
}

/**
 * Split `text` into line-bounded chunks of at most `maxLines` lines each,
 * preferring blank-line breaks near the window end and capping each chunk
 * at MAX_CHUNK_CHARS. Chunks whose trimmed content is shorter than
 * MIN_CHUNK_CONTENT_CHARS are dropped (fully blank regions).
 */
export function chunkText(text: string, maxLines = 50): TextChunk[] {
  const chunks: TextChunk[] = [];
  const { lineStarts, lineCount, hasLongLine } = scanLineStartOffsets(text);

  if (!hasLongLine) {
    // Fast path (no line exceeds the cap — the overwhelmingly common case):
    // rows map 1:1 onto source lines and every chunk is a contiguous slice
    // of `text`, so content comes straight from substring() — in V8 an O(1)
    // sliced string sharing the parent, no bytes copied and no join needed.
    //
    // The joined length of rows [i, end) needs no separate prefix-sum array:
    // it is lineStarts[end] - lineStarts[i] - 1 (each line contributes its
    // length plus the newline, which lands exactly on the next line's
    // start); the sentinel at lineStarts[lineCount] covers the last window
    // with no per-chunk "last line" ternary. So the char-cap check is O(1)
    // per chunk (a short binary search only when the window actually
    // exceeds MAX_CHUNK_CHARS). The blank-line scan checks the first char
    // inline and only calls rangeIsBlank for rows that start with
    // whitespace.
    let lineIndex = 0;
    while (lineIndex < lineCount) {
      let windowEnd = Math.min(lineIndex + maxLines, lineCount);
      // Prefer breaking at a blank line near the window end (structural
      // boundary) — look back up to 14 rows, but never into the first 11
      // rows of the window (keeps chunks from collapsing to near-nothing).
      for (let candidate = windowEnd - 1; candidate > lineIndex + 10 && candidate > windowEnd - 15; candidate--) {
        const candidateStart = lineStarts[candidate];
        const candidateEnd = lineStarts[candidate + 1] - 1;
        if (candidateStart === candidateEnd) { windowEnd = candidate + 1; break; }
        const firstCharCode = text.charCodeAt(candidateStart);
        if (isTrimWhitespaceCharCode(firstCharCode) && rangeIsBlank(text, candidateStart, candidateEnd)) {
          windowEnd = candidate + 1;
          break;
        }
      }
      // Shrink the window further if the joined content would exceed
      // MAX_CHUNK_CHARS (≈1k tokens — the embedding model's sweet spot) via
      // binary search over the line-start offsets.
      const windowSpanEnd = lineStarts[windowEnd];
      if (windowSpanEnd - lineStarts[lineIndex] - 1 > MAX_CHUNK_CHARS) {
        const charBudgetBoundary = lineStarts[lineIndex] + MAX_CHUNK_CHARS + 1;
        let low = lineIndex + 1;
        let high = windowEnd;
        while (high - low > 1) {
          const mid = (low + high) >> 1;
          if (lineStarts[mid] > charBudgetBoundary) high = mid;
          else low = mid;
        }
        windowEnd = high - 1;
      }
      const contentStart = lineStarts[lineIndex];
      const contentEnd = lineStarts[windowEnd] - 1;
      if (trimmedRangeLengthExceeds(text, contentStart, contentEnd, MIN_CHUNK_CONTENT_CHARS)) {
        chunks.push({
          content: text.substring(contentStart, contentEnd),
          lineStart: lineIndex + 1,
          lineEnd: windowEnd,
        });
      }
      lineIndex = windowEnd;
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
  // source line number — replace a string[] of row texts (one sliced-string
  // object per row) and are sized exactly up front from the line offsets
  // instead of growing.
  let totalRowCount = 0;
  for (let lineIdx = 0; lineIdx < lineCount; lineIdx++) {
    const lineLength = lineStarts[lineIdx + 1] - 1 - lineStarts[lineIdx];
    totalRowCount += lineLength <= MAX_LINE_CHARS ? 1 : Math.ceil(lineLength / MAX_LINE_CHARS);
  }
  const rowStartOffsets = new Int32Array(totalRowCount);
  const rowEndOffsets = new Int32Array(totalRowCount);
  const rowSourceLineNumbers = new Int32Array(totalRowCount);
  let nextRow = 0;
  for (let lineIdx = 0; lineIdx < lineCount; lineIdx++) {
    const lineStart = lineStarts[lineIdx];
    const lineEnd = lineStarts[lineIdx + 1] - 1;
    const sourceLineNumber = lineIdx + 1;
    if (lineEnd - lineStart <= MAX_LINE_CHARS) {
      rowStartOffsets[nextRow] = lineStart;
      rowEndOffsets[nextRow] = lineEnd;
      rowSourceLineNumbers[nextRow] = sourceLineNumber;
      nextRow++;
    } else {
      for (let segmentStart = lineStart; segmentStart < lineEnd; segmentStart += MAX_LINE_CHARS) {
        rowStartOffsets[nextRow] = segmentStart;
        rowEndOffsets[nextRow] = Math.min(segmentStart + MAX_LINE_CHARS, lineEnd);
        rowSourceLineNumbers[nextRow] = sourceLineNumber;
        nextRow++;
      }
    }
  }

  let rowIndex = 0;
  while (rowIndex < totalRowCount) {
    let windowEnd = Math.min(rowIndex + maxLines, totalRowCount);
    // Prefer a blank row near the window end as the break point.
    for (let candidate = windowEnd - 1; candidate > rowIndex + 10 && candidate > windowEnd - 15; candidate--) {
      if (rangeIsBlank(text, rowStartOffsets[candidate], rowEndOffsets[candidate])) {
        windowEnd = candidate + 1;
        break;
      }
    }
    // Cap the joined chunk length at MAX_CHUNK_CHARS.
    let joinedLength = rowEndOffsets[rowIndex] - rowStartOffsets[rowIndex];
    for (let row = rowIndex + 1; row < windowEnd; row++) {
      joinedLength += 1 + rowEndOffsets[row] - rowStartOffsets[row];
      if (joinedLength > MAX_CHUNK_CHARS) { windowEnd = row; break; }
    }
    // Join the window's rows with newlines (segments of one source line
    // were originally contiguous, so this re-separates them harmlessly).
    const rowTexts: string[] = new Array(windowEnd - rowIndex);
    for (let row = rowIndex; row < windowEnd; row++) {
      rowTexts[row - rowIndex] = text.substring(rowStartOffsets[row], rowEndOffsets[row]);
    }
    const chunkContent = rowTexts.join("\n");
    if (trimmedLengthExceeds(chunkContent, MIN_CHUNK_CONTENT_CHARS)) {
      chunks.push({
        content: chunkContent,
        lineStart: rowSourceLineNumbers[rowIndex],
        lineEnd: rowSourceLineNumbers[windowEnd - 1],
      });
    }
    rowIndex = windowEnd;
  }
  return chunks;
}
