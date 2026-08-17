/**
 * Content hashing used for change detection (skip unchanged files) and for
 * stable chunk ids.
 *
 * Hashes are truncated to 12 hex characters — enough to avoid collisions at
 * realistic corpus sizes while keeping ids and the files table compact.
 */
import { createHash } from "node:crypto";

/** Hash a UTF-8 string; used for plain-text and HTML content. */
export function sha256(text: string): string {
  return createHash("sha256").update(text).digest("hex").slice(0, 12);
}

/**
 * Hash a binary buffer; used for PDF/DOCX so the *source file's* identity
 * (not the extracted text) drives skip-on-rebuild.
 *
 * The digest is streamed in 64 KiB latin-1 windows: update() UTF-8-encodes
 * each window exactly the way the historical one-shot
 * `sha256(buf.toString("binary"))` encoded the whole string, and SHA-256
 * digests concatenated updates identically — so stored hashes stay
 * compatible — while the transient string stays bounded at 64 KiB instead
 * of materializing a full copy of a multi-megabyte file per concurrent
 * indexing producer.
 */
export function sha256Buffer(buffer: Buffer): string {
  const hash = createHash("sha256");
  for (let offset = 0; offset < buffer.length; offset += 65_536) {
    hash.update(buffer.subarray(offset, Math.min(offset + 65_536, buffer.length)).toString("binary"));
  }
  return hash.digest("hex").slice(0, 12);
}
