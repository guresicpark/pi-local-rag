/**
 * rag_kb_read resolver — map a human-typed note reference to an indexed file.
 *
 * Ported from pi-knowledge-search's kb-reader.ts (see that project's
 * LICENSE) and adapted to pi-local-rag's storage: the SQLite files table
 * stores absolute paths, and the source dirs come from the config's
 * trackedPaths. `buildIndexedFiles()` bridges the two into the in-memory
 * index shape the resolver expects.
 *
 * Accepts:
 *   - Absolute paths (must be inside a configured source dir).
 *   - Relative paths from the cwd or any source dir.
 *   - `[[wikilink]]` or `[[target|alias]]` forms.
 *   - Plain basenames with or without a known extension.
 *   - Fuzzy substring matches as a last resort.
 *
 * Intentionally does NOT hit the filesystem except to read the final resolved
 * file — all candidate paths come from the in-memory index.
 */

import * as fs from "node:fs";
import * as path from "node:path";

export interface IndexedFile {
  absPath: string;
  /** POSIX-style relative path from its source dir. */
  relPath: string;
  sourceDir: string;
}

export interface ResolveMatch {
  absPath: string;
  relPath: string;
  sourceDir: string;
  /** Why this match was selected (for disambiguation messages). */
  reason:
    | "absolute"
    | "relative-to-source"
    | "basename-exact"
    | "basename-ci"
    | "relpath-suffix"
    | "substring";
}

export interface ResolveResult {
  /** Matches in descending confidence. Empty when nothing matched. */
  matches: ResolveMatch[];
  /** True when exactly one high-confidence match was found. */
  unique: boolean;
  /** The normalized reference that was searched for. */
  normalizedRef: string;
  /** Optional subheading from a `[[note#section]]` reference. */
  subheading?: string;
}

/**
 * Strip wikilink wrappers and aliases. `[[Foo]]` → `Foo`, `[[a/b|display]]` → `a/b`,
 * `[[Foo#Heading]]` → `{ref: "Foo", subheading: "Heading"}`.
 */
export function normalizeRef(ref: string): { ref: string; subheading?: string } {
  let s = ref.trim();
  // Strip surrounding [[ ]] if present. Forgiving — also handles just `[[foo` etc.
  s = s.replace(/^\[\[/, "").replace(/\]\]$/, "");
  // Drop display alias after `|`.
  const pipeIdx = s.indexOf("|");
  if (pipeIdx !== -1) s = s.slice(0, pipeIdx);
  s = s.trim();
  // Pull out `#subheading` suffix.
  let subheading: string | undefined;
  const hashIdx = s.indexOf("#");
  if (hashIdx !== -1) {
    subheading = s.slice(hashIdx + 1).trim() || undefined;
    s = s.slice(0, hashIdx).trim();
  }
  return { ref: s, subheading };
}

const DEFAULT_EXTS = [".md", ".txt"];

/** Basename without extension, lowercased. */
function normBasename(p: string, exts: string[]): string {
  const base = path.basename(p);
  for (const ext of exts) {
    if (base.toLowerCase().endsWith(ext)) {
      return base.slice(0, base.length - ext.length).toLowerCase();
    }
  }
  return base.toLowerCase();
}

/** Does the string end with any of the given extensions (case-insensitive)? */
function hasKnownExt(s: string, exts: string[]): boolean {
  const lower = s.toLowerCase();
  return exts.some((e) => lower.endsWith(e));
}

export interface ResolveOptions {
  /** Known file extensions to consider when matching by basename (default: .md, .txt). */
  fileExtensions?: string[];
  /** Max matches to return. Default: 10. */
  maxMatches?: number;
  /** Optional cwd to resolve relative paths against. */
  cwd?: string;
}

export function resolveNote(
  rawRef: string,
  indexedFiles: IndexedFile[],
  opts: ResolveOptions = {}
): ResolveResult {
  // Normalize extensions to always have a leading dot — config loaders don't
  // enforce this and a bare "md" would break all the `ref + ext` comparisons.
  const exts = (opts.fileExtensions ?? DEFAULT_EXTS).map((e) =>
    e.startsWith(".") ? e : "." + e
  );
  const maxMatches = opts.maxMatches ?? 10;
  const { ref, subheading } = normalizeRef(rawRef);

  if (!ref) {
    return { matches: [], unique: false, normalizedRef: "", subheading };
  }

  const matches: ResolveMatch[] = [];
  const seen = new Set<string>();

  function push(m: ResolveMatch) {
    if (seen.has(m.absPath)) return;
    seen.add(m.absPath);
    matches.push(m);
  }

  // 1. Absolute path — only honor if it's an indexed file.
  if (path.isAbsolute(ref)) {
    const hit = indexedFiles.find((f) => f.absPath === ref);
    if (hit) push({ ...hit, reason: "absolute" });
  }

  // 2. Cwd-relative path — resolve and check against indexed files.
  if (opts.cwd && !path.isAbsolute(ref) && (ref.includes("/") || ref.includes(path.sep))) {
    const abs = path.resolve(opts.cwd, ref);
    const hit = indexedFiles.find((f) => f.absPath === abs);
    if (hit) push({ ...hit, reason: "absolute" });
  }

  // 3. Path relative to any source dir — try both "as-is" and "with extension".
  for (const f of indexedFiles) {
    const rel = f.relPath;
    if (rel === ref) {
      push({ ...f, reason: "relative-to-source" });
      continue;
    }
    if (hasKnownExt(ref, exts)) continue;
    for (const ext of exts) {
      if (rel === ref + ext) {
        push({ ...f, reason: "relative-to-source" });
        break;
      }
    }
  }

  // 4. Basename exact match (case-sensitive). Prefer files whose basename
  //    equals the reference (with or without a known extension).
  const refLower = ref.toLowerCase();
  const refBaseLower = normBasename(ref, exts);
  for (const f of indexedFiles) {
    const b = path.basename(f.relPath);
    if (b === ref || (!hasKnownExt(ref, exts) && exts.some((e) => b === ref + e))) {
      push({ ...f, reason: "basename-exact" });
    }
  }

  // 5. Basename case-insensitive.
  for (const f of indexedFiles) {
    const bLower = normBasename(f.relPath, exts);
    if (bLower === refBaseLower) {
      push({ ...f, reason: "basename-ci" });
    }
  }

  // 6. Relative-path suffix match — handles `[[subdir/foo]]` when the note is
  //    actually at `top/subdir/foo.md`.
  if (ref.includes("/")) {
    const targets = hasKnownExt(ref, exts) ? [ref] : exts.map((e) => ref + e);
    const targetsLower = targets.map((t) => t.toLowerCase());
    for (const f of indexedFiles) {
      const rLower = f.relPath.toLowerCase();
      for (const t of targetsLower) {
        if (rLower.endsWith("/" + t) || rLower === t) {
          push({ ...f, reason: "relpath-suffix" });
          break;
        }
      }
    }
  }

  // 7. Fuzzy substring — only if nothing better matched, and keep it tight.
  if (matches.length === 0) {
    for (const f of indexedFiles) {
      const bLower = normBasename(f.relPath, exts);
      if (bLower.includes(refBaseLower) || refBaseLower.includes(bLower)) {
        push({ ...f, reason: "substring" });
      } else if (f.relPath.toLowerCase().includes(refLower)) {
        push({ ...f, reason: "substring" });
      }
      if (matches.length >= maxMatches) break;
    }
  }

  // Confidence ordering is already "by step number" since we pushed in order.
  const trimmed = matches.slice(0, maxMatches);
  // Unique when step 1-5 produced exactly one hit. Step 6+ is never unique
  // (they're best-effort hints that still warrant a disambiguation prompt).
  const highConfidenceReasons: ResolveMatch["reason"][] = [
    "absolute",
    "relative-to-source",
    "basename-exact",
    "basename-ci",
    "relpath-suffix",
  ];
  const highConfidence = trimmed.filter((m) => highConfidenceReasons.includes(m.reason));
  const unique = highConfidence.length === 1;

  return { matches: trimmed, unique, normalizedRef: ref, subheading };
}

export interface ReadNoteOptions {
  /** Truncate output to this many bytes (post-UTF8). Default: 64 KiB. */
  maxBytes?: number;
}

export interface ReadNoteResult {
  path: string;
  content: string;
  truncated: boolean;
  totalBytes: number;
}

export function readNote(absPath: string, opts: ReadNoteOptions = {}): ReadNoteResult {
  const maxBytes = opts.maxBytes ?? 64 * 1024;
  const buf = fs.readFileSync(absPath);
  const total = buf.length;
  if (buf.length <= maxBytes) {
    return {
      path: absPath,
      content: buf.toString("utf-8"),
      truncated: false,
      totalBytes: total,
    };
  }
  // Back off from the cut point to the last UTF-8 boundary so we never emit a
  // partial codepoint. UTF-8 continuation bytes are 10xxxxxx (0x80–0xBF); walk
  // backwards until we find a byte that isn't a continuation.
  let end = maxBytes;
  while (end > 0 && (buf[end] & 0xc0) === 0x80) {
    end--;
  }
  const slice = buf.subarray(0, end);
  const text = slice.toString("utf-8");
  // Keep the truncation point clean — cut on a newline if one is nearby. When
  // there's no newline at all, `lastIndexOf` returns -1, which must not be
  // treated as a cut position (would drop the trailing UTF-16 code unit and
  // orphan a surrogate half).
  const lastNewline = text.lastIndexOf("\n");
  const newlineNearby = lastNewline >= 0 && lastNewline > end - 2048;
  const safe = newlineNearby ? text.slice(0, lastNewline) : text;
  return {
    path: absPath,
    content: safe,
    truncated: true,
    totalBytes: total,
  };
}

// ─── pi-local-rag index bridging ─────────────────────────────────────────────

export interface BuildIndexOptions {
  /** Absolute paths of every indexed file (from the SQLite files table). */
  paths: string[];
  /** Absolute source roots the files were indexed under (config.trackedPaths). */
  roots: string[];
  /** cwd used as the fallback source dir when no root covers a file. */
  cwd?: string;
}

/**
 * Turn the flat list of indexed absolute paths into `IndexedFile[]` entries
 * carrying a source dir + rel path, so resolveNote() can do relative-path and
 * basename matching. A file is attributed to its deepest covering tracked
 * root; files outside every root (legacy/untracked) fall back to `cwd`.
 */
export function buildIndexedFiles(opts: BuildIndexOptions): IndexedFile[] {
  const roots = opts.roots.map((root) => root.replace(/[/\\]+$/, ""));
  const fallbackRoot = opts.cwd ?? process.cwd();
  const files: IndexedFile[] = [];

  for (const absPath of opts.paths) {
    // The deepest covering root is the one whose rel path has the FEWEST
    // segments (e.g. for /v/Notes/Evergreen/hybrid-search.md, root /v/Notes
    // wins over root /v).
    let bestRoot = "";
    let bestDepth = Number.MAX_SAFE_INTEGER;
    for (const root of roots) {
      const rel = path.relative(root, absPath);
      if (rel === "" || rel === ".." || rel.startsWith(`..${path.sep}`) || path.isAbsolute(rel)) {
        continue;
      }
      const depth = rel.split(path.sep).length;
      if (depth < bestDepth) {
        bestDepth = depth;
        bestRoot = root;
      }
    }

    const sourceDir = bestRoot || fallbackRoot;
    // POSIX-style relPath — resolveNote() matches on forward slashes (and the
    // user's typed references), regardless of the host separator.
    const relPath = (bestRoot ? path.relative(bestRoot, absPath) : path.relative(fallbackRoot, absPath))
      .split(path.sep).join("/");
    files.push({ absPath, relPath, sourceDir });
  }

  return files;
}
