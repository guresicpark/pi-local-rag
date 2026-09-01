/**
 * rag_kb_read resolver tests — ported from pi-knowledge-search's kb-reader.test.ts
 * (adapted from node:test to vitest) plus buildIndexedFiles coverage for the
 * pi-local-rag tracked-paths → IndexedFile bridging.
 */
import { describe, it, expect } from "vitest";
import * as fs from "node:fs";
import * as os from "node:os";
import * as path from "node:path";
import {
  normalizeRef,
  resolveNote,
  readNote,
  buildIndexedFiles,
  type IndexedFile,
} from "../index.ts";

describe("normalizeRef", () => {
  it("strips [[ ]] wrappers", () => {
    expect(normalizeRef("[[Foo bar]]")).toEqual({ ref: "Foo bar", subheading: undefined });
  });

  it("takes the left side of a piped alias", () => {
    expect(normalizeRef("[[a/b|Display name]]")).toEqual({ ref: "a/b", subheading: undefined });
  });

  it("extracts #subheading", () => {
    expect(normalizeRef("[[Foo#Heading]]")).toEqual({ ref: "Foo", subheading: "Heading" });
  });

  it("handles plain bare references", () => {
    expect(normalizeRef("   just-a-name   ")).toEqual({ ref: "just-a-name", subheading: undefined });
  });

  it("handles forgiving unterminated wikilinks", () => {
    expect(normalizeRef("[[foo")).toEqual({ ref: "foo", subheading: undefined });
    expect(normalizeRef("foo]]")).toEqual({ ref: "foo", subheading: undefined });
  });
});

describe("resolveNote", () => {
  const files: IndexedFile[] = [
    { absPath: "/v/Notes/Evergreen/hybrid-search.md", relPath: "Notes/Evergreen/hybrid-search.md", sourceDir: "/v" },
    { absPath: "/v/Notes/Evergreen/Rosie architecture.md", relPath: "Notes/Evergreen/Rosie architecture.md", sourceDir: "/v" },
    { absPath: "/v/TaskNotes/Tasks/foo.md", relPath: "TaskNotes/Tasks/foo.md", sourceDir: "/v" },
    { absPath: "/other/foo.md", relPath: "foo.md", sourceDir: "/other" },
  ];

  it("resolves an absolute path that matches an indexed file", () => {
    const r = resolveNote("/v/Notes/Evergreen/hybrid-search.md", files);
    expect(r.unique).toBe(true);
    expect(r.matches[0].absPath).toBe("/v/Notes/Evergreen/hybrid-search.md");
    expect(r.matches[0].reason).toBe("absolute");
  });

  it("ignores absolute paths that aren't indexed", () => {
    const r = resolveNote("/random/elsewhere.md", files);
    expect(r.matches.length).toBe(0);
  });

  it("resolves a relative path to a source dir", () => {
    const r = resolveNote("Notes/Evergreen/hybrid-search.md", files);
    expect(r.unique).toBe(true);
    expect(r.matches[0].reason).toBe("relative-to-source");
  });

  it("resolves a relative path without extension", () => {
    const r = resolveNote("Notes/Evergreen/hybrid-search", files);
    expect(r.unique).toBe(true);
    expect(r.matches[0].absPath).toBe("/v/Notes/Evergreen/hybrid-search.md");
  });

  it("resolves an exact basename match", () => {
    const r = resolveNote("hybrid-search", files);
    expect(r.unique).toBe(true);
    expect(r.matches[0].reason).toBe("basename-exact");
  });

  it("resolves a case-insensitive basename", () => {
    const r = resolveNote("HYBRID-SEARCH", files);
    expect(r.unique).toBe(true);
  });

  it("resolves wikilink forms", () => {
    const r = resolveNote("[[hybrid-search]]", files);
    expect(r.unique).toBe(true);
    const r2 = resolveNote("[[Notes/Evergreen/hybrid-search|Hybrid]]", files);
    expect(r2.unique).toBe(true);
  });

  it("extracts subheading from wikilink", () => {
    const r = resolveNote("[[hybrid-search#RRF]]", files);
    expect(r.unique).toBe(true);
    expect(r.subheading).toBe("RRF");
  });

  it("handles filenames with spaces and mixed case", () => {
    const r = resolveNote("Rosie architecture", files);
    expect(r.unique).toBe(true);
    expect(r.matches[0].absPath).toBe("/v/Notes/Evergreen/Rosie architecture.md");
  });

  it("treats a unique relpath suffix as high confidence", () => {
    const r = resolveNote("Evergreen/hybrid-search", files);
    expect(r.unique, `expected unique for relpath suffix, got matches=${JSON.stringify(r.matches)}`).toBe(true);
    expect(r.matches[0].absPath).toBe("/v/Notes/Evergreen/hybrid-search.md");
  });

  it("returns multiple matches when the name is ambiguous", () => {
    const r = resolveNote("foo", files);
    expect(r.matches.length).toBe(2);
    expect(r.unique).toBe(false);
  });

  it("falls back to substring match as a last resort", () => {
    const r = resolveNote("Evergreen", files);
    // No file is literally named "Evergreen"; substring hits both evergreen notes.
    expect(r.matches.length).toBeGreaterThanOrEqual(2);
    expect(r.matches.every((m) => m.reason === "substring")).toBe(true);
    expect(r.unique).toBe(false);
  });

  it("returns empty result for an empty reference", () => {
    const r = resolveNote("", files);
    expect(r.matches.length).toBe(0);
  });

  it("returns empty result when nothing matches", () => {
    const r = resolveNote("nonexistent-note-xyz", files);
    expect(r.matches.length).toBe(0);
  });

  it("normalizes fileExtensions that lack a leading dot", () => {
    // Users who set fileExtensions to ["md"] (no dot) in the config file
    // used to break extension-stripping logic. resolveNote should tolerate it.
    const r = resolveNote("hybrid-search", files, { fileExtensions: ["md"] });
    expect(r.unique).toBe(true);
    expect(r.matches[0].absPath).toBe("/v/Notes/Evergreen/hybrid-search.md");
  });

  it("matches indexed code files by basename like notes (pi-local-rag allowlist)", () => {
    const codeFiles: IndexedFile[] = [
      { absPath: "/repo/src/payments.ts", relPath: "src/payments.ts", sourceDir: "/repo" },
      { absPath: "/repo/src/handlers/refunds.ts", relPath: "src/handlers/refunds.ts", sourceDir: "/repo" },
      { absPath: "/repo/docs/webhooks.md", relPath: "docs/webhooks.md", sourceDir: "/repo" },
    ];
    const exts = [".ts", ".md", ".txt"];
    const r = resolveNote("payments", codeFiles, { fileExtensions: exts });
    expect(r.unique).toBe(true);
    expect(r.matches[0].reason).toBe("basename-exact");
    expect(r.matches[0].absPath).toBe("/repo/src/payments.ts");
    const r2 = resolveNote("payments.ts", codeFiles, { fileExtensions: exts });
    expect(r2.unique).toBe(true);
    expect(r2.matches[0].absPath).toBe("/repo/src/payments.ts");
    const r3 = resolveNote("docs/webhooks", codeFiles, { fileExtensions: exts });
    expect(r3.unique).toBe(true);
    expect(r3.matches[0].reason).toBe("relative-to-source");
  });
});

describe("readNote", () => {
  it("reads a file in full when under the byte cap", () => {
    const tmp = fs.mkdtempSync(path.join(os.tmpdir(), "pi-rag-read-"));
    try {
      const p = path.join(tmp, "n.md");
      fs.writeFileSync(p, "Hello world\nSecond line\n");
      const r = readNote(p);
      expect(r.truncated).toBe(false);
      expect(r.content).toMatch(/Hello world/);
      expect(r.content).toMatch(/Second line/);
    } finally {
      fs.rmSync(tmp, { recursive: true, force: true });
    }
  });

  it("truncates large files and reports total bytes", () => {
    const tmp = fs.mkdtempSync(path.join(os.tmpdir(), "pi-rag-read-big-"));
    try {
      const p = path.join(tmp, "big.md");
      const body = "line\n".repeat(20000); // 100_000 bytes
      fs.writeFileSync(p, body);
      const r = readNote(p, { maxBytes: 1024 });
      expect(r.truncated).toBe(true);
      expect(r.content.length).toBeLessThanOrEqual(1024);
      expect(r.totalBytes).toBe(body.length);
    } finally {
      fs.rmSync(tmp, { recursive: true, force: true });
    }
  });

  it("never emits a partial UTF-8 codepoint at the truncation boundary", () => {
    const tmp = fs.mkdtempSync(path.join(os.tmpdir(), "pi-rag-read-utf8-"));
    try {
      const p = path.join(tmp, "utf8.md");
      // 4-byte codepoint (📚 = U+1F4DA) repeated enough to push past any byte cap.
      // No newlines so the "cut at last newline" fallback doesn't help.
      const body = "📚".repeat(4000);
      fs.writeFileSync(p, body);
      // Pick a maxBytes that lands mid-codepoint (4-byte char starts at multiples of 4,
      // so 1023 is guaranteed to be inside a codepoint).
      const r = readNote(p, { maxBytes: 1023 });
      expect(r.truncated).toBe(true);
      // No replacement character at the end — we backed off to a valid boundary.
      expect(r.content.includes("\uFFFD")).toBe(false);
      // Every codepoint is still the full book emoji.
      for (const c of r.content) {
        expect(c).toBe("📚");
      }
    } finally {
      fs.rmSync(tmp, { recursive: true, force: true });
    }
  });
});

describe("buildIndexedFiles", () => {
  it("attributes each file to its deepest covering tracked root", () => {
    const files = buildIndexedFiles({
      paths: [
        "/v/Notes/Evergreen/hybrid-search.md",
        "/v/Notes/TaskNotes/foo.md",
        "/v/other.md",
        "/unrelated/x.md",
      ],
      roots: ["/v", "/v/Notes"],
      cwd: "/v",
    });
    const byPath = new Map(files.map(f => [f.absPath, f]));
    expect(byPath.get("/v/Notes/Evergreen/hybrid-search.md")).toEqual({
      absPath: "/v/Notes/Evergreen/hybrid-search.md",
      relPath: "Evergreen/hybrid-search.md",
      sourceDir: "/v/Notes",
    });
    // /v/other.md is not under /v/Notes → deepest covering root is /v.
    expect(byPath.get("/v/other.md")!.sourceDir).toBe("/v");
    expect(byPath.get("/v/other.md")!.relPath).toBe("other.md");
    // Files outside every root fall back to cwd with a cwd-relative path.
    expect(byPath.get("/unrelated/x.md")!.sourceDir).toBe("/v");
  });

  it("tolerates trailing slashes on roots", () => {
    const files = buildIndexedFiles({
      paths: ["/repo/src/a.ts"],
      roots: ["/repo/"],
      cwd: "/",
    });
    expect(files[0].sourceDir).toBe("/repo");
    expect(files[0].relPath).toBe("src/a.ts");
  });

  it("round-trips through resolveNote for relative-path matching", () => {
    const files = buildIndexedFiles({
      paths: ["/v/Notes/Evergreen/hybrid-search.md", "/repo/src/payments.ts"],
      roots: ["/v", "/repo"],
      cwd: "/v",
    });
    const md = resolveNote("Evergreen/hybrid-search", files);
    expect(md.unique).toBe(true);
    expect(md.matches[0].absPath).toBe("/v/Notes/Evergreen/hybrid-search.md");
    const ts = resolveNote("src/payments", files, { fileExtensions: [".ts"] });
    expect(ts.unique).toBe(true);
    expect(ts.matches[0].absPath).toBe("/repo/src/payments.ts");
  });
});
