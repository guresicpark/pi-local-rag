/**
 * End-to-end verification of the rag_kb_read tool: registers the real tool set
 * through registerRagTools() and drives the rag_kb_read execute handler against
 * a seeded SQLite store, covering resolution tiers, disambiguation, empty
 * index, truncation, and PDF decoding.
 */
import { describe, it, expect, beforeEach, afterEach, afterAll } from "vitest";
import { mkdtempSync, writeFileSync, readFileSync, rmSync, mkdirSync } from "node:fs";
import { tmpdir } from "node:os";
import { join, dirname } from "node:path";
import type { ExtensionAPI } from "@earendil-works/pi-coding-agent";
import { registerRagTools } from "../src/extension/rag-tools.ts";
import { openDatabase } from "../src/database.ts";
import { insertChunk, upsertFile } from "../src/repository.ts";
import { defaultConfig, saveConfig } from "../src/config.ts";
import { sha256 } from "../src/hashing.ts";

const SAMPLE_PDF = readFileSync(join(__dirname, "fixtures", "sample.pdf"));

/** Captures the tool definitions registered by registerRagTools(). */
function captureTools(): Record<string, any> {
  const tools: Record<string, any> = {};
  const pi = { registerTool: (def: any) => { tools[def.name] = def; } };
  registerRagTools(pi as unknown as Pick<ExtensionAPI, "registerTool">);
  return tools;
}

/** Inserts a file row + one chunk per entry into the active store. */
function seedFiles(entries: Array<{ rel: string; content?: string; binary?: Buffer }>): void {
  const db = openDatabase();
  try {
    for (const entry of entries) {
      const absPath = join(process.env.PI_RAG_DIR!, entry.rel);
      mkdirSync(dirname(absPath), { recursive: true });
      if (entry.binary) writeFileSync(absPath, entry.binary);
      else writeFileSync(absPath, entry.content ?? "");
      const body = entry.binary ? entry.rel : entry.content ?? "";
      insertChunk(db, {
        id: sha256(absPath),
        filePath: absPath,
        content: body,
        lineStart: 1,
        lineEnd: 1,
        hash: sha256(body),
        indexedAt: new Date().toISOString(),
        tokens: 1,
      });
      upsertFile(db, absPath, sha256(body), 1, new Date().toISOString(), entry.binary?.length ?? Buffer.byteLength(body), true);
    }
  } finally {
    db.close();
  }
}

let tmp: string;
let savedRagDir: string | undefined;
let kbRead: any;

beforeEach(() => {
  tmp = mkdtempSync(join(tmpdir(), "pi-rag-kb-e2e-"));
  process.env.PI_RAG_DIR = tmp;
  saveConfig({ ...defaultConfig(), trackedPaths: [tmp] });
  kbRead = captureTools().rag_kb_read;
  expect(kbRead, "rag_kb_read tool should be registered").toBeTruthy();
});

afterEach(async () => {
  const { closeDbConn } = await import("../src/database.ts");
  closeDbConn();
  rmSync(tmp, { recursive: true, force: true });
});

afterAll(() => {
  if (savedRagDir !== undefined) process.env.PI_RAG_DIR = savedRagDir;
  else delete process.env.PI_RAG_DIR;
});

/** Invoke the rag_kb_read execute handler with a session cwd = the store root. */
async function call(name: string, maxBytes?: number): Promise<{ content: Array<{ text: string }>; details: any }> {
  const params = maxBytes === undefined ? { name } : { name, max_bytes: maxBytes };
  return kbRead.execute("call-1", params, undefined, undefined, { cwd: tmp });
}

describe("rag_kb_read tool end-to-end", () => {
  it("resolves a note by basename and returns its content with a header", async () => {
    seedFiles([{ rel: "notes/hybrid-search.md", content: "# Hybrid search\n\nFull body here.\n" }]);
    const result = await call("hybrid-search");
    const text = result.content[0].text;
    expect(text).toContain("hybrid-search.md");
    expect(text).toContain("Full body here.");
    expect(result.details.resolvedPath).toBe(join(tmp, "notes/hybrid-search.md"));
    expect(result.details.truncated).toBe(false);
  });

  it("resolves a relative path without an extension via tracked roots", async () => {
    seedFiles([{ rel: "docs/webhooks.md", content: "Webhook signing docs.\n" }]);
    const result = await call("docs/webhooks");
    expect(result.content[0].text).toContain("Webhook signing docs.");
    expect(result.details.resolvedPath).toBe(join(tmp, "docs/webhooks.md"));
  });

  it("resolves wikilink and piped-alias forms", async () => {
    seedFiles([{ rel: "notes/foo.md", content: "Foo note.\n" }]);
    const direct = await call("[[foo]]");
    expect(direct.content[0].text).toContain("Foo note.");
    const alias = await call("[[foo|Display]]");
    expect(alias.content[0].text).toContain("Foo note.");
  });

  it("extracts a subheading from a wikilink reference", async () => {
    seedFiles([{ rel: "notes/foo.md", content: "Body.\n" }]);
    const result = await call("[[foo#RRF]]");
    expect(result.content[0].text).toContain('section "RRF"');
  });

  it("returns a disambiguation prompt for ambiguous basenames", async () => {
    seedFiles([
      { rel: "notes/foo.md", content: "notes foo.\n" },
      { rel: "bar/foo.md", content: "bar foo.\n" },
    ]);
    const result = await call("foo");
    const text = result.content[0].text;
    expect(text).toContain('is ambiguous');
    expect(text).toContain("2 candidates");
    expect(result.details.candidates).toHaveLength(2);
  });

  it("returns a helpful message when nothing matches", async () => {
    seedFiles([{ rel: "notes/foo.md", content: "Foo note.\n" }]);
    const result = await call("nonexistent-note-xyz");
    expect(result.content[0].text).toContain('No indexed file matched "nonexistent-note-xyz"');
  });

  it("reports an empty index instead of resolving", async () => {
    const result = await call("anything");
    expect(result.content[0].text).toContain("pi-local-rag index is empty");
  });

  it("flags fuzzy (substring) matches instead of silently resolving", async () => {
    seedFiles([{ rel: "Evergreen/hybrid-search.md", content: "Deep note.\n" }]);
    // No file is named "Evergreen" — resolves only via substring.
    const result = await call("Evergreen");
    expect(result.content[0].text).toContain("Deep note.");
    expect(result.content[0].text).toContain("fuzzy match via substring");
  });

  it("truncates large files at max_bytes", async () => {
    seedFiles([{ rel: "big.md", content: "line\n".repeat(20000) }]); // ~100 KB
    const result = await call("big", 1024);
    const text = result.content[0].text;
    expect(result.details.truncated).toBe(true);
    expect(text).toContain("truncated: showing first");
    expect(text.length).toBeLessThanOrEqual(2048); // header + ≤1024-byte body
  });

  it("decodes a PDF through the extraction pipeline instead of raw bytes", async () => {
    seedFiles([{ rel: "docs/report.pdf", binary: SAMPLE_PDF }]);
    const result = await call("report.pdf");
    const text = result.content[0].text;
    expect(text).toContain("RagPdfMarker");
    expect(text).not.toContain("%PDF");
  });
});
