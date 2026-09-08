/**
 * Path and store-scope helpers for presenting indexed files to the user
 * (and to the model) in cwd-relative form.
 */
import { homedir } from "node:os";
import { join, relative, isAbsolute, sep } from "node:path";
import { GLOBAL_RAG_DIR } from "../store-paths.ts";

/** "project" for a cwd-scoped store, "global" for the home-dir fallback. */
export function storeScope(ragDir: string): "global" | "project" {
  return ragDir === GLOBAL_RAG_DIR() ? "global" : "project";
}

/** True when `filePath` is `root` itself or nested beneath it. */
export function isUnderRoot(filePath: string, root: string): boolean {
  const pathRelativeToRoot = relative(root, filePath);
  return pathRelativeToRoot === ""
    || (pathRelativeToRoot !== ".." && !pathRelativeToRoot.startsWith(`..${sep}`) && !isAbsolute(pathRelativeToRoot));
}

/**
 * cwd-relative path for display (e.g. `src/myfile.php`); falls back to the
 * full path for files outside cwd so the model still gets a resolvable path.
 */
export function displayPath(filePath: string, cwd: string): string {
  return isUnderRoot(filePath, cwd) ? relative(cwd, filePath) : filePath;
}

/**
 * Expand a leading `~` / `~/…` to the user's home directory. Command args
 * reach the extension without shell expansion, so `/rag index ~/notes`
 * would otherwise resolve to a literal `<cwd>/~/notes` path.
 */
export function expandTildePath(pathArgument: string): string {
  if (pathArgument === "~") return homedir();
  if (pathArgument.startsWith("~/")) return join(homedir(), pathArgument.slice(2));
  return pathArgument;
}
