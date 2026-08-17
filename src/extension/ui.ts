/**
 * TUI progress rendering for /rag index|rebuild: status-line updates,
 * block progress bars, per-model embed lines, and model-download notices.
 */
import { ANSI_RESET, ANSI_BOLD, ANSI_DIM, ANSI_GREEN, ANSI_CYAN, type EmbedGroup } from "../constants.ts";

/** The slice of ctx.ui the progress renderers touch. */
export interface RagUi {
  setStatus: (key: string, value: string | undefined) => void;
  setWidget: (key: string, value: string[] | undefined) => void;
  notify: (message: string, type?: "info" | "error" | "warning") => void;
}

/** Render a 24-cell block progress bar (cyan filled / dim empty). */
export function renderProgressBar(current: number, total: number, width = 24): string {
  const filledCells = Math.round((current / total) * width);
  return ANSI_CYAN + "█".repeat(filledCells) + ANSI_DIM + "░".repeat(width - filledCells) + ANSI_RESET;
}

/**
 * Shared embed-progress renderer for /rag index|rebuild: renders one line
 * per embedding model (code → jina, text → nomic) plus a combined bar, and
 * notifies when a model is about to be downloaded (a cold cache can stall
 * for minutes with no other visual feedback).
 */
export function createEmbedProgressRenderer(ctx: { ui: RagUi }, verb: string) {
  const progressByGroup: Partial<Record<EmbedGroup, { done: number; total: number }>> = {};
  return {
    onEmbed(done: number, total: number, group: EmbedGroup) {
      progressByGroup[group] = { done, total };
      const groups = (Object.keys(progressByGroup) as EmbedGroup[]).sort();
      const doneAll = groups.reduce((sum, g) => sum + progressByGroup[g]!.done, 0);
      const totalAll = groups.reduce((sum, g) => sum + progressByGroup[g]!.total, 0);
      const percent = totalAll ? Math.round((doneAll / totalAll) * 100) : 0;
      const bar = renderProgressBar(doneAll, totalAll);
      ctx.ui.setStatus("rag", `■ ${verb} ${percent}% │ ${doneAll}/${totalAll} chunks`);
      ctx.ui.setWidget("rag", [
        `${ANSI_BOLD}${ANSI_CYAN}${verb}${ANSI_RESET}  ${bar}  ${ANSI_GREEN}${percent}%${ANSI_RESET}`,
        ...groups.map(group =>
          `${ANSI_DIM}${group.padEnd(5)} ${ANSI_RESET}${progressByGroup[group]!.done}/${progressByGroup[group]!.total}${ANSI_DIM}  ${group === "code" ? "jina-code" : "nomic"}${ANSI_RESET}`),
      ]);
    },
    onModelLoad(group: EmbedGroup, model: string) {
      ctx.ui.notify(
        `⏳ Loading ${group} embedding model: ${model} — first run downloads it (this can take a few minutes)`,
        "info",
      );
    },
  };
}

/**
 * Shared file/embed/save progress renderer for /rag index|rebuild. The
 * two commands differ only in the active verb and the per-file "done"
 * label, so the callback bundle is built once and parameterized.
 */
export function createIndexProgressRenderer(ctx: { ui: RagUi }, verb: string, doneLabel: string) {
  return {
    onFile(current: number, total: number, filename: string, skipped: number) {
      const percent = Math.round((current / total) * 100);
      const bar = renderProgressBar(current, total);
      ctx.ui.setStatus("rag", `■ ${verb} ${percent}% │ ${current}/${total} files │ ${skipped} unchanged`);
      ctx.ui.setWidget("rag", [
        `${ANSI_BOLD}${ANSI_CYAN}${verb}${ANSI_RESET}  ${bar}  ${ANSI_GREEN}${percent}%${ANSI_RESET}`,
        `${ANSI_DIM}file:    ${ANSI_RESET}${filename}`,
        `${ANSI_DIM}done:    ${ANSI_RESET}${ANSI_GREEN}${current - skipped} ${doneLabel}${ANSI_RESET}  ${ANSI_DIM}${skipped} unchanged${ANSI_RESET}`,
      ]);
    },
    ...createEmbedProgressRenderer(ctx, "Embedding"),
    onChunk(fileChunk: number, totalChunks: number, filename: string) {
      ctx.ui.setStatus("rag", `■ Embedding ${filename} — chunk ${fileChunk}/${totalChunks}`);
    },
    onSave() {
      ctx.ui.setStatus("rag", `■ Saving index...`);
    },
  };
}
