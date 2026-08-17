/**
 * Small runtime helpers shared by several modules: event-loop yielding for
 * long CPU-bound phases and single-line stderr progress rendering.
 */

/**
 * Yield to the event loop so the TUI can render progress updates. ONNX
 * inference and other heavy phases are synchronous from the event loop's
 * perspective; without this, the UI freezes while they run.
 */
export function yieldToEventLoop(): Promise<void> {
  return new Promise<void>(resolve => setTimeout(resolve, 0));
}

/**
 * Overwrite the current stderr line with `message` (clears the line first).
 * Used for lightweight CLI progress when no TUI callbacks are available.
 */
export function writeProgressLineToStderr(message: string): void {
  process.stderr.write(`\r\x1b[2K${message}`);
}

/** Clear the current stderr progress line. */
export function clearProgressLineOnStderr(): void {
  process.stderr.write("\r\x1b[2K");
}
