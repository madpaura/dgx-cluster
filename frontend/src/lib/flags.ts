/** Reading the extra-vLLM-flags box.
 *
 *  Shared by the deploy dialog and the catalog editor, because both had the same
 *  bug: `catch { return {} }`. That turns a trailing comma into a deploy that
 *  quietly runs with none of the flags you typed — the container starts, looks
 *  healthy, and serves at the wrong settings. A parse failure has to be loud.
 *
 *  Whether vLLM actually *accepts* a flag is a separate question, answered by the
 *  backend (services/vllm_args.py) once the JSON is readable. */

export interface ParsedFlags {
  value: Record<string, unknown>;
  /** Human-readable reason the text could not be used, or "" when it parsed. */
  error: string;
}

const EMPTY: ParsedFlags = { value: {}, error: "" };

export function parseFlags(text: string): ParsedFlags {
  if (!text.trim()) return EMPTY;

  let parsed: unknown;
  try {
    parsed = JSON.parse(text);
  } catch (e) {
    return { value: {}, error: tidy(e instanceof Error ? e.message : String(e)) };
  }

  if (parsed === null || typeof parsed !== "object" || Array.isArray(parsed)) {
    return {
      value: {},
      error: `Expected a JSON object mapping flags to values, got ${describe(parsed)}.`,
    };
  }
  return { value: parsed as Record<string, unknown>, error: "" };
}

function describe(v: unknown): string {
  if (v === null) return "null";
  if (Array.isArray(v)) return "an array";
  return `a ${typeof v}`;
}

/** Browsers phrase JSON errors differently; strip the noise, keep the position. */
function tidy(message: string): string {
  return message
    .replace(/^JSON\.parse:\s*/, "")
    .replace(/^Unexpected token (.+) in JSON at position (\d+).*$/, "Unexpected $1 at position $2")
    .replace(/\s+at JSON\.parse.*$/s, "")
    .trim();
}
