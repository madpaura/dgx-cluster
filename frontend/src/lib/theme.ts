/** Theme registry.
 *
 *  Every theme is a block of CSS custom properties in styles.css keyed off
 *  `<html data-theme="…">`. Nothing in a component references a colour
 *  directly, so adding a theme is a CSS-only change plus one entry here.
 */
export const THEMES = [
  {
    id: "nexus",
    label: "Nexus",
    hint: "Cool grey canvas, indigo and teal",
    swatch: ["#eef0f4", "#4f46e5", "#14b8a6"],
  },
  {
    id: "crextio",
    label: "Crextio",
    hint: "Warm cream canvas, ink and yellow",
    swatch: ["#ece9e2", "#23221f", "#f5cf49"],
  },
  {
    id: "midnight",
    label: "Midnight",
    hint: "Dark, for wall displays and night shifts",
    swatch: ["#11141b", "#6366f1", "#22d3ee"],
  },
] as const;

export type ThemeId = (typeof THEMES)[number]["id"];

export const DEFAULT_THEME: ThemeId = "nexus";
const STORAGE_KEY = "dgxctl.theme";

function isTheme(value: unknown): value is ThemeId {
  return THEMES.some((t) => t.id === value);
}

export function loadTheme(): ThemeId {
  try {
    const stored = localStorage.getItem(STORAGE_KEY);
    if (isTheme(stored)) return stored;
  } catch {
    /* private mode, blocked storage — fall through to the default */
  }
  return DEFAULT_THEME;
}

export function applyTheme(id: ThemeId): void {
  document.documentElement.dataset.theme = id;
  try {
    localStorage.setItem(STORAGE_KEY, id);
  } catch {
    /* the theme still applies for this session */
  }
}
