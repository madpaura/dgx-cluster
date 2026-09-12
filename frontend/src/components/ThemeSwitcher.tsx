import { useEffect, useState } from "react";
import { THEMES, applyTheme, loadTheme, type ThemeId } from "../lib/theme";

/** Three swatches in a pill. No popover, no menu state — the whole control is
 *  visible, and picking is one click. */
export function ThemeSwitcher() {
  const [theme, setTheme] = useState<ThemeId>(loadTheme);

  useEffect(() => {
    applyTheme(theme);
  }, [theme]);

  return (
    <div className="themer" role="group" aria-label="Colour theme">
      {THEMES.map((t) => (
        <button
          key={t.id}
          className={`swatch ${theme === t.id ? "on" : ""}`}
          title={`${t.label} — ${t.hint}`}
          aria-label={t.label}
          aria-pressed={theme === t.id}
          onClick={() => setTheme(t.id)}
        >
          <span style={{ background: t.swatch[0] }} />
          <span style={{ background: t.swatch[1] }} />
          <span style={{ background: t.swatch[2] }} />
        </button>
      ))}
    </div>
  );
}
