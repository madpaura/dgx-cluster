import type { Finding } from "../types";

/** Failure explained in the operator's terms, with the fix stated outright.
 *  This is the difference between a dashboard and a log viewer. */
export function Findings({ items }: { items: Finding[] }) {
  if (!items.length) return null;
  return (
    <div className="mb">
      {items.map((f) => (
        <div key={f.code} className={`finding ${f.severity}`}>
          <div className="t">{f.title}</div>
          <div className="d">{f.detail}</div>
          <div className="fix">
            <b>Fix →</b> {f.fix}
          </div>
          {f.evidence && <div className="ev">{f.evidence}</div>}
        </div>
      ))}
    </div>
  );
}
