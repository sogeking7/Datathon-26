import type { Violation } from "@/lib/api";
import { fmt, isLow, plural, span } from "@/lib/format";

type Props = { violations: Violation[]; duration: number; time: number; onSeek: (t: number) => void };

// Одна дорожка на тип нарушения, эпизоды этого типа — сегменты на ней.
export default function Lanes({ violations, duration, time, onSeek }: Props) {
  const pct = (t: number) => `${Math.min(100, (t / duration) * 100)}%`;
  const types = [...new Set(violations.map((v) => v.type_code))];

  return (
    <div className="lanes">
      <div className="lanes-h">
        <span>Таймлайн нарушений</span>
        <span>{violations.length} {plural(violations.length, "эпизод", "эпизода", "эпизодов")}</span>
      </div>
      {types.map((code) => {
        const items = violations.filter((v) => v.type_code === code);
        return (
          <div className="lane" key={code}>
            <div className="name" title={items[0].type}>{items[0].type}</div>
            <div className="bar">
              {items.map((v, i) => {
                const [a, b] = span(v);
                return (
                  <button key={i} className={`seg${isLow(v) ? " low" : ""}`} aria-label={`${fmt(v.timecode_sec)} · ${v.type}`}
                          style={{ left: pct(a), width: `calc(${pct(b - a)})` }} onClick={() => onSeek(v.timecode_sec)} />
                );
              })}
              <div className="ph" style={{ left: pct(time) }} />
            </div>
          </div>
        );
      })}
      <div className="ticks">
        <div />
        <div className="num">{[0, 0.25, 0.5, 0.75, 1].map((f) => <span key={f}>{fmt(duration * f)}</span>)}</div>
      </div>
    </div>
  );
}
