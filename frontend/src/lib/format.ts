import type { Violation } from "./api";

// Ниже этого порога эпизод помечается «на проверку» (как в макете design/report).
export const LOW_CONFIDENCE = 0.6;

export const isLow = (v: Violation) => v.confidence < LOW_CONFIDENCE;

export const fmt = (s: number) => `${Math.floor(s / 60)}:${String(Math.floor(s % 60)).padStart(2, "0")}`;

export const fmtPrecise = (s: number) => `${Math.floor(s / 60)}:${(s % 60).toFixed(1).padStart(4, "0")}`;

export function plural(n: number, one: string, few: string, many: string) {
  const m10 = n % 10, m100 = n % 100;
  if (m10 === 1 && m100 !== 11) return one;
  if (m10 >= 2 && m10 <= 4 && (m100 < 12 || m100 > 14)) return few;
  return many;
}

// Интервал эпизода: от первого до последнего кадра-доказательства.
export function span(v: Violation): [number, number] {
  const ts = v.evidence_frames.map((f) => f.timestamp_sec).concat(v.timecode_sec);
  return [Math.min(...ts), Math.max(...ts)];
}

// В description сервис дописывает «Основание: …» — нормы показываем отдельным блоком.
export const observation = (v: Violation) => v.description.split(" Основание:")[0];
