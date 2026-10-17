"use client";

import { useRef, useState, type Ref } from "react";
import type { Violation } from "@/lib/api";
import { fmt, isLow } from "@/lib/format";

const RATES = [1, 1.5, 2, 0.5];

type Props = {
  ref: Ref<HTMLVideoElement>;
  src: string | null;
  violations: Violation[];
  active: number;
  time: number;
  duration: number;
  onTime: (t: number) => void;
  onDuration: (d: number) => void;
  onSeek: (t: number) => void;
  onAttach: (f: File) => void;
};

export default function Player({ ref, src, violations, active, time, duration, onTime, onDuration, onSeek, onAttach }: Props) {
  const el = useRef<HTMLVideoElement | null>(null);
  const picker = useRef<HTMLInputElement>(null);
  const [playing, setPlaying] = useState(false);
  const [rate, setRate] = useState(1);
  const [failed, setFailed] = useState<string | null>(null);

  const setRefs = (v: HTMLVideoElement | null) => {
    el.current = v;
    if (typeof ref === "function") ref(v);
    else if (ref) ref.current = v;
  };

  const pct = (t: number) => `${Math.min(100, (t / duration) * 100)}%`;
  const times = violations.map((v) => v.timecode_sec).sort((a, b) => a - b);

  function jump(dir: 1 | -1) {
    const target = dir > 0 ? times.find((x) => x > time + 0.5) : [...times].reverse().find((x) => x < time - 0.5);
    if (target != null) onSeek(target);
  }

  function cycleRate() {
    const next = RATES[(RATES.indexOf(rate) + 1) % RATES.length];
    setRate(next);
    if (el.current) el.current.playbackRate = next;
  }

  const cur = active >= 0 ? violations[active] : null;

  return (
    <section className="player">
      <div className="stage">
        {src && failed !== src ? (
          <video
            ref={setRefs}
            src={src}
            playsInline
            muted
            preload="auto"
            onTimeUpdate={(e) => onTime(e.currentTarget.currentTime)}
            onLoadedMetadata={(e) => onDuration(e.currentTarget.duration)}
            onPlay={() => setPlaying(true)}
            onPause={() => setPlaying(false)}
            onError={() => setFailed(src)}
          />
        ) : (
          <div className="empty">
            <span>
              {src
                ? "Браузер не смог воспроизвести этот файл. Кадры-доказательства доступны в карточках справа."
                : "Видео не хранится на сервере. Выберите тот же файл, чтобы смотреть эпизоды в плеере."}
            </span>
            <button className="btn" onClick={() => picker.current?.click()}>Выбрать видео</button>
            <input ref={picker} type="file" accept="video/*,.mov" hidden
                   onChange={(e) => e.target.files?.[0] && onAttach(e.target.files[0])} />
          </div>
        )}
        {cur && (
          <div className={`flag${isLow(cur) ? " low" : ""}`}><span className="dot" /><span>{cur.type}</span></div>
        )}
      </div>
      <div className="controls">
        <div
          className="scrub"
          onClick={(e) => {
            const r = e.currentTarget.getBoundingClientRect();
            onSeek(((e.clientX - r.left) / r.width) * duration);
          }}
        >
          <div className="track" />
          <div className="fill" style={{ width: pct(time) }} />
          {violations.map((v, i) => (
            <div key={i} className={`mk${isLow(v) ? " low" : ""}`} style={{ left: pct(v.timecode_sec) }}
                 title={`${fmt(v.timecode_sec)} · ${v.type}`} />
          ))}
          <div className="head" style={{ left: pct(time) }} />
        </div>
        <div className="crow">
          <button disabled={!src} aria-label={playing ? "Пауза" : "Воспроизвести"}
                  onClick={() => el.current && (el.current.paused ? el.current.play() : el.current.pause())}>
            {playing ? "❚❚" : "▶"}
          </button>
          <button aria-label="Предыдущее нарушение" title="Предыдущее нарушение" onClick={() => jump(-1)}>⏮</button>
          <button aria-label="Следующее нарушение" title="Следующее нарушение" onClick={() => jump(1)}>⏭</button>
          <span className="t num">{fmt(time)} / {fmt(duration)}</span>
          <span className="grow" />
          <button className="num" style={{ fontSize: 13 }} disabled={!src} onClick={cycleRate}>{rate}×</button>
        </div>
      </div>
    </section>
  );
}
