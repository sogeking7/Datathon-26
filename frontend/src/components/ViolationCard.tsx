import type { Violation } from "@/lib/api";
import { fmt, isLow, observation } from "@/lib/format";

export type Review = "confirmed" | "rejected";

type Props = {
  id: string;
  v: Violation;
  active: boolean;
  review: Review | null;
  onReview: (r: Review | null) => void;
  onSeek: (t: number) => void;
  onZoom: (src: string) => void;
};

export default function ViolationCard({ id, v, active, review, onReview, onSeek, onZoom }: Props) {
  const low = isLow(v);
  const cls = ["card", low && "low", active && "active", review === "rejected" && "rejected"].filter(Boolean).join(" ");
  const toggle = (r: Review) => onReview(review === r ? null : r);

  return (
    <article className={cls} id={id}>
      <div className="card-top">
        <button className="tc num" onClick={() => onSeek(v.timecode_sec)}>{fmt(v.timecode_sec)}</button>
        <div>
          <h3>{v.type}</h3>
          <span className="code">{v.type_code}</span>
          {low && <> · <span className="pill warn small">Нужна ручная проверка</span></>}
        </div>
        <div className="conf">
          Уверенность<b className="num">{Math.round(v.confidence * 100)}%</b>
          <div className="meter"><i style={{ width: `${v.confidence * 100}%` }} /></div>
        </div>
      </div>

      <div className="obs-row">
        <div className="frames">
          {v.evidence_frames.map((f) => {
            const src = `data:image/jpeg;base64,${f.image_base64}`;
            return (
              <button key={f.timestamp_sec} className={`frame${f.timestamp_sec === v.timecode_sec ? " key" : ""}`}
                      title="Перейти к кадру; двойной клик — увеличить"
                      onClick={() => onSeek(f.timestamp_sec)} onDoubleClick={() => onZoom(src)}>
                {/* eslint-disable-next-line @next/next/no-img-element -- base64-кадр, оптимизация next/image не нужна */}
                <img src={src} alt={`Кадр ${fmt(f.timestamp_sec)}`} />
                <span className="num">{fmt(f.timestamp_sec)}</span>
              </button>
            );
          })}
        </div>
        <div className="obs-wrap">
          <div className="lbl">Что видно на кадре</div>
          <p className="obs">{observation(v)}</p>
        </div>
      </div>

      {v.rules.map((r, i) => (
        <div className="norm" key={i}>
          <div className="lbl">Норма НБРК</div>
          <div className="norm-h">
            <b>{r.clause}</b>
            <span className="doc">{r.document}</span>
            <a href={r.url} target="_blank" rel="noopener noreferrer">adilet.zan.kz ↗</a>
          </div>
          <q>{r.quote}</q>
        </div>
      ))}

      <div className="actions">
        <button className={`btn${review === "confirmed" ? " on" : ""}`} onClick={() => toggle("confirmed")}>Подтвердить</button>
        <button className={`btn${review === "rejected" ? " on" : ""}`} onClick={() => toggle("rejected")}>Не нарушение</button>
      </div>
    </article>
  );
}
