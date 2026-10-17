"use client";

import Link from "next/link";
import { useEffect, useMemo, useRef, useState } from "react";
import type { Job } from "@/lib/api";
import { fmtPrecise, isLow, plural, span } from "@/lib/format";
import { playableUrl, recallVideo } from "@/lib/videoStore";
import Player from "./Player";
import Lanes from "./Lanes";
import ViolationCard, { type Review } from "./ViolationCard";

type DoneJob = Extract<Job, { status: "done" }>;
type Sort = "time" | "confidence";

// Эпизод «активен», пока плеер внутри его интервала (с запасом в секунду).
const PAD = 1;

function useReviews(jobId: string) {
  const key = `reviews:${jobId}`;
  // Отчёт рендерится только в браузере (после запроса задачи), localStorage доступен сразу.
  const [reviews, setReviews] = useState<Record<number, Review>>(() => {
    try { return JSON.parse(localStorage.getItem(key) || "{}"); } catch { return {}; }
  });
  const set = (i: number, r: Review | null) =>
    setReviews((prev) => {
      const next = { ...prev };
      if (r) next[i] = r; else delete next[i];
      try { localStorage.setItem(key, JSON.stringify(next)); } catch {}
      return next;
    });
  return [reviews, set] as const;
}

export default function Report({ job }: { job: DoneJob }) {
  const video = useRef<HTMLVideoElement>(null);
  const [video0] = useState(() => recallVideo(job.job_id));
  const [src, setSrc] = useState<string | null>(() => (video0 ? playableUrl(video0) : null));
  const [fileName, setFileName] = useState<string | null>(video0?.name ?? null);
  const [time, setTime] = useState(0);
  const [duration, setDuration] = useState(0);
  const [sort, setSort] = useState<Sort>("time");
  const [zoom, setZoom] = useState<string | null>(null);
  const [reviews, setReview] = useReviews(job.job_id);
  const dialog = useRef<HTMLDialogElement>(null);

  const list = job.violations;
  const fallbackDuration = Math.max(10, ...list.map((v) => span(v)[1] + 5));
  const total = duration || fallbackDuration;

  function attach(file: File) {
    setSrc(playableUrl(file));
    setFileName(file.name);
  }

  useEffect(() => {
    if (zoom) dialog.current?.showModal(); else dialog.current?.close();
  }, [zoom]);

  const active = list.findIndex((v) => {
    const [a, b] = span(v);
    return time >= a - PAD && time <= b + PAD;
  });

  const order = useMemo(() => {
    const idx = list.map((_, i) => i);
    return sort === "time"
      ? idx.sort((a, b) => list[a].timecode_sec - list[b].timecode_sec)
      : idx.sort((a, b) => list[b].confidence - list[a].confidence);
  }, [list, sort]);

  function seek(t: number) {
    setTime(t);
    if (video.current) video.current.currentTime = t;
    const i = list.findIndex((v) => { const [a, b] = span(v); return t >= a - PAD && t <= b + PAD; });
    if (i >= 0) document.getElementById(`v${i}`)?.scrollIntoView({ block: "nearest", behavior: "smooth" });
  }

  function download() {
    const data = { ...job, review: Object.fromEntries(Object.entries(reviews).map(([i, r]) => [list[+i].timecode_sec, { type_code: list[+i].type_code, decision: r }])) };
    const a = document.createElement("a");
    a.href = URL.createObjectURL(new Blob([JSON.stringify(data, null, 2)], { type: "application/json" }));
    a.download = `${job.job_id}.json`;
    a.click();
    URL.revokeObjectURL(a.href);
  }

  const high = list.filter((v) => !isLow(v)).length;
  const low = list.length - high;
  const reviewed = Object.keys(reviews).length;

  return (
    <>
      <header className="top">
        <div className="crumbs">
          <Link href="/">Анализы</Link><span>/</span>
          <b>{fileName ?? <span className="code">{job.job_id}</span>}</b>
        </div>
        <div className="grow" />
        <button className="btn" onClick={download}>Скачать JSON</button>
        <Link className="btn primary" href="/">Новый анализ</Link>
      </header>

      <main className="report">
        <Player
          ref={video}
          src={src}
          violations={list}
          active={active}
          time={time}
          duration={total}
          onTime={setTime}
          onDuration={setDuration}
          onSeek={seek}
          onAttach={attach}
        />

        <section>
          <div className="summary">
            <div className="verdict">
              {list.length ? (
                <>
                  <span className={`pill ${high ? "bad" : "warn"}`}><span className="dot" />
                    {high ? "Найдены нарушения" : "Нужна проверка"}</span>
                  <h1>
                    {list.length} {plural(list.length, "эпизод", "эпизода", "эпизодов")}
                    {low > 0 && high > 0 && `: ${high} с высокой уверенностью, ${low} на проверку`}
                  </h1>
                </>
              ) : (
                <>
                  <span className="pill ok"><span className="dot" />Нарушений не найдено</span>
                  <h1>Признаков нарушений не обнаружено</h1>
                </>
              )}
            </div>
            <div className="meta">
              {duration > 0 && <span>Длительность <b className="num">{fmtPrecise(duration)}</b></span>}
              {list.length > 0 && <span>Проверено <b className="num">{reviewed} из {list.length}</b></span>}
              <span>Задача <b className="code">{job.job_id}</b></span>
            </div>
            {list.length > 0 && <Lanes violations={list} duration={total} time={time} onSeek={seek} />}
            <p className="disclaimer">
              Результат — подсказка для проверяющего, а не окончательное заключение. Выводы, требующие невидимых
              на видео данных (проводки, сверки), сервис не делает.
            </p>
          </div>

          {list.length > 0 && (
            <>
              <div className="list-h">
                <h2>Нарушения</h2>
                <div className="seg-ctl" role="tablist">
                  <button role="tab" aria-selected={sort === "time"} className={sort === "time" ? "on" : ""} onClick={() => setSort("time")}>По времени</button>
                  <button role="tab" aria-selected={sort === "confidence"} className={sort === "confidence" ? "on" : ""} onClick={() => setSort("confidence")}>По уверенности</button>
                </div>
              </div>
              {order.map((i) => (
                <ViolationCard
                  key={i}
                  id={`v${i}`}
                  v={list[i]}
                  active={i === active}
                  review={reviews[i] ?? null}
                  onReview={(r) => setReview(i, r)}
                  onSeek={seek}
                  onZoom={setZoom}
                />
              ))}
            </>
          )}
        </section>
      </main>

      <dialog ref={dialog} className="zoom" onClick={() => setZoom(null)} onClose={() => setZoom(null)}>
        {/* eslint-disable-next-line @next/next/no-img-element -- base64-кадр */}
        {zoom && <img src={zoom} alt="Кадр-доказательство" />}
      </dialog>
    </>
  );
}

