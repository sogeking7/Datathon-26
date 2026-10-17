"use client";

import Link from "next/link";
import { useParams } from "next/navigation";
import { useEffect, useState } from "react";
import Report from "@/components/Report";
import { getJob, type Job } from "@/lib/api";

const POLL_MS = 1500;

export default function JobPage() {
  const { id } = useParams<{ id: string }>();
  const [job, setJob] = useState<Job | null>(null);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    const ctl = new AbortController();
    let timer: ReturnType<typeof setTimeout>;
    const tick = async () => {
      try {
        const j = await getJob(id, ctl.signal);
        setJob(j);
        setError(null);
        if (j.status === "processing") timer = setTimeout(tick, POLL_MS);
      } catch (e) {
        if (ctl.signal.aborted) return;
        setError(e instanceof Error ? e.message : String(e));
        // 404 не исправится повтором; сетевые сбои — повторяем
        if (!(e instanceof Error && "status" in e && e.status === 404)) timer = setTimeout(tick, POLL_MS * 2);
      }
    };
    tick();
    return () => { ctl.abort(); clearTimeout(timer); };
  }, [id]);

  if (job?.status === "done") return <Report job={job} />;

  const progress = job?.status === "processing" ? job.progress ?? 0 : 0;
  return (
    <>
      <header className="top">
        <div className="crumbs"><Link href="/">Анализы</Link><span>/</span><b className="code">{id}</b></div>
        <div className="grow" />
        <Link className="btn primary" href="/">Новый анализ</Link>
      </header>
      <div className="upload">
        <section className="panel">
          {job?.status === "failed" ? (
            <div className="status err">Ошибка обработки: {job.error || "неизвестная ошибка"}</div>
          ) : error && !job ? (
            <div className="status err">{error}</div>
          ) : (
            <>
              <strong>Анализ видео…</strong>
              <div className="progress"><div style={{ width: `${progress * 100}%` }} /></div>
              <div className="status num">{Math.round(progress * 100)}%{error ? ` · ${error}, повторяем…` : ""}</div>
            </>
          )}
        </section>
      </div>
    </>
  );
}
