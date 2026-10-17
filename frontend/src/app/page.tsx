"use client";

import { useRouter } from "next/navigation";
import { useRef, useState } from "react";
import { uploadVideo } from "@/lib/api";
import { rememberVideo } from "@/lib/videoStore";

const ACCEPT = ".mov,.mp4,.m4v,.avi,.mkv,video/*";

export default function UploadPage() {
  const router = useRouter();
  const input = useRef<HTMLInputElement>(null);
  const [file, setFile] = useState<File | null>(null);
  const [over, setOver] = useState(false);
  const [progress, setProgress] = useState<number | null>(null);
  const [error, setError] = useState<string | null>(null);

  const busy = progress !== null;

  function pick(f: File | undefined) {
    if (!f || busy) return;
    setFile(f);
    setError(null);
  }

  async function submit() {
    if (!file) return;
    setError(null);
    setProgress(0);
    try {
      const id = await uploadVideo(file, setProgress);
      rememberVideo(id, file);
      router.push(`/jobs/${id}`);
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
      setProgress(null);
    }
  }

  return (
    <div className="upload">
      <h1>Контроль кассовой дисциплины</h1>
      <p className="sub">
        Загрузите видеозапись кассы. Сервис найдёт признаки возможных нарушений и покажет кадр, время и норму НБРК.
      </p>
      <section className="panel">
        <div
          className={`drop${over ? " over" : ""}`}
          role="button"
          tabIndex={0}
          onClick={() => input.current?.click()}
          onKeyDown={(e) => (e.key === "Enter" || e.key === " ") && input.current?.click()}
          onDragOver={(e) => { e.preventDefault(); setOver(true); }}
          onDragLeave={() => setOver(false)}
          onDrop={(e) => { e.preventDefault(); setOver(false); pick(e.dataTransfer.files[0]); }}
        >
          <p><strong>Перетащите видео сюда</strong> или нажмите, чтобы выбрать файл</p>
          <p className="muted">
            {file ? `${file.name} · ${(file.size / 1048576).toFixed(1)} МБ` : ".mov, .mp4 — до 300 МБ"}
          </p>
          <input ref={input} type="file" accept={ACCEPT} hidden onChange={(e) => pick(e.target.files?.[0])} />
        </div>
        <div className="upload-row">
          <button className="btn primary" disabled={!file || busy} onClick={submit}>Проверить</button>
          {busy && <span className="muted num">Загрузка видео… {Math.round(progress * 100)}%</span>}
        </div>
        {busy && <div className="progress"><div style={{ width: `${progress * 100}%` }} /></div>}
        {error && <div className="status err">Ошибка: {error}</div>}
      </section>
      <p className="note">
        Результат — подсказка для проверяющего, а не окончательное заключение. Видео обрабатывается локально.
      </p>
    </div>
  );
}
