// Типы и вызовы веб-эндпоинтов сервиса (/ui/*, без ключа). Схема ответа — app/analyzer.py.

export type Rule = { document: string; clause: string; quote: string; url: string };

export type EvidenceFrame = { timestamp_sec: number; image_base64: string };

export type Violation = {
  type: string;
  type_code: string;
  timecode_sec: number;
  description: string;
  confidence: number;
  rules: Rule[];
  evidence_frames: EvidenceFrame[];
};

export type Job =
  | { job_id: string; status: "processing"; progress?: number }
  | { job_id: string; status: "done"; verdict: "violation" | "no_violation"; violations: Violation[] }
  | { job_id: string; status: "failed"; error?: string | null };

export class ApiError extends Error {
  constructor(message: string, public status: number) {
    super(message);
  }
}

function detail(body: string, status: number): string {
  try {
    const d = JSON.parse(body).detail;
    if (typeof d === "string") return d;
  } catch {}
  return body || `HTTP ${status}`;
}

// XHR, а не fetch: нужен прогресс отправки файла.
export function uploadVideo(file: File, onProgress: (p: number) => void): Promise<string> {
  return new Promise((resolve, reject) => {
    const fd = new FormData();
    fd.append("file", file);
    const xhr = new XMLHttpRequest();
    xhr.open("POST", "/ui/analyze");
    xhr.upload.onprogress = (e) => e.lengthComputable && onProgress(e.loaded / e.total);
    xhr.onload = () => {
      if (xhr.status !== 200) return reject(new ApiError(detail(xhr.responseText, xhr.status), xhr.status));
      resolve(JSON.parse(xhr.responseText).job_id);
    };
    xhr.onerror = () => reject(new ApiError("Сервис недоступен", 0));
    xhr.send(fd);
  });
}

export async function getJob(id: string, signal?: AbortSignal): Promise<Job> {
  const r = await fetch(`/ui/jobs/${encodeURIComponent(id)}`, { signal, cache: "no-store" });
  const text = await r.text();
  if (!r.ok) throw new ApiError(detail(text, r.status), r.status);
  return JSON.parse(text);
}
