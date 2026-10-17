// Сервис не отдаёт загруженное видео обратно, поэтому плеер играет локальный файл.
// Файл живёт в памяти вкладки: после перезагрузки его можно выбрать заново.

const files = new Map<string, File>();

export const rememberVideo = (jobId: string, file: File) => files.set(jobId, file);
export const recallVideo = (jobId: string) => files.get(jobId) ?? null;

// URL кэшируется на файл и не отзывается: файл всё равно держится в памяти вкладки,
// а отзыв в эффекте ломается двойным монтированием в StrictMode.
const urls = new WeakMap<File, string>();

// Chromium не берёт blob с типом video/quicktime, хотя сам контейнер MOV (H.264) читает как MP4.
export function playableUrl(file: File): string {
  let url = urls.get(file);
  if (!url) {
    const blob = file.type === "video/quicktime" || /\.mov$/i.test(file.name) ? new Blob([file], { type: "video/mp4" }) : file;
    url = URL.createObjectURL(blob);
    urls.set(file, url);
  }
  return url;
}
