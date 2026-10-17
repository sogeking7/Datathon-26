import type { NextConfig } from "next";

// Бэкенд (FastAPI из service/). Браузер ходит только на этот Next-сервер,
// запросы /ui/* и /api/* проксируются, поэтому CORS на бэкенде не нужен.
const BACKEND_URL = process.env.BACKEND_URL ?? "http://localhost:8000";

const nextConfig: NextConfig = {
  output: "standalone",
  experimental: {
    // Видео до 300 МБ (MAX_UPLOAD_MB в сервисе) идут через прокси целиком.
    proxyClientMaxBodySize: "300mb",
  },
  async rewrites() {
    return [
      { source: "/ui/:path*", destination: `${BACKEND_URL}/ui/:path*` },
      { source: "/api/v1/:path*", destination: `${BACKEND_URL}/api/v1/:path*` },
      // Документация и healthcheck сервиса доступны и через фронтенд.
      { source: "/docs", destination: `${BACKEND_URL}/docs` },
      { source: "/openapi.json", destination: `${BACKEND_URL}/openapi.json` },
      { source: "/health", destination: `${BACKEND_URL}/health` },
    ];
  },
};

export default nextConfig;
