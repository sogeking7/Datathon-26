import type { Metadata } from "next";
import "./globals.css";

export const metadata: Metadata = {
  title: "Контроль кассовой дисциплины",
  description: "Поиск визуальных признаков нарушений кассовой дисциплины на видеозаписях кассы",
};

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="ru">
      <body>{children}</body>
    </html>
  );
}
