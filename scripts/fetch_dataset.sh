#!/usr/bin/env bash
# Скачивает датасет организатора из приватного релиза dataset-v1 в ./open.
# Нужен gh с доступом к репозиторию: gh auth login
set -euo pipefail

REPO="sogeking7/Datathon-26"
TAG="dataset-v1"
SHA256="4fe7920e2603e3df5e53a24175a6524a90794742267c9df2ec1e25d5ab8191bb"

cd "$(dirname "$0")/.."
if [ -d open/videos ] && [ "$(ls open/videos/*.mov 2>/dev/null | wc -l)" -ge 42 ]; then
  echo "Датасет уже на месте: open/videos"
  exit 0
fi

tmp=$(mktemp -d)
trap 'rm -rf "$tmp"' EXIT
for attempt in 1 2 3; do
  gh release download "$TAG" --repo "$REPO" --pattern open.zip --dir "$tmp" --clobber && break
  [ "$attempt" = 3 ] && { echo "Не удалось скачать архив" >&2; exit 1; }
  echo "Обрыв загрузки, повтор ($attempt/3)…" >&2
  sleep 5
done
echo "$SHA256  $tmp/open.zip" | shasum -a 256 -c -
unzip -q -o "$tmp/open.zip" -d .
echo "Готово: $(ls open/videos/*.mov | wc -l | tr -d ' ') видео в open/videos"
