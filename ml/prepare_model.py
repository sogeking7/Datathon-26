"""Скачивает VideoJev-4B и делает FP16-копию для T4 (BF16 на Turing не поддерживается).

python ml/prepare_model.py --out /models/videojev-4b-fp16
После этого интернет для обработки видео не нужен (требование переносимости из ТЗ).
"""

import argparse
import shutil
from pathlib import Path

REPO = "ZhuYuhan/VideoJev-4B"
REVISION = "2fc62181d25ed055104754b7884e17db4969c297"  # проверенный снапшот


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out", required=True)
    a = p.parse_args()
    from huggingface_hub import snapshot_download

    src = Path(snapshot_download(REPO, revision=REVISION))
    out = Path(a.out)
    if out.exists():
        shutil.rmtree(out)
    shutil.copytree(src, out, symlinks=False)
    f = out / "modeling_videojev.py"
    code = f.read_text()
    assert code.count("dtype=torch.bfloat16") == 2, "remote code изменился, проверьте патч"
    f.write_text(code.replace("dtype=torch.bfloat16", "dtype=torch.float16"))
    print(out)


if __name__ == "__main__":
    main()
