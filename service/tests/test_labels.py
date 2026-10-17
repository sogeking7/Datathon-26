import json
from pathlib import Path

import pytest

from app.rules import CATALOG

ROOT = Path(__file__).resolve().parents[2]
LABELS = ROOT / "analysis" / "labels.json"
VIDEOS = ROOT / "open" / "videos"
CONFIDENCE = {"high", "medium", "low"}

videos = json.loads(LABELS.read_text())["videos"]


def test_unique_files():
    names = [v["file"] for v in videos]
    assert len(names) == len(set(names)) == 42


@pytest.mark.parametrize("v", videos, ids=lambda v: v["file"])
def test_label_is_consistent(v):
    assert v["verdict"] in {"violation", "no_violation"}
    if v["verdict"] == "no_violation":
        assert "events" not in v and "timecode_sec" not in v
        return
    assert 0 <= v["timecode_sec"] <= v["duration_sec"]
    assert v["events"]
    for e in v["events"]:
        assert e["type_code"] in CATALOG
        assert all(c in CATALOG for c in e.get("alt_type_codes", []))
        assert e["confidence"] in CONFIDENCE
        assert 0 <= e["timecode_sec"] <= v["duration_sec"]
        assert e["observation"]


@pytest.mark.skipif(not VIDEOS.is_dir(), reason="датасет организатора не в репозитории")
def test_files_exist_locally():
    missing = [v["file"] for v in videos if not (VIDEOS / v["file"]).is_file()]
    assert not missing
