import os
import tempfile
import time

os.environ["DATA_DIR"] = tempfile.mkdtemp()
os.environ["API_KEY"] = "test-key"

import cv2  # noqa: E402
import numpy as np  # noqa: E402
import pytest  # noqa: E402
from fastapi.testclient import TestClient  # noqa: E402

from app import analyzer  # noqa: E402
from app.analyzer import Detection  # noqa: E402
from app.main import app  # noqa: E402

client = TestClient(app)
KEY = {"X-API-Key": "test-key"}


@pytest.fixture(scope="module")
def video_path(tmp_path_factory):
    path = str(tmp_path_factory.mktemp("v") / "clip.mov")
    w = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*"mp4v"), 30, (160, 120))
    for i in range(90):  # 3 секунды
        frame = np.full((120, 160, 3), i * 2 % 255, np.uint8)
        w.write(frame)
    w.release()
    return path


def _wait(job_id, path="/api/v1/jobs", headers=KEY, timeout=20):
    deadline = time.time() + timeout
    while time.time() < deadline:
        r = client.get(f"{path}/{job_id}", headers=headers)
        assert r.status_code == 200
        if r.json()["status"] != "processing":
            return r.json()
        time.sleep(0.1)
    raise AssertionError("job did not finish")


def test_requires_api_key(video_path):
    with open(video_path, "rb") as f:
        assert client.post("/api/v1/analyze", files={"file": ("a.mov", f)}).status_code == 401
    assert client.get("/api/v1/jobs/abc", headers={"X-API-Key": "wrong"}).status_code == 401


def test_rejects_bad_extension():
    r = client.post("/api/v1/analyze", headers=KEY, files={"file": ("a.txt", b"hello")})
    assert r.status_code == 400


def test_unknown_job():
    assert client.get("/api/v1/jobs/nope123", headers=KEY).status_code == 404


def test_no_violation_flow(video_path):
    with open(video_path, "rb") as f:
        r = client.post("/api/v1/analyze", headers=KEY, files={"file": ("clip.mov", f)})
    assert r.status_code == 200
    job = _wait(r.json()["job_id"])
    assert job["status"] == "done"
    assert job["verdict"] == "no_violation"
    assert job["violations"] == []


def test_violation_payload_matches_spec(video_path, monkeypatch):
    class Fake:
        name = "fake"

        def run(self, path, info, progress):
            return [
                Detection("SAFE_OPEN", 1.0, "Дверца сейфа открыта.", 0.9, [1.0, 1.5]),
                Detection("SAFE_OPEN", 2.0, "Дубль рядом.", 0.6),  # схлопнется с первым
                Detection("DOOR_OPEN", 2.5, "Слабое.", 0.2),  # ниже порога
            ]

    monkeypatch.setattr(analyzer, "DETECTORS", [Fake()])
    with open(video_path, "rb") as f:
        r = client.post("/ui/analyze", files={"file": ("clip.mov", f)})
    job = _wait(r.json()["job_id"], path="/ui/jobs", headers={})
    assert job["verdict"] == "violation"
    assert len(job["violations"]) == 1
    v = job["violations"][0]
    assert v["timecode_sec"] == 1.0
    assert v["type"] and v["description"]
    assert "п. 20" in v["description"]
    assert 1 <= len(v["evidence_frames"]) <= 3
    assert v["evidence_frames"][0]["image_base64"].startswith("/9j/")  # JPEG


def test_broken_video_fails_gracefully():
    r = client.post("/api/v1/analyze", headers=KEY, files={"file": ("bad.mov", b"not a video")})
    job = _wait(r.json()["job_id"])
    assert job["status"] == "failed"
    assert job["error"]
