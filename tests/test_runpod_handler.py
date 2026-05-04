from pathlib import Path

import runpod.handler as runpod_handler
from src.identity.deployment_preflight import REQUIRED_RUNPOD_ARTIFACTS


def test_runpod_handler_accepts_legacy_video_url_without_storage(monkeypatch, tmp_path):
    model_path = tmp_path / "model.pt"
    model_path.write_bytes(b"model")
    captured_kwargs = {}

    def fake_download(_url: str, target_path: Path) -> None:
        target_path.parent.mkdir(parents=True, exist_ok=True)
        target_path.write_bytes(b"video")

    def fake_run_batch_analysis(**kwargs):
        captured_kwargs.update(kwargs)
        output_dir = Path(kwargs["output_dir"])
        output_dir.mkdir(parents=True, exist_ok=True)
        video = output_dir / "out.mp4"
        report_json = output_dir / "report.json"
        report_csv = output_dir / "report.csv"
        video.write_bytes(b"video")
        report_json.write_text("{}", encoding="utf-8")
        report_csv.write_text("id\n", encoding="utf-8")
        return {
            "report": {
                "status": "completed",
                "stats": {"processed_frames": 1},
                "warnings": [],
                "confidence": {"field_calibration": 0.0},
            },
            "paths": {
                "annotated_video": video,
                "report_json": report_json,
                "report_csv": report_csv,
            },
        }

    monkeypatch.setattr(runpod_handler, "DEFAULT_MODEL_PATH", str(model_path))
    monkeypatch.setattr(runpod_handler, "TMP_ROOT", tmp_path / "tmp")
    monkeypatch.setattr(runpod_handler, "ObjectStorageClient", lambda: (_ for _ in ()).throw(RuntimeError("no storage")))
    monkeypatch.setattr(runpod_handler, "_download_url", fake_download)
    monkeypatch.setattr(runpod_handler, "run_batch_analysis", fake_run_batch_analysis)

    result = runpod_handler.handler(
        {
            "id": "job-1",
            "input": {
                "video_url": "https://example.com/video.mp4",
                "identity_merge_map": {"430": 12},
                "tracker_backend": "strongsort",
            },
        }
    )

    assert result["status"] == "completed"
    assert result["stats"]["processed_frames"] == 1
    assert "annotated_video" in result["artifacts"]
    assert captured_kwargs["identity_merge_map"] == {"430": 12}
    assert captured_kwargs["tracker_backend"] == "strongsort"


def test_runpod_handler_uploads_all_identity_artifacts_when_present(monkeypatch, tmp_path):
    model_path = tmp_path / "model.pt"
    model_path.write_bytes(b"model")
    uploaded = []

    class FakeStorage:
        def upload_file(self, local_path, *, file_kind, job_id, content_type):
            uploaded.append(
                {
                    "local_path": local_path,
                    "file_kind": file_kind,
                    "job_id": job_id,
                    "content_type": content_type,
                }
            )
            name = Path(local_path).name
            return {
                "object_key": f"artifacts/{job_id}/{name}",
                "public_url": f"https://cdn.example/{job_id}/{name}",
            }

    def fake_download(_url: str, target_path: Path) -> None:
        target_path.parent.mkdir(parents=True, exist_ok=True)
        target_path.write_bytes(b"video")

    def fake_run_batch_analysis(**kwargs):
        output_dir = Path(kwargs["output_dir"])
        output_dir.mkdir(parents=True, exist_ok=True)
        paths = {}
        for artifact_name in REQUIRED_RUNPOD_ARTIFACTS:
            suffix = ".mp4" if artifact_name == "annotated_video" else ".zip" if artifact_name.endswith("_zip") else ".json"
            if artifact_name == "raw_tracklets_jsonl":
                suffix = ".jsonl"
            if artifact_name == "report_csv":
                suffix = ".csv"
            artifact_path = output_dir / f"{artifact_name}{suffix}"
            artifact_path.write_bytes(b"x")
            paths[artifact_name] = artifact_path
        return {
            "report": {
                "status": "completed",
                "stats": {"processed_frames": 1},
                "warnings": [],
                "confidence": {"field_calibration": 0.0},
            },
            "paths": paths,
        }

    for key in runpod_handler.REQUIRED_STORAGE_ENV:
        monkeypatch.setenv(key, f"{key.lower()}_value")
    monkeypatch.setattr(runpod_handler, "DEFAULT_MODEL_PATH", str(model_path))
    monkeypatch.setattr(runpod_handler, "TMP_ROOT", tmp_path / "tmp")
    monkeypatch.setattr(runpod_handler, "ObjectStorageClient", FakeStorage)
    monkeypatch.setattr(runpod_handler, "_download_url", fake_download)
    monkeypatch.setattr(runpod_handler, "run_batch_analysis", fake_run_batch_analysis)

    result = runpod_handler.handler(
        {
            "id": "job-artifacts",
            "input": {
                "video_url": "https://example.com/video.mp4",
            },
        }
    )

    assert result["status"] == "completed"
    assert set(REQUIRED_RUNPOD_ARTIFACTS).issubset(result["artifacts"])
    assert len(uploaded) == len(REQUIRED_RUNPOD_ARTIFACTS)
    assert result["final_render_identity_manifest_json_url"].endswith(
        "/final_render_identity_manifest_json.json"
    )
    assert result["vision_review_results_json_url"].endswith(
        "/vision_review_results_json.json"
    )
    assert result["identity_review_decisions_json_url"].endswith(
        "/identity_review_decisions_json.json"
    )
