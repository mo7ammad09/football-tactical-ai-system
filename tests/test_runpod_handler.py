from pathlib import Path

import runpod.handler as runpod_handler


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
