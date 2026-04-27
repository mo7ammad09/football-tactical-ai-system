from src.api.runpod_serverless_client import RunPodServerlessClient


class DummyResponse:
    def __init__(self, payload, status_code=200):
        self._payload = payload
        self.status_code = status_code

    def raise_for_status(self):
        if self.status_code >= 400:
            raise RuntimeError(f"HTTP {self.status_code}")

    def json(self):
        return self._payload


class DummySession:
    def __init__(self):
        self.posts = []
        self.gets = []

    def post(self, url, headers=None, json=None, timeout=None):
        self.posts.append({"url": url, "headers": headers, "json": json, "timeout": timeout})
        return DummyResponse({"id": "rp-job-1"})

    def get(self, url, headers=None, timeout=None):
        self.gets.append({"url": url, "headers": headers, "timeout": timeout})
        return DummyResponse(
            {
                "id": "rp-job-1",
                "status": "COMPLETED",
                "output": {
                    "status": "completed",
                    "stats": {"processed_frames": 10},
                    "annotated_video_url": "https://cdn.example/output.mp4",
                    "warnings": [],
                    "confidence": {"field_calibration": 0.0},
                },
            }
        )


class DummyStorage:
    def __init__(self):
        self.uploaded = []

    def upload_file(self, local_path, **kwargs):
        self.uploaded.append({"local_path": local_path, **kwargs})
        return {"object_key": f"football-ai/video/{local_path}", "public_url": None}


def test_runpod_client_uploads_to_storage_and_submits_job(tmp_path):
    client = RunPodServerlessClient(
        api_key="key",
        endpoint_id="endpoint",
        storage_client=DummyStorage(),
        base_url="https://api.runpod.ai/v2",
    )
    dummy_session = DummySession()
    client.session = dummy_session

    video_path = tmp_path / "match.mp4"
    video_path.write_bytes(b"video")

    job_id = client.upload_video(
        str(video_path),
        analysis_fps=3,
        output_fps=30,
        max_frames=None,
        resize_width=1280,
        model_path="models/model.pt",
        identity_merge_map={430: 12},
        tracker_backend="strongsort",
    )

    assert job_id == "rp-job-1"
    assert dummy_session.posts[0]["url"] == "https://api.runpod.ai/v2/endpoint/run"
    payload = dummy_session.posts[0]["json"]["input"]
    assert payload["video_object_key"].endswith("/match.mp4")
    assert payload["analysis_fps"] == 3.0
    assert payload["output_fps"] == 30.0
    assert payload["resize_width"] == 1280
    assert payload["max_frames"] is None
    assert payload["model_path"] == "models/model.pt"
    assert payload["identity_merge_map"] == {430: 12}
    assert payload["tracker_backend"] == "strongsort"


def test_runpod_client_polls_completed_result():
    client = RunPodServerlessClient(
        api_key="key",
        endpoint_id="endpoint",
        storage_client=DummyStorage(),
    )
    client.session = DummySession()

    status = client.get_status("rp-job-1")
    result = client.get_results("rp-job-1")

    assert status["status"] == "completed"
    assert status["progress"] == 100
    assert result["stats"]["processed_frames"] == 10
