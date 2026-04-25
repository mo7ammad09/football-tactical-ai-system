from fastapi.testclient import TestClient

import server.main as server_main


def test_storage_health_reports_disabled_without_bucket(monkeypatch, tmp_path):
    monkeypatch.setattr(server_main, "GPU_API_KEY", "")
    monkeypatch.setattr(server_main, "OBJECT_STORAGE_BUCKET", "")
    monkeypatch.setattr(server_main, "JOB_DB_PATH", tmp_path / "jobs.sqlite3")

    with TestClient(server_main.app) as client:
        response = client.get("/storage/health")

    assert response.status_code == 200
    assert response.json()["enabled"] is False


def test_api_key_is_enforced_when_configured(monkeypatch, tmp_path):
    monkeypatch.setattr(server_main, "GPU_API_KEY", "secret")
    monkeypatch.setattr(server_main, "JOB_DB_PATH", tmp_path / "jobs.sqlite3")

    with TestClient(server_main.app) as client:
        unauthorized = client.get("/health")
        authorized = client.get("/health", headers={"Authorization": "Bearer secret"})

    assert unauthorized.status_code == 401
    assert authorized.status_code == 200
