from src.identity.model_review import (
    build_identity_model_review_request,
    classify_provider_error_text,
    invoke_identity_review_provider,
    normalize_identity_review_model_outputs,
    resolve_identity_review_model_config,
    sanitize_provider_error_text,
)


def test_phase11_builds_provider_neutral_model_review_request():
    request = build_identity_model_review_request(
        vision_review_queue={
            "cases": [
                {
                    "case_id": "candidate_goalkeeper_identity_fragmentation",
                    "question": "goalkeeper_identity_fragmentation",
                    "priority": "medium",
                }
            ]
        },
        player_crop_index={
            "cases": [
                {
                    "case_id": "candidate_goalkeeper_identity_fragmentation",
                    "contact_sheet_path": "/tmp/gk.jpg",
                    "target_count": 3,
                    "crop_requests": [
                        {
                            "crop_id": "c1",
                            "track_id": 21,
                            "raw_track_id": 114,
                            "source_frame_idx": 275,
                            "display_label": "GK",
                            "display_role": "goalkeeper",
                            "crop_path": "/tmp/c1.jpg",
                        }
                    ],
                }
            ]
        },
        provider="external_structured",
        model="test-reviewer",
    )

    assert request["phase"] == "phase_11_identity_model_review_request"
    assert request["provider"] == "external_structured"
    assert request["model"] == "test-reviewer"
    assert request["case_count"] == 1
    case = request["cases"][0]
    assert case["status"] == "ready_for_model_review"
    assert case["contact_sheet_path"] == "/tmp/gk.jpg"
    assert case["crop_count"] == 1
    assert case["output_schema"]["required"] == [
        "case_id",
        "status",
        "verdict",
        "confidence",
        "reason",
        "evidence",
    ]


def test_phase11_includes_audit_evidence_for_team_uncertain_review_cases():
    request = build_identity_model_review_request(
        vision_review_queue={
            "cases": [
                {
                    "case_id": "review_display_team_uncertain_18",
                    "question": "team_assignment_uncertain",
                    "priority": "high",
                    "reason": "Rendered team flickers on a player-dominant track.",
                    "audit_evidence": {
                        "track_id": 18,
                        "visible_team_counts": {"1": 145, "2": 141},
                        "player_team_confidence": 0.5069,
                    },
                }
            ]
        },
        player_crop_index={
            "cases": [
                {
                    "case_id": "review_display_team_uncertain_18",
                    "contact_sheet_path": "/tmp/team.jpg",
                    "target_count": 1,
                    "crop_requests": [
                        {
                            "crop_id": "c1",
                            "track_id": 18,
                            "source_frame_idx": 120,
                            "team": 1,
                            "display_team": 2,
                            "team_color": [27, 35, 31],
                            "display_color": [0, 0, 0],
                            "confidence": 0.82,
                            "crop_path": "/tmp/c1.jpg",
                        }
                    ],
                }
            ]
        },
        provider="google_gemma_api",
        model="gemma-test",
    )

    case = request["cases"][0]
    assert case["status"] == "ready_for_model_review"
    assert case["audit_evidence"]["track_id"] == 18
    assert "team_1" in case["prompt"]
    assert "team_1" in case["output_schema"]["properties"]["verdict"]["enum"]
    assert case["crop_evidence"][0]["display_team"] == 2
    assert case["crop_evidence"][0]["team_color"] == [27, 35, 31]


def test_phase11_normalizes_model_outputs_from_string_and_dict():
    outputs = normalize_identity_review_model_outputs(
        '{"results": [{"case_id": "case-1", "verdict": "same_player"}]}'
    )

    assert outputs == [{"case_id": "case-1", "verdict": "same_player"}]
    assert normalize_identity_review_model_outputs({"case_id": "case-2"}) == [
        {"case_id": "case-2"}
    ]
    assert normalize_identity_review_model_outputs("not json") == []


def test_phase11_resolves_model_config_when_outputs_are_supplied():
    config = resolve_identity_review_model_config(
        provider=None,
        model=None,
        provider_enabled=None,
        model_outputs=[{"case_id": "case-1"}],
    )

    assert config["provider_enabled"] is True
    assert config["provider"] == "external_structured"
    assert config["model"] == "identity-review-structured-v1"
    assert config["model_output_count"] == 1


def test_phase11_resolves_google_gemma_config_from_provider_and_key(monkeypatch):
    monkeypatch.setenv("GOOGLE_API_KEY", "test-key")

    config = resolve_identity_review_model_config(
        provider="google_gemma_api",
        model=None,
        provider_enabled=None,
        model_outputs=None,
    )

    assert config["provider_enabled"] is True
    assert config["provider"] == "google_gemma_api"
    assert config["model"] == "gemma-4-31b-it"
    assert config["model_output_count"] == 0


def test_phase11_resolves_openrouter_gemma_config_from_provider_and_key(monkeypatch):
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-key")

    config = resolve_identity_review_model_config(
        provider="openrouter",
        model=None,
        provider_enabled=None,
        model_outputs=None,
    )

    assert config["provider_enabled"] is True
    assert config["provider"] == "openrouter"
    assert config["model"] == "google/gemma-4-31b-it"
    assert config["model_output_count"] == 0


def test_phase11_invokes_google_gemma_provider(monkeypatch, tmp_path):
    contact_sheet = tmp_path / "sheet.jpg"
    contact_sheet.write_bytes(b"fake-jpeg")
    request = build_identity_model_review_request(
        vision_review_queue={
            "cases": [
                {
                    "case_id": "candidate_goalkeeper_identity_fragmentation",
                    "question": "goalkeeper_identity_fragmentation",
                    "priority": "medium",
                }
            ]
        },
        player_crop_index={
            "cases": [
                {
                    "case_id": "candidate_goalkeeper_identity_fragmentation",
                    "contact_sheet_path": str(contact_sheet),
                    "target_count": 2,
                    "crop_requests": [
                        {
                            "crop_id": "c1",
                            "track_id": 21,
                            "source_frame_idx": 275,
                            "display_label": "GK",
                            "crop_path": "/tmp/c1.jpg",
                        }
                    ],
                }
            ]
        },
        provider="google_gemma_api",
        model="gemma-test",
    )
    captured = {}

    class FakeResponse:
        def raise_for_status(self):
            return None

        def json(self):
            return {
                "candidates": [
                    {
                        "content": {
                            "parts": [
                                {
                                    "text": (
                                        '{"case_id":"candidate_goalkeeper_identity_fragmentation",'
                                        '"status":"reviewed","verdict":"same_player",'
                                        '"confidence":0.91,"reason":"visual match",'
                                        '"evidence":[{"type":"contact_sheet"}]}'
                                    )
                                }
                            ]
                        }
                    }
                ]
            }

    def fake_post(url, params, json, timeout):
        captured["url"] = url
        captured["params"] = params
        captured["json"] = json
        captured["timeout"] = timeout
        return FakeResponse()

    monkeypatch.setenv("GOOGLE_API_KEY", "test-key")
    monkeypatch.setattr("requests.post", fake_post)

    result = invoke_identity_review_provider(
        request=request,
        provider="google_gemma_api",
        model="gemma-test",
        provider_enabled=True,
        model_outputs=None,
    )

    assert result["status"] == "invoked"
    assert result["model_outputs"][0]["verdict"] == "same_player"
    assert "models/gemma-test:generateContent" in captured["url"]
    assert captured["params"]["key"] == "test-key"
    parts = captured["json"]["contents"][0]["parts"]
    assert parts[0]["text"].startswith("You are a cautious football identity-review model")
    assert "inline_data" in parts[1]


def test_phase11_invokes_openrouter_gemma_provider(monkeypatch, tmp_path):
    contact_sheet = tmp_path / "sheet.jpg"
    contact_sheet.write_bytes(b"fake-jpeg")
    request = build_identity_model_review_request(
        vision_review_queue={
            "cases": [
                {
                    "case_id": "review_display_team_flicker_2",
                    "question": "team_assignment_uncertain",
                    "priority": "high",
                }
            ]
        },
        player_crop_index={
            "cases": [
                {
                    "case_id": "review_display_team_flicker_2",
                    "contact_sheet_path": str(contact_sheet),
                    "target_count": 1,
                    "crop_requests": [
                        {
                            "crop_id": "c1",
                            "track_id": 2,
                            "source_frame_idx": 75,
                            "display_team": 1,
                            "crop_path": "/tmp/c1.jpg",
                        }
                    ],
                }
            ]
        },
        provider="openrouter",
        model="google/gemma-4-31b-it",
    )
    captured = {}

    class FakeResponse:
        def raise_for_status(self):
            return None

        def json(self):
            return {
                "choices": [
                    {
                        "message": {
                            "content": (
                                '{"case_id":"review_display_team_flicker_2",'
                                '"status":"reviewed","verdict":"team_1",'
                                '"confidence":0.94,"reason":"consistent team evidence",'
                                '"evidence":[{"type":"numeric_and_visual"}]}'
                            )
                        }
                    }
                ]
            }

    def fake_post(url, headers, json, timeout):
        captured["url"] = url
        captured["headers"] = headers
        captured["json"] = json
        captured["timeout"] = timeout
        return FakeResponse()

    monkeypatch.setenv("OPENROUTER_API_KEY", "test-openrouter-key")
    monkeypatch.setattr("requests.post", fake_post)

    result = invoke_identity_review_provider(
        request=request,
        provider="openrouter",
        model="google/gemma-4-31b-it",
        provider_enabled=True,
        model_outputs=None,
    )

    assert result["status"] == "invoked"
    assert result["model_outputs"][0]["verdict"] == "team_1"
    assert captured["url"] == "https://openrouter.ai/api/v1/chat/completions"
    assert captured["headers"]["Authorization"] == "Bearer test-openrouter-key"
    assert captured["json"]["model"] == "google/gemma-4-31b-it"
    content = captured["json"]["messages"][0]["content"]
    assert content[0]["type"] == "text"
    assert content[1]["type"] == "image_url"
    assert content[1]["image_url"]["url"].startswith("data:image/jpeg;base64,")


def test_phase11_google_gemma_missing_key_fails_closed(monkeypatch):
    monkeypatch.delenv("GOOGLE_API_KEY", raising=False)
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.delenv("GOOGLE_AI_API_KEY", raising=False)
    request = {
        "case_count": 1,
        "cases": [
            {
                "case_id": "case-1",
                "status": "ready_for_model_review",
                "prompt": "review",
            }
        ],
    }

    result = invoke_identity_review_provider(
        request=request,
        provider="google_gemma_api",
        model="gemma-test",
        provider_enabled=True,
        model_outputs=None,
    )

    assert result["status"] == "missing_api_key"
    assert result["model_outputs"][0]["verdict"] == "unresolved"
    assert result["model_outputs"][0]["failure_category"] == "provider_missing_or_invalid_key"
    assert "API key" in result["model_outputs"][0]["reason"]


def test_phase11_openrouter_missing_key_fails_closed(monkeypatch):
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    monkeypatch.delenv("OPENROUTER_KEY", raising=False)
    request = {
        "case_count": 1,
        "cases": [
            {
                "case_id": "case-1",
                "status": "ready_for_model_review",
                "prompt": "review",
            }
        ],
    }

    result = invoke_identity_review_provider(
        request=request,
        provider="openrouter",
        model="google/gemma-4-31b-it",
        provider_enabled=True,
        model_outputs=None,
    )

    assert result["status"] == "missing_api_key"
    output = result["model_outputs"][0]
    assert output["verdict"] == "unresolved"
    assert output["failure_category"] == "provider_missing_or_invalid_key"
    assert output["evidence"][1]["env"] == "OPENROUTER_API_KEY"


def test_phase11_sanitizes_google_provider_errors():
    text = sanitize_provider_error_text(
        "500 Server Error for url: "
        "https://generativelanguage.googleapis.com/v1beta/models/gemma:generateContent"
        "?key=AIzaSyDjbIjOuR8SJb_7Qq0rl3Lsq4ZWOmZE44c"
    )

    assert "AIza" not in text
    assert "key=" not in text
    assert "generativelanguage.googleapis.com/v1beta" not in text


def test_phase11_sanitizes_openrouter_provider_errors():
    text = sanitize_provider_error_text(
        "403 Forbidden for url: https://openrouter.ai/api/v1/chat/completions "
        "Authorization: Bearer sk-or-v1-secretvalue"
    )

    assert "sk-or-v1" not in text
    assert "openrouter.ai/api" not in text
    assert "OpenRouter API" in text


def test_phase11_classifies_common_provider_errors():
    assert classify_provider_error_text("429 Too Many Requests") == "provider_rate_limited"
    assert classify_provider_error_text("403 Forbidden") == "provider_forbidden"
    assert classify_provider_error_text("Read timed out") == "provider_timeout"


def test_phase11_retries_google_gemma_case_failures(monkeypatch):
    request = {
        "case_count": 1,
        "cases": [
            {
                "case_id": "case-1",
                "status": "ready_for_model_review",
                "prompt": "review",
            }
        ],
    }
    calls = {"count": 0}

    class FakeResponse:
        def raise_for_status(self):
            raise RuntimeError(
                "500 Server Error for url: "
                "https://generativelanguage.googleapis.com/v1beta/models/gemma-test"
                ":generateContent?key=AIzaSyDjbIjOuR8SJb_7Qq0rl3Lsq4ZWOmZE44c"
            )

    def fake_post(url, params, json, timeout):
        calls["count"] += 1
        return FakeResponse()

    monkeypatch.setenv("GOOGLE_API_KEY", "test-key")
    monkeypatch.setenv("IDENTITY_REVIEW_PROVIDER_RETRIES", "2")
    monkeypatch.setenv("IDENTITY_REVIEW_PROVIDER_RETRY_BACKOFF", "0")
    monkeypatch.setattr("requests.post", fake_post)

    result = invoke_identity_review_provider(
        request=request,
        provider="google_gemma_api",
        model="gemma-test",
        provider_enabled=True,
        model_outputs=None,
    )

    assert calls["count"] == 2
    output = result["model_outputs"][0]
    assert output["verdict"] == "unresolved"
    assert output["failure_category"] == "provider_error"
    assert "AIza" not in output["reason"]
    assert "key=" not in output["reason"]
