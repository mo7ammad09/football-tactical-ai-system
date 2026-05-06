from src.identity.model_review import (
    build_identity_model_review_request,
    invoke_identity_review_provider,
    normalize_identity_review_model_outputs,
    resolve_identity_review_model_config,
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
    assert "API key" in result["model_outputs"][0]["reason"]
