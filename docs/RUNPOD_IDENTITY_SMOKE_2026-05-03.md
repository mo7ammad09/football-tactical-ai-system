# RunPod Identity Smoke - 2026-05-03

## Purpose

Verify that the RunPod Serverless endpoint is no longer running the known-bad
image that spread the goalkeeper label/color to field players, and confirm that
the pre-render identity correction artifacts are produced before final review
video rendering.

## Images

- Baseline safe image kept as reference:
  `ghcr.io/mo7ammad09/football-tactical-ai-runpod:sha-bbe8dec`
- Known-bad regression image found on the endpoint before this smoke:
  `ghcr.io/mo7ammad09/football-tactical-ai-runpod:sha-5f969e6`
- Current deployed candidate image:
  `ghcr.io/mo7ammad09/football-tactical-ai-runpod:sha-5a3dec2`

## Endpoint Update

- Endpoint: `football-analysis`
- Endpoint ID: `1hogxx8sb5ivq8`
- Template ID: `kj1wiwjabn`
- Previous template image: `sha-5f969e6`
- Updated template image: `sha-5a3dec2`
- Final workers setting restored after smoke:
  - `workersMin=0`
  - `workersMax=1`

During the smoke, `workersMax` was temporarily raised from `1` to `2` because
RunPod health reported one queued job with one throttled worker. After the job
moved into progress and completed, `workersMax` was restored to `1`.

## Smoke Jobs

### Before Endpoint Image Update

- Job ID: `fb33dd3b-df7e-4dce-9617-a41cff618e6c-e1`
- Result: completed, but only legacy artifacts were present:
  - `annotated_video_url`
  - `report_json_url`
  - `raw_tracklets_jsonl_url`
  - `identity_debug_json_url`
- Missing expected new artifacts:
  - render audits
  - correction candidates/plan/applied
  - vision review queue/results
  - final render identity manifest
  - player crop index/contact sheets

Conclusion: the endpoint was still running the old/bad image.

### After Endpoint Image Update

- Job ID: `4fe38c7f-9fab-4816-8ad5-9d9bcd690e39-e1`
- Result: completed successfully after the endpoint pulled the candidate image.
- Required artifact URLs were present:
  - `annotated_video_url`
  - `report_json_url`
  - `raw_tracklets_jsonl_url`
  - `identity_debug_json_url`
  - `identity_events_json_url`
  - `render_audit_before_json_url`
  - `render_audit_after_json_url`
  - `correction_candidates_json_url`
  - `correction_plan_json_url`
  - `correction_applied_json_url`
  - `vision_review_queue_json_url`
  - `vision_review_results_json_url`
  - `final_render_identity_manifest_json_url`
  - `player_crop_index_json_url`
  - `vision_contact_sheets_zip_url`

## Artifact Validation Summary

Final render identity manifest:

- `release_status`: `review_required`
- `output_identity_mode`: `review_output_with_identity_artifacts`
- `render_policy`: `produce_video_with_identity_review_artifacts`
- validator: `deterministic_phase7_final_render_validator`
- validation verdict: `PASS`
- validation reasons: `[]`

Render audit before correction:

- `raw_record_count`: `67`
- `person_record_count`: `65`
- `visible_goalkeeper_record_count`: `0`
- `gk_false_positive_segment_count`: `0`
- `simultaneous_goalkeeper_conflict_count`: `0`
- `unsafe_gk_display_spread_count`: `0`
- issue counts: `identity_debug_role_flicker=1`

Render audit after correction:

- `raw_record_count`: `67`
- `person_record_count`: `65`
- `visible_goalkeeper_record_count`: `0`
- `gk_false_positive_segment_count`: `0`
- `simultaneous_goalkeeper_conflict_count`: `0`
- `unsafe_gk_display_spread_count`: `0`
- issue counts: `identity_debug_role_flicker=1`

Correction and vision review:

- `applied_action_count`: `0`
- `vision_review.case_count`: `0`
- `vision_review.unresolved_count`: `0`

## Conclusion

The endpoint is now running the candidate image `sha-5a3dec2`, not the known-bad
`sha-5f969e6` image. The smoke job confirmed that the new pre-render identity
artifacts are produced and the final manifest validates with `PASS`.

This smoke was intentionally short and only verifies packaging/deployment and
artifact generation. A longer sample with known goalkeeper/player flicker should
still be run before treating the candidate as fully production-approved for full
matches.

## Rollback

If a regression appears, update template `kj1wiwjabn` back to the baseline image:

`ghcr.io/mo7ammad09/football-tactical-ai-runpod:sha-bbe8dec`

Do not roll back to `ghcr.io/mo7ammad09/football-tactical-ai-runpod:sha-5f969e6`.
