# YOLO11 Football Detector Training Plan

This document defines the first market-grade detector upgrade for Football Tactical AI.

## Decision

Start with **YOLO11l** as the primary high-quality detector.

Fallbacks:

- **YOLO11m**: if training/runtime cost is too high.
- **YOLO11x**: only after the dataset is strong enough and the extra cost is justified.

The detector replaces the current Abdullah YOLOv5-style model for object detection only. It does not, by itself, solve long-term player identity. It improves the input quality for tracking and identity systems.

## Classes

Train exactly these four classes first:

```text
0 player
1 goalkeeper
2 referee
3 ball
```

Do not add more classes in the first version unless they are consistently labeled across the dataset. Assistant referees should be labeled as `referee` in v1.

## Data Source Strategy

Yes, collect frames from every stadium and club environment you expect to serve, but the frames must come from **real match footage** whenever possible.

Good training data should cover:

- Target stadiums and camera angles.
- Day, night, shade, floodlights, and mixed lighting.
- Home and away kits.
- Similar kit colors.
- Crowded player clusters.
- Goalkeeper kits.
- Main referee and assistant referees.
- Ball at different sizes, speeds, and backgrounds.
- Wide broadcast camera and zoomed replays if those appear in your workflow.

Screenshots are acceptable for a first small batch, but sampled frames from video are better because they preserve the real camera distribution.

## Minimum Dataset Targets

For a serious first market test:

| Stage | Images | Purpose |
|---|---:|---|
| Smoke test | 300-500 | Verify label schema and training notebook |
| First useful model | 2,000-5,000 | Noticeable detector improvement |
| Commercial baseline | 10,000-20,000 | Strong stadium/kit generalization |
| Ball-focused expansion | +3,000 hard ball images | Reduce missed ball detections |

Split target:

```text
train 80%
val   15%
test   5%
```

Keep validation/test frames from different matches or time ranges than training. Do not put near-duplicate consecutive frames in both train and val.

## Annotation Rules

General:

- Draw tight boxes around visible object extent.
- Label every visible player, goalkeeper, referee, assistant referee, and ball.
- If an object is heavily cut off but still recognizable, label it.
- If a ball is too blurry to locate reliably, skip it rather than guessing.

Players:

- `player`: all outfield players.
- `goalkeeper`: goalkeeper even when far away or wearing similar colors.
- `referee`: center referee and assistant referees.

Ball:

- Use the smallest tight box that contains the ball.
- Include ball on grass, in air, near touchline, near players' feet, and in crowded boxes.
- Add extra ball examples from difficult frames. Ball quality usually limits football analysis.

## Local Frame Extraction

Put source videos in `input_videos/`, then extract annotation images:

```bash
python scripts/extract_training_frames.py input_videos \
  --output-dir training_data/raw_frames \
  --every-seconds 2 \
  --max-width 1920 \
  --max-frames-per-video 900
```

For fast first labeling:

```bash
python scripts/extract_training_frames.py input_videos/test_match.mp4 \
  --output-dir training_data/raw_frames \
  --every-seconds 1 \
  --max-frames-per-video 300
```

Upload the extracted images to an annotation tool such as CVAT, Roboflow, or Label Studio, then export in YOLO format.

## Expected Dataset Layout

After annotation/export:

```text
football_yolo11_dataset/
├── data.yaml
├── images/
│   ├── train/
│   ├── val/
│   └── test/
└── labels/
    ├── train/
    ├── val/
    └── test/
```

Use `config/yolo11_football_4class.yaml` as the canonical `data.yaml` template.

## Colab Training Defaults

Primary training command:

```python
from ultralytics import YOLO

model = YOLO("yolo11l.pt")
results = model.train(
    data="/content/football_yolo11_dataset/data.yaml",
    epochs=120,
    imgsz=1280,
    batch=-1,
    patience=30,
    cos_lr=True,
    close_mosaic=15,
    device=0,
    project="/content/runs/football_yolo11",
    name="yolo11l_4class_v1",
)
```

If Colab memory fails:

- Lower `imgsz` to `960`.
- Use `yolo11m.pt`.
- Set a fixed batch size such as `batch=8`.

## Acceptance Criteria

Before replacing the production model, compare the new model against the current Abdullah model on the same videos:

- Player detection recall improves in crowded scenes.
- Goalkeeper/referee class flicker decreases.
- Ball detection frames increase.
- Tracking ID switches caused by missed detections decrease.
- False detections outside the pitch stay controlled.

The new model should not be accepted only because mAP is higher. It must improve downstream tracking stability.

## Next Phases

After the detector:

1. Save raw tracklets and embeddings as reusable artifacts.
2. Build Identity Debug Report.
3. Add graph-based tracklet optimization.
4. Add jersey number OCR.
5. Add human-in-the-loop identity correction without re-running GPU inference.
