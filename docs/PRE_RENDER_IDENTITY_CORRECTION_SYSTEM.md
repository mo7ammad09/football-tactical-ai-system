# Pre-Render Identity Correction System

مرجع تصميم قبل التنفيذ لنظام تصحيح الهوية قبل إنتاج الفيديو النهائي.

> الحالة الحالية: لا تعتمد آخر image سيئة:
> `ghcr.io/mo7ammad09/football-tactical-ai-runpod:sha-5f969e6`
>
> سبب اعتبارها سيئة: ثبتت علامة الحارس `GK` جزئيا، لكنها نشرت وسم الحارس الوردي على مدافعين ولاعبين ليسوا حراسا. أي تصميم جديد يجب أن يحتوي على audit يمنع هذا النوع من التصحيح قبل الرندر النهائي.

Baseline المعتمد للرجوع والاختبار:

`ghcr.io/mo7ammad09/football-tactical-ai-runpod:sha-bbe8dec`

## الهدف

بناء مرحلة قبل الرندر النهائي اسمها:

`Pre-Render Identity Correction`

وظيفتها ليست منع الرندر فقط. وظيفتها الأساسية:

1. قراءة ملفات التحليل الناتجة من التتبع.
2. اكتشاف أخطاء الهوية والدور والفريق.
3. اقتراح تصحيحات قابلة للتنفيذ.
4. تطبيق التصحيحات الآمنة فقط على نسخة مؤقتة.
5. تشغيل audit بعد التصحيح.
6. إخراج الفيديو النهائي بهوية أكثر ثباتا ودقة.

## أين يعمل النظام؟

نعم، في النسخة العملية يجب أن تعمل الأجزاء الثقيلة كلها على RunPod.

### Local

الجزء المحلي يبقى خفيفا:

- Streamlit UI.
- رفع الفيديو والموديل إلى object storage.
- إرسال job إلى RunPod.
- متابعة حالة job.
- عرض الفيديو النهائي وملفات التقرير.
- عرض تحذيرات النظام للمستخدم.

### RunPod

RunPod يشغل كل العمل الثقيل:

- YOLO / detector.
- ByteTrack / tracker.
- ReID feature extraction.
- team assignment.
- role assignment.
- ball assignment.
- artifact generation.
- identity correction model.
- optional vision review.
- correction validator.
- final render.
- upload outputs back to storage.

السبب: تصحيح الهوية يحتاج نفس البيئة التي أنتجت التتبع: crops، embeddings، frame indices، raw track ids، وربما vision/OCR. تشغيل هذا كله في RunPod يقلل اختلاف البيئة ويمنع قرارات محلية ناقصة.

## النموذج المقترح

### الافتراضي

`Gemma 4 26B MoE`

يستخدم كـ text reasoning model يقرأ ملفات JSON/JSONL المختصرة والمنظمة ويخرج `correction_plan.json`.

سبب اختياره:

- مفتوح المصدر ومجاني من ناحية API.
- مناسب للتشغيل داخل RunPod.
- جيد للـ structured JSON reasoning.
- لا يحتاج إرسال بيانات المباراة إلى خدمة خارجية.
- قابل للتقييد بسكيما صارمة.

### نموذج مراجعة أثقل عند الحاجة

`Gemma 4 31B Dense`

لا يستخدم لكل مباراة كبداية. يستخدم كـ judge أو benchmark في الحالات الصعبة أو أثناء التطوير.

### Vision عند الحاجة فقط

Vision ليس المرحلة الأولى. يستخدم فقط للحالات الرمادية:

- لاعب اختفى فترة طويلة ثم عاد.
- ReID قريب من threshold.
- لاعبان من نفس الفريق ومتشابهان.
- حارس ضد مدافع.
- قراءة رقم القميص.
- تصحيح لا يستطيع JSON وحده حسمه.

## ما لا نريده

1. لا نعطي النموذج فيديو كامل ونطلب منه "حل كل شيء".
2. لا نعطي النموذج raw dump ضخم بدون تنظيم.
3. لا نسمح للنموذج بتعديل الفيديو مباشرة.
4. لا نطبق merge أو role override بدون validator.
5. لا نكرر خطأ `sha-5f969e6`: لا يجوز أن يتحول لاعب أو مدافع إلى `GK` فقط لأن جزءا قصيرا من tracklet تشابه مع الحارس.

## الفكرة الصحيحة

النموذج لا "يتخيل" الملعب. التعبير الأدق:

يبني reasoning منظم من أدلة مكانية وزمنية:

- bbox عبر الزمن.
- raw track id.
- track id.
- display id.
- role timeline.
- team timeline.
- ReID distances.
- gaps.
- overlap.
- crop quality.
- color evidence.
- render audit.

ثم ينتج خطة تصحيح محدودة وقابلة للتحقق.

## ملفات الإدخال الأساسية

### 1. `raw_tracklets.jsonl`

ملف كل ظهور لكل شخص أو كرة.

حقول مهمة:

- `source_frame_idx`
- `track_id`
- `raw_track_id`
- `merged_from_id`
- `bbox`
- `confidence`
- `object_type`
- `role`
- `display_role`
- `team`
- `display_team`
- `display_label`
- `has_ball`
- `reid_available`
- `reid_dim`

مهم: وجود `reid_available=true` لا يعني أن النموذج يرى embedding نفسه. النظام يجب أن يحسب distances وtop-k ويقدمها للنموذج في ملفات منفصلة.

### 2. `identity_debug.json`

ملف تشخيص الهوية الحالي.

حقول مهمة:

- `candidate_links`
- `accepted_auto_links`
- `manual_merge_map`
- `auto_merge_map`
- `rejected_examples`
- `reject_reason_counts`
- `role_stability`
- `tracklet_profiles`
- `warnings`

مهم: `candidate_links` وحده غير كاف. هو فقط ما سمحت له thresholds الحالية بالظهور. نحتاج `reid_topk.json` لرؤية near misses.

## ملفات جديدة مطلوبة قبل النموذج

### 1. `identity_events.json`

يسجل أحداث الهوية بدل أن يقرأ النموذج كل الصفوف الخام.

مثال schema:

```json
{
  "event_id": "role_flicker_14_958_1060",
  "event_type": "role_change",
  "severity": "high",
  "track_id": 14,
  "raw_track_id": 36,
  "first_source_frame_idx": 958,
  "last_source_frame_idx": 1060,
  "before": {"role": "player", "team": 1},
  "after": {"role": "goalkeeper", "team": 0},
  "evidence": {
    "frames_seen": 24,
    "dominant_track_role": "player",
    "track_role_confidence": 0.9
  }
}
```

أنواع الأحداث:

- `role_change`
- `team_change`
- `display_label_change`
- `short_tracklet`
- `identity_gap`
- `candidate_merge`
- `rejected_merge`
- `gk_conflict`
- `raw_track_fragmentation`
- `same_identity_overlap`

### 2. `reid_topk.json`

يعرض أقرب tracklets لكل tracklet، حتى لو لم تدخل في `candidate_links`.

مثال schema:

```json
{
  "source_id": 37,
  "source_role": "goalkeeper",
  "source_team": 0,
  "neighbors": [
    {
      "target_id": 18,
      "reid_distance": 0.2403,
      "gap_source_frames": 255,
      "overlap": false,
      "position_distance": 6.4,
      "role_pair": ["player", "goalkeeper"],
      "team_pair": [1, 0],
      "reid_count": 3,
      "color_count": 3,
      "verdict_hint": "gray_needs_review"
    }
  ]
}
```

هذا الملف مهم لحالات اللاعب الذي يختفي 40 ثانية ثم يعود. النموذج يحتاج كل المرشحين الأقرب، لا المرشحين الذين نجوا من threshold قديم فقط.

### 3. `role_logits_timeline.json`

بدل role النهائي فقط، نخزن ثقة الأدوار عبر الزمن.

مثال:

```json
{
  "track_id": 14,
  "segments": [
    {
      "first_source_frame_idx": 958,
      "last_source_frame_idx": 1060,
      "raw_track_id": 36,
      "role_scores": {
        "player": 0.42,
        "goalkeeper": 0.51,
        "referee": 0.07
      }
    }
  ]
}
```

هذا يمنع قرارا قاسيا مبنيا على label واحد فقط.

### 4. `team_color_palette.json`

يلخص ألوان الفرق والحارس والحكم.

مثال:

```json
{
  "teams": {
    "1": {"main_colors": ["#ffffff", "#e6e6e6"], "confidence": 0.91},
    "2": {"main_colors": ["#1b4f9c", "#102a54"], "confidence": 0.94}
  },
  "goalkeeper_candidates": [
    {"color": "#111111", "team_hint": 1, "confidence": 0.78}
  ],
  "referee_candidates": [
    {"color": "#f2d100", "confidence": 0.84}
  ]
}
```

مهم: لون أسود لا يعني حارس دائما. يجب أن يمر عبر قيود الموقع، التكرار، role timeline، وعدم وجود أكثر من حارس لنفس الفريق في نفس اللحظة.

### 5. `render_audit.json`

هذا أهم ملف لمنع تكرار مشكلة image السيئة.

يقيس ما سيظهر في الفيديو النهائي، لا ما في البيانات فقط.

أمثلة metrics:

- `gk_false_positive_segments`
- `simultaneous_goalkeepers_by_team`
- `goalkeeper_display_spread`
- `role_flicker_tracklets`
- `team_flicker_tracklets`
- `display_label_switches`
- `same_identity_overlap`
- `short_identity_segments`
- `unsafe_gk_overrides`

مثال issue:

```json
{
  "issue_id": "gk_false_positive_14_958_1060",
  "issue_type": "gk_false_positive",
  "severity": "critical",
  "track_id": 14,
  "raw_track_id": 36,
  "first_source_frame_idx": 958,
  "last_source_frame_idx": 1060,
  "visible_label": "GK",
  "visible_color": "pink",
  "dominant_track_role": "player",
  "reason": "Goalkeeper display applied to player-dominant tracklet segment."
}
```

### 6. `player_crop_index.json`

فهرس أفضل الصور لكل tracklet/segment.

النموذج النصي لا يحتاج الصور، لكنه يحدد أي الحالات تحتاج Vision. بعدها نستخدم هذا الملف لاختيار crops.

مثال:

```json
{
  "track_id": 37,
  "best_crops": [
    {
      "path": "artifacts/crops/track_37/frame_1460.jpg",
      "source_frame_idx": 1460,
      "bbox": [568.8, 631.2, 611.8, 720.5],
      "quality_score": 0.82,
      "occlusion_score": 0.12,
      "role": "goalkeeper"
    }
  ]
}
```

### 7. `correction_candidates.json`

ملف وسيط ينتجه النظام قبل LLM.

الغرض: النموذج لا يبحث في كل شيء من الصفر. النظام يعطيه حالات مركزة:

- likely merge.
- likely split.
- likely false role.
- likely false team.
- needs vision.
- do not touch.

### 8. `correction_plan.json`

المخرج الوحيد المسموح من النموذج.

النموذج يخرج JSON صارم فقط، وليس نص حر.

## أنواع التصحيح المسموحة

### 1. `merge_tracklets`

دمج tracklet قديم وجديد كهوية واحدة.

مسموح فقط إذا:

- لا يوجد overlap زمني.
- ReID قوي أو vision أكد.
- الفريق متوافق.
- الدور متوافق أو تفسير اختلاف الدور محدود.
- gap منطقي.

### 2. `split_tracklet_segment`

تقسيم tracklet لأن داخله هويتين أو دورين متداخلين.

هذا أهم من merge في مشكلة الحارس:

لو tracklet رقم 14 لاعب أغلب الوقت، وظهر جزء قصير كحارس، لا نحول tracklet كله إلى GK. إما نصحح segment فقط أو نرفض وسم GK.

### 3. `display_override`

تعديل ما يظهر في الفيديو بدون تغيير raw tracking.

مثال:

- عرض `GK` على segment مؤكد فقط.
- إعادة segment مشكوك فيه إلى رقم اللاعب.
- منع اللون الوردي عن لاعب غير حارس.

### 4. `role_override`

تصحيح role في نطاق frame محدد.

لا يطبق على كامل tracklet إلا إذا كانت الثقة عالية جدا.

### 5. `team_override`

تصحيح الفريق في نطاق frame محدد.

مهم عند تبدل لون بسبب إضاءة أو قص crop سيء.

### 6. `reject_candidate`

رفض merge أو override مقترح لأنه غير آمن.

### 7. `needs_vision`

طلب رؤية crops/contact sheet قبل القرار.

## Schema لخطة التصحيح

```json
{
  "plan_version": "1.0",
  "match_id": "optional",
  "model": "gemma-4-26b-moe",
  "summary": {
    "safe_fix_count": 0,
    "needs_vision_count": 0,
    "do_not_touch_count": 0
  },
  "actions": [
    {
      "action_id": "fix_gk_false_positive_14_958_1060",
      "action_type": "display_override",
      "confidence": 0.91,
      "track_id": 14,
      "raw_track_id": 36,
      "first_source_frame_idx": 958,
      "last_source_frame_idx": 1060,
      "set_display_role": "player",
      "set_display_label": "14",
      "set_display_color_policy": "team",
      "evidence_ids": [
        "gk_false_positive_14_958_1060",
        "role_flicker_14_958_1060"
      ],
      "reason": "Tracklet is player-dominant; GK label appears on a short unstable segment and would create false goalkeeper display."
    }
  ],
  "needs_vision": [
    {
      "case_id": "long_gap_18_37",
      "question": "same_player_after_gap",
      "track_ids": [18, 37],
      "crop_refs": [
        "artifacts/crops/track_18/best.jpg",
        "artifacts/crops/track_37/best.jpg"
      ],
      "reason": "Low ReID distance but role/team disagreement and long gap."
    }
  ],
  "do_not_touch": [
    {
      "case_id": "unsafe_merge_37_14",
      "reason": "Role/team conflict and low evidence count."
    }
  ]
}
```

## Validator الحتمي

النموذج لا يطبق أي شيء مباشرة. كل action تمر على validator.

### قواعد عامة

1. ممنوع merge إذا يوجد overlap زمني.
2. ممنوع merge إذا نفس الهوية ستظهر مرتين في نفس frame.
3. ممنوع تحويل tracklet كامل إلى GK بسبب segment قصير.
4. ممنوع نشر `GK` على أكثر من لاعب لنفس الفريق في نفس frame.
5. ممنوع اعتماد `team=0` كفريق لاعب إلا إذا role هو referee أو unknown مؤقت.
6. أي تصحيح يحتاج `first_source_frame_idx` و`last_source_frame_idx`.
7. أي action تحتاج evidence IDs موجودة فعلا في artifacts.
8. إذا action تضر `render_audit` بعد التطبيق، يتم rollback.

### قواعد الحارس

1. `GK` label لا يطبق إلا على segment مؤكد.
2. الفريق يجب أن يكون معروفا أو مستنتجا بثقة، وليس `team=0` فقط.
3. لون الحارس لا يكفي وحده.
4. إذا لاعب دفاع ظهر باللون الأسود، لا يصبح حارسا بدون:
   - role evidence مستقر.
   - position evidence منطقي.
   - عدم وجود حارس آخر في نفس الفريق ونفس frame.
   - ReID/segment evidence داعم.
5. الحارس قد يظهر بأكثر من raw_track_id، لكن لا يجوز أن ينتشر وسم الحارس على لاعبين من نفس الفريق.

### قواعد long gap

لحالة لاعب يختفي 40 ثانية ثم يعود:

1. لا نعتمد gap وحده.
2. نحتاج ReID top-k.
3. نحتاج توافق الفريق.
4. نحتاج عدم overlap.
5. نحتاج position/movement plausibility.
6. إذا نفس الفريق ولاعبان متشابهان والـReID رمادي، تذهب الحالة إلى Vision.

## Post-fix audit

بعد تطبيق أي خطة على نسخة مؤقتة:

1. يعاد إنتاج `render_audit_after.json`.
2. تقارن النتائج مع `render_audit_before.json`.
3. إذا زادت أخطاء critical، يتم rollback.
4. إذا تحسنت أخطاء الهوية بدون إدخال أخطاء حارس/فريق، تعتمد.

معايير النجاح الأساسية:

- `gk_false_positive_segments = 0` أو أقل من قبل.
- `unsafe_gk_overrides = 0`.
- انخفاض `goalkeeper_display_spread`.
- انخفاض `role_flicker_tracklets`.
- عدم زيادة `same_identity_overlap`.
- عدم زيادة `team_flicker_tracklets`.
- عدم ظهور أكثر من `GK` لنفس الفريق في نفس frame.

## Prompt contract للنموذج

النموذج يأخذ:

- مختصر match metadata.
- `identity_debug.json` compact.
- `identity_events.json`.
- `reid_topk.json`.
- `role_logits_timeline.json`.
- `team_color_palette.json`.
- `render_audit.json`.
- `correction_candidates.json`.

النموذج ممنوع من:

- تخمين رقم قميص غير موجود.
- إنشاء track_id غير موجود.
- افتراض أن `team=0` يعني فريق الحارس.
- تصحيح كامل tracklet إذا المشكلة segment.
- اتخاذ قرار merge إذا evidence غير كاف.
- كتابة output خارج JSON schema.

النموذج يجب أن:

- يصنف كل حالة إلى `safe_fix`, `needs_vision`, `do_not_touch`.
- يذكر evidence IDs.
- يحدد frame range.
- يعطي confidence.
- يعطي reason مختصر.
- يفضل `needs_vision` على التخمين.

## Vision escalation

Vision يعمل بعد text review، وليس قبله.

### مدخلات Vision

- crop قبل الانقطاع.
- crop بعد الانقطاع.
- contact sheet صغير.
- JSON مختصر للحالة.
- سؤال محدد.

### أمثلة أسئلة

- هل هذان crop لنفس اللاعب؟
- هل هذا الحارس أم لاعب دفاع؟
- هل رقم القميص ظاهر؟
- هل اللون الوردي/الأسود يعود لحارس أم لاعب؟

### مخرج Vision

```json
{
  "case_id": "long_gap_18_37",
  "verdict": "same_player",
  "confidence": 0.84,
  "visual_evidence": [
    "similar jersey color",
    "similar body shape",
    "no visible jersey number"
  ],
  "limits": [
    "face not visible",
    "motion blur"
  ]
}
```

## مراحل التنفيذ المقترحة

### Phase 0: تثبيت baseline

- لا نعتمد `sha-5f969e6`.
- نختبر على image أقدم أقل سوءا أو build جديد من branch قبل التغيير السيئ.
- نوثق حالة فشل الحارس الوردي كاختبار إلزامي.

### Phase 1: Audit فقط

لا يوجد تصحيح.

نولد:

- `identity_events.json`
- `render_audit.json`
- `reid_topk.json`

الهدف: كشف نفس المشكلة بدون تغيير الفيديو.

### Phase 2: Dry-run correction

النموذج ينتج `correction_plan.json`.

لا يطبق شيء.

الهدف: رؤية هل الخطة منطقية وهل تصنف الحالات الخطرة كـ `needs_vision` أو `do_not_touch`.

### Phase 3: Safe corrections only

تطبيق actions منخفضة المخاطر:

- إزالة GK false positives.
- منع display label خاطئ.
- split segment واضح.
- reject unsafe links.

لا نطبق merges طويلة الغياب إلا إذا الأدلة قوية جدا.

### Phase 4: Vision review queue

للحالات الرمادية فقط.

نولد `vision_review_queue.json`.

لا نشغل Vision في هذه المرحلة ولا نعدل الهوية.

### Phase 5: Crop/contact-sheet evidence

نجهز الأدلة البصرية للحالات الموجودة في queue:

- `player_crop_index.json`
- `identity/vision_crops/`
- `identity/contact_sheets/`
- `vision_contact_sheets.zip`

الهدف أن أي نموذج Vision لاحق يرى crops محددة وسؤال محدد بدلا من المباراة كاملة.

### Phase 6: Vision review results

نولد `vision_review_results.json`.

إذا لم يكن provider مفعل بوضوح:

- `vision_model_invoked=false`
- كل الحالات تبقى `unresolved`
- `confidence=0.0`
- `recommended_action=keep_unresolved_no_identity_mutation`

لا يسمح للنظام بإعطاء حكم بصري من التخمين أو من ملفات JSON فقط.

### Phase 7: Final correction/render integration

بعد نجاح audit:

- apply correction plan.
- render final video.
- export reports.
- upload outputs.

نولد أيضا `final_render_identity_manifest.json`.

هذا الملف لا يستخدم كمنع للرندر، بل كتصنيف نهائي:

- `identity_trusted`: لا توجد مشاكل عالية الخطورة ولا حالات Vision غير محلولة.
- `review_required`: الفيديو خرج، لكن الهوية تحتاج مراجعة بسبب audit أو Vision queue.
- `invalid_render`: الرندر نفسه غير صالح أو artifact validation فشلت.

إذا بقيت حالات غير محلولة، ينتج النظام فيديو review مع artifacts ولا يدعي أن الهوية النهائية موثوقة.

### Phase 8: UI/API surfacing

تعرض الواجهة حالة الهوية النهائية بوضوح:

- `identity_trusted`: تظهر كحالة نجاح.
- `review_required`: تظهر كتحذير، مع توضيح أن الفيديو ليس نهائيا من ناحية الهوية.
- `invalid_render`: تظهر كخطأ.

وتعرض روابط ملفات المراجعة:

- `final_render_identity_manifest.json`
- `vision_review_results.json`
- `player_crop_index.json`
- `vision_contact_sheets.zip`
- audit/correction artifacts

هذه المرحلة لا تضيف تصحيحا جديدا ولا تشغل Vision. وظيفتها منع سوء فهم النتائج في الواجهة.

### Phase 9: RunPod packaging preflight

قبل بناء image جديدة أو رفعها:

```bash
python scripts/check_runpod_phase9_preflight.py
```

الفحص يتأكد من:

- `runpod/Dockerfile` ينسخ `src/` و `runpod/handler.py`.
- `.dockerignore` لا يستبعد `src/` أو handler.
- `runpod/handler.py` يرفع كل artifacts الجديدة.
- `batch_analyzer.py` يصدّر كل artifacts الجديدة.
- GitHub workflow يعيد البناء عند تغييرات `src/**` وRunPod worker.
- baseline يبقى `sha-bbe8dec` إلى أن يتم اعتماد image جديدة.
- image السيئة `sha-5f969e6` تبقى مسجلة كـ regression.

ملاحظة: `.dockerignore` يستبعد `models`، لذلك يجب أن يزوّد RunPod `model_path` أو تُبنى weights بشكل واضح داخل image. هذا تحذير وليس فشل preflight.

### Phase 10: Release candidate gate

قبل اعتماد أي image جديدة:

```bash
python scripts/create_runpod_phase10_release_manifest.py \
  --candidate-image ghcr.io/mo7ammad09/football-tactical-ai-runpod:sha-NEW \
  --tests-passed
```

الـ manifest يعطي:

- `candidate_ready`: فقط إذا preflight نجح، والاختبارات مسجلة كناجحة، والشجرة نظيفة، والـ image ليست معروفة كسوء.
- `blocked`: إذا candidate image غير موجودة، أو الشجرة فيها تغييرات غير ملتزمة، أو image هي `sha-5f969e6`، أو الاختبارات غير مسجلة.

حتى بعد `candidate_ready` لا نعتمد الصورة نهائيا قبل RunPod smoke job قصير يؤكد:

- وجود `final_render_identity_manifest_json_url`.
- وجود `vision_review_results_json_url`.
- عدم انتقال `GK`/اللون الوردي للاعبي الدفاع.
- `release_status` ليس `invalid_render`.

### Local completion report

لإغلاق المشروع محليا قبل build:

```bash
python scripts/create_identity_project_completion_report.py
```

سيكون طبيعيا أن يعطي:

- `implementation_complete_deploy_blocked` إذا لم توجد candidate image أو كانت الشجرة dirty.
- `ready_for_candidate_build` فقط بعد commit نظيف + candidate image + تسجيل الاختبارات.

## اختبارات إلزامية قبل الاعتماد

### Test 1: image السيئة

Input behavior:

- `sha-5f969e6` نشر `GK` على لاعبين ومدافعين.

Expected:

- `render_audit.json` يكتشف `gk_false_positive_segments`.
- correction plan يزيل GK عن غير الحارس.
- post-fix audit لا يزيد عدد GK الخاطئين.

### Test 2: tracklet 14

Input:

- tracklet 14 player-dominant.
- segments قصيرة ظهرت goalkeeper.

Expected:

- ممنوع تحويل tracklet 14 كاملا إلى GK.
- التصحيح يكون segment-level أو reject.

### Test 3: goalkeeper fragmentation

Input:

- goalkeeper يظهر تحت عدة display IDs.

Expected:

- النظام يحاول توحيد الهوية فقط إذا لا يوجد overlap وReID/vision داعم.
- إذا غير مؤكد، يستخدم `needs_vision`.

### Test 4: player long gap

Input:

- لاعب اختفى 40 ثانية ورجع.

Expected:

- `reid_topk.json` يقدم المرشحين.
- text model يصنف الحالة.
- إذا ReID رمادي أو لاعبان متشابهان، يطلب Vision.

### Test 5: no regression

Expected:

- لا تزيد أخطاء الفريق.
- لا تزيد أخطاء role flicker.
- لا تظهر هوية واحدة في مكانين بنفس frame.

## مخرجات RunPod job

بعد اكتمال job يجب أن ينتج:

```text
artifacts/
  raw_tracklets.jsonl
  identity_debug.json
  identity_events.json
  reid_topk.json
  role_logits_timeline.json
  team_color_palette.json
  render_audit_before.json
  correction_candidates.json
  correction_plan.json
  render_audit_after.json
  correction_applied.json
  vision_review_queue.json
  player_crop_index.json
  vision_review_results.json
  final_render_identity_manifest.json
  final_annotated_video.mp4
  review_contact_sheets/
  crops/
```

## قرار الرندر

النظام لا يجب أن يكون "يمنع الرندر" كهدف.

القرار الصحيح:

1. إذا التصحيح آمن: طبق وصدر الفيديو النهائي.
2. إذا التصحيح يحتاج Vision: شغل Vision ثم أعد التقييم.
3. إذا لا يزال غير موثوق: صدر فيديو review واضح أو استخدم fallback لا يدعي ثقة زائفة.
4. لا تصدر فيديو نهائي يدعي أن لاعب دفاع هو `GK`.

## قواعد تمنع الهلوسة

1. كل action يجب أن تشير إلى evidence IDs.
2. كل evidence ID يجب أن يكون موجودا في ملفات artifacts.
3. كل action يجب أن تكون محدودة بإطارات.
4. النموذج لا يرى embeddings خام؛ يرى distances محسوبة.
5. النموذج لا يقرر من لون واحد.
6. النموذج لا يخترع لاعب أو رقم قميص.
7. الـvalidator هو صاحب القرار التنفيذي.
8. post-fix audit هو شرط اعتماد التصحيح.
9. أي action تخالف القيود تتحول إلى `rejected_by_validator`.
10. أي خطة تزيد أخطاء critical يتم rollback لها.

## الخلاصة المعتمدة

نظام التصحيح قبل الرندر يجب أن يكون:

- RunPod-first.
- Text-first.
- Vision-on-demand.
- Evidence-driven.
- Segment-aware.
- Validator-controlled.
- Audit-before-and-after.

ولا يجوز أن يبدأ التنفيذ من سلوك image:

`ghcr.io/mo7ammad09/football-tactical-ai-runpod:sha-5f969e6`

هذه image تسجل كحالة فشل واختبار regression، وليست baseline صالحا.
