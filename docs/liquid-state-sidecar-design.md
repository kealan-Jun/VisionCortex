# Liquid State Sidecar Design

## Decision

Do not place a new liquid model in the current production decision path yet.
The current laboratory footage primarily contains colored or otherwise
non-transparent liquids, so a transparent-contact-line detector is not the
primary runtime architecture.

Freeze the interface now, then add the real model as a feature-flagged shadow
sidecar after the DEV-035 multiview quality candidate passes. Shadow output is
written as observations and uncertainty but cannot change experiment
boundaries, event acceptance, key-material selection, daily-report claims or
archive promotion.

## Fit to the actual scene

The preferred task is container-conditioned liquid-state perception:

1. use the existing 21-class YOLO models to locate containers, tools and hands;
2. crop only the container regions involved in accepted/candidate actions;
3. segment colored liquid inside the container ROI;
4. estimate fill ratio/level band and visible flow or transfer direction;
5. compare short aligned time windows to infer state before, state after and
   change confidence;
6. fuse first-person and third-person observations without requiring both
   views to see the liquid equally well;
7. return `unobservable` or `uncertain` when glare, occlusion, opacity or camera
   geometry prevents a defensible claim.

Transparent liquid support remains a fallback evaluation slice, not the design
center.

## Model roles

- A small supervised segmentation model trained on our own container ROIs is
  the preferred production component for colored liquids. A YOLO segmentation
  head or lightweight U-Net/SegFormer-class head is sufficient; it need not be
  an open-vocabulary foundation model.
- SAM 2 is useful for propagating an initialized ROI mask through a short video
  clip, but it is not itself a liquid-state classifier.
- DINOv3 ConvNeXt Tiny can be used as a frozen visual feature backbone for a
  small temporal state head, but it is not itself a liquid detector.
- TCLD is a useful transparent-liquid baseline/teacher and dataset reference,
  but its published domain is narrow and should not define our production
  architecture.
- LabLiquidVision is a useful academic reference for liquid segmentation,
  depth and volume estimation in transparent vessels, not a drop-in universal
  runtime model.
- Grounding DINO is unnecessary in the normal known-scene path because the
  existing task-specific 21-class YOLO already supplies faster object and
  container localization. It may be retained as an offline annotation helper.

Primary references:

- DINOv3: <https://ai.meta.com/research/dinov3/> and
  <https://github.com/facebookresearch/dinov3>
- SAM 2: <https://ai.meta.com/research/sam2/>
- TCLD: <https://github.com/dualtransparency/TCLD>
- LabLiquidVision: <https://github.com/DaniSchober/LabLiquidVision>
- Grounding DINO 1.5 API reference:
  <https://github.com/IDEA-Research/Grounding-DINO-1.5-API>

## Output contract

Each observation is additive to the existing key-material event JSON:

```json
{
  "liquid_observation": {
    "schema_version": "visioncortex-liquid-state/1.0.0",
    "status": "observed|uncertain|unobservable|not_applicable",
    "container_object_id": "tube-21",
    "appearance": "colored|opaque|transparent|unknown",
    "liquid_present": true,
    "fill_ratio": 0.42,
    "fill_ratio_uncertainty": 0.06,
    "level_band": "mid",
    "flow_direction": "source_to_target|target_to_source|none|unknown",
    "state_change": "filled|emptied|increased|decreased|mixed|unchanged|unknown",
    "first_person_support": 0.83,
    "third_person_support": 0.71,
    "cross_view_status": "supported|single_view_supported|contradicted|unobservable",
    "evidence_frame_ids": [],
    "evidence_clip_ids": [],
    "model_name": "",
    "model_version": "",
    "confidence": 0.0,
    "uncertainty_reasons": []
  }
}
```

The existing event ID, aligned timestamps, object IDs, key-frame/key-clip paths
and provenance remain canonical. The liquid result references them rather than
creating a parallel timeline.

## RTX 4060 execution design

The 8 GB RTX 4060 should run the liquid sidecar phase-separated from YOLO:

- never scan all six three-hour streams with the liquid model;
- reuse already decoded/cached frames from accepted container-action windows;
- process only short ROI windows, initially 3-8 seconds at 2-5 FPS;
- batch same-size container crops across views and events;
- release or suspend YOLO/TensorRT workspaces before loading the liquid model;
- use FP16 TensorRT/ONNX where the selected model supports it;
- cap peak VRAM below 7.2 GiB and preserve Windows/NVDEC/NVENC headroom;
- run one sidecar inference context, not one model copy per camera;
- write results incrementally to staging and NAS archive only after the owning
  stage passes its schema/quality checks.

Initial performance budget for the complete six-view experiment is no more
than 120-180 additional wall-clock seconds. A sidecar exceeding that budget must
be optimized or narrowed before production promotion.

## Dataset and acceptance

Build the training/evaluation set from our actual accepted container-action
windows, stratified by camera, container type, liquid color/opacity, fill level,
pouring/pipetting/mixing action, lighting and occlusion. Foundation models may
assist annotation, but reviewed masks and state transitions remain the ground
truth for model development.

Minimum evaluation dimensions:

- liquid-present precision/recall;
- mask IoU for visible colored liquid;
- fill-ratio absolute error and level-band accuracy;
- transfer/state-change event precision/recall;
- false-confirmation rate on empty containers and occluded containers;
- first/third-view agreement and correctly declared single-view support;
- `unobservable` calibration instead of forced guesses;
- added wall time, peak VRAM and NAS read volume.

## Rollout

1. Pass and freeze DEV-035 without the liquid sidecar.
2. Add only the provider interface, schema, feature flag defaulting to `off`,
   resource accounting and unit tests.
3. Train/evaluate a colored-liquid ROI segmentation/state model on actual data.
4. Run it on the RTX 4060 in shadow mode; it writes JSON observations but cannot
   change any production decision.
5. Promote it first as an evidence confirmer for liquid movement/container
   state change after accuracy, calibration and performance gates pass.
6. Consider boundary/key-selection influence only after a separate real-video
   acceptance proves that it improves quality without causing false segments.
