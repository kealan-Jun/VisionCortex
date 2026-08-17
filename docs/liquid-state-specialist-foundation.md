# Liquid-State Specialist Foundation

## Outcome

VisionCortex now has a structured, opt-in contract for liquid visibility, fill
level, meniscus geometry, visible flow, flow direction, and source/target state
change. The feature is disabled by default and is not connected to motion,
coarse YOLO, Fine YOLO, boundary grouping, or the RTX 4060 production runtime.
It therefore adds no decode, inference, API, or Token cost to the validated
preprocessing path.

## Why this is a specialist layer

The existing 21-class YOLO models locate hands, tools, vessels, panels, and
other lab objects. They do not directly observe a transparent liquid body,
meniscus, fill ratio, or fine surface change. Those claims need evidence from a
bounded container ROI and must remain separate from the object detector's
action-candidate logic.

The intended data flow is:

1. Existing YOLO and tracking identify a vessel and a bounded experiment event.
2. The pipeline samples only the referenced key frame/burst inside that vessel
   ROI.
3. A registered liquid-state adapter returns per-view observations.
4. Deterministic fusion records observed facts, supported inferences, uncertain
   claims, and a model receipt.
5. The result is embedded under `provenance.liquid_state` in the existing key
   material JSON contract and becomes searchable in the rebuildable SQLite
   index.

Audio is deliberately outside this foundation.

## Public-model integration directions

The adapter boundary is intentionally model-agnostic:

- A liquid-container segmentation model is the preferred primary specialist
  when transparent-liquid masks and fill levels are required.
- SAM 2 can be used as a promptable mask tracker after YOLO supplies a vessel
  box; it is a segmentation/tracking component, not a liquid-state classifier.
- DINOv2-style visual features can support a small lab-specific state head or
  retrieval baseline; they do not directly output a liquid mask or meniscus.
- A compact TensorRT segmentation/state head is the production target for the
  8 GiB RTX 4060 after accuracy and latency are proven on the golden set.

No one of these backends is imported by the core package. A deployment must
register an adapter explicitly, which prevents an unavailable or partially
installed model from silently weakening production evidence.

## Structured evidence contract

`LiquidStateEvidence` contains:

- evidence status: `not_evaluated`, `observed`, `inferred`, `uncertain`, or
  `occluded`;
- backend and model receipt;
- source and target container state before/after;
- per-view presence, fill ratio, meniscus polyline, visible flow, direction,
  visibility, confidence, observed facts, and uncertainty;
- fused visible-flow, direction, state-change, confidence, facts, inferences,
  and uncertain claims.

All coordinates are normalized. All times use the aligned global timeline in
microseconds. Mask outputs are references, not embedded binary blobs.

## Golden benchmark

`build_liquid_benchmark_manifest` converts archived key-material JSON records
into a references-only manifest. Media remains in the archive; the manifest
does not create a second copy. Split assignment is performed by experiment
group to prevent adjacent frames from the same operation leaking across
training and evaluation.

The initial golden set should include:

- visible and partially occluded transparent liquids;
- empty/filled negative pairs;
- pipette aspiration and dispensing;
- container pours and stationary containers;
- glare, gloves, labels, colored liquid, colorless liquid, foam, bubbles, and
  opaque-container exclusions;
- matched first-person and third-person observations where each view is scored
  independently.

The deterministic evaluator reports presence precision/recall/F1, visible-flow
precision/recall/F1, direction and state-change accuracy, fill-ratio MAE,
normalized meniscus distance, coverage, and false confirmation of uncertain or
occluded cases. It consumes zero Tokens.

## Configuration and rollout gate

The default and RTX 4060 profiles contain:

```yaml
liquid_state:
  enabled: false
  backend: disabled
```

Enabling a non-disabled backend without registering its adapter fails closed
before any specialist inference. A future production integration should remain
blocked until all of the following are demonstrated on held-out experiments:

- the golden benchmark passes agreed accuracy thresholds;
- false confirmation on uncertain/occluded samples is within the safety limit;
- bounded ROI sampling stays inside its independent runtime budget;
- first/third-view evidence and uncertainty survive JSON and index round trips;
- the existing five-experiment boundary and key-material quality gates do not
  regress;
- the 20-minute preprocessing target is measured separately and remains
  unchanged while the feature is disabled.

## Files

- `src/labvision_evidence/schemas.py`: evidence and benchmark data models
- `src/labvision_evidence/liquid_state.py`: adapter registry, disabled backend,
  manifest builder, and deterministic evaluator
- `src/labvision_evidence/archive.py`: optional JSON embedding
- `src/labvision_evidence/indexing.py`: optional SQLite columns and filters
- `benchmarks/liquid-state/golden-manifest.template.json`: fixed manifest shell
- `tests/test_liquid_state.py`: contract, fail-closed, manifest, and evaluator
  coverage
