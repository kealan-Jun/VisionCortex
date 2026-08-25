# Key-event ground-truth contract

VisionCortex only reports key-material Precision/Recall when a reviewed label file explicitly applies to the current experiment_id. The labels remain evaluation-only and never enter candidate generation, grouping, boundary selection, or model prompts.

Start from configs/evaluation/key-event-ground-truth.example.json.

Required rules:

- Every reviewed event has a unique event_id, a supported action type, and end > start.
- Time uses the aligned global experiment timeline.
- labeled_windows states exactly which part of the source was reviewed. Predictions outside those windows are excluded rather than counted as false positives.
- Confirmed, uncertain and rejected labels are separated. Only confirmed labels enter Precision/Recall.
- An event may use the production key-material shape (`start_us`, `end_us`, `peak_timestamp_us`, role-to-object `objects`, and `decision.status`) or the compact review shape with an object string array.
- When an event declares objects, a prediction must share at least one canonical non-hand object as well as the action class and temporal overlap.
- Action aliases such as liquid_transfer/liquid_movement and panel_operation/device_panel_operation are canonicalized.
- Object aliases such as paper/weighing_paper and spearhead/pipette_tip are canonicalized.
- The receipt reports tIoU 0.3, 0.5 and 0.7, per-class metrics, annotation coverage, excluded IDs, one-to-one matches and Wilson 95% confidence intervals.

Do not convert a sampled “category appears somewhere in the clip” review into overall accuracy. For the colleague report discussed on 2026-08-17, raw event labels and matching receipts are still required before its numbers can be imported as production ground truth.
