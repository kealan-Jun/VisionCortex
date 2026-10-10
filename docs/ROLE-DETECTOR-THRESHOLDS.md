# Detector confidence by role and phase

`models.confidence` remains the default for all detector roles. Optional
`models.confidence_by_role` calibrates fine detection by `first_person` or
`third_person`. Optional `models.coarse_confidence_by_role` overrides coarse
detection independently; an absent coarse override uses the global default,
not the fine override. Keys describe roles, never camera identities.

Thresholds must be finite numbers in `(0, 1]`. The scanner uses the selected
threshold for prediction and reports it as `prediction_confidence` in scan
runtime evidence. Cache identity includes the selected role/phase threshold;
changing another role or phase does not invalidate a compatible detector cache.

Example: a calibrated first-person fine model can use `0.225` while the retained
coarse model and third-person model continue using a global default of `0.25`:

```yaml
models:
  confidence: 0.25
  confidence_by_role:
    first_person: 0.225
```

A new threshold is a model acceptance decision. Record the development selection,
actual engine, runtime configuration and rollback with each deployment; this
configuration feature does not automatically promote a model.
