# RTX 4050 performance and staged delivery review, 2026-09-10

This is development evidence, not a stable-release or real-video quality receipt.
Base commit: `24070c4d45dbeae6b596df4dcf6f293a7c81e33a`.
Changes go to development `codex/rtx4050-optimization-20260908`.

## Supplied execution evidence

- `s05-performance-20260910-105101.zip`: two approximately 27.26-second inputs;
  2,045.80 seconds total. Group understanding took 386.66 seconds, key-material
  understanding 903.72 seconds, and material refinement 520.06 seconds.
  Fine YOLO inference itself took 2.606 seconds. Eleven logical calls, twelve
  attempts: 131,161 input and 92,809 output tokens. No recorded HTTP 429.
- Initial semantic workers could wait at a batch boundary despite one completed
  request. Auxiliary models were repeatedly loaded and some ran on CPU.
- GPU utilization averaged about 31.7%; VRAM peaked at 89.6% during coarse scan,
  not throughout the run. These observations do not prove sustained GPU saturation.
- The final s05 exception was a NoneType attribute error; the archive has no full
  traceback or original media. Its exact failing code path is NOT_PROVEN.
- `s06-view-failure-20260910-105705.zip`: a 74.66-second retry requested a routed
  video interval 0–30,471.013 ms while the third-person source covers approximately
  338.812–27,528.7 ms. The failure was in route coverage validation. The derived
  contract now represents missing head/tail footage explicitly and retains
  completed role exports on failure; it does not invent frames or action evidence.

## Implemented changes and evidence boundaries

- Refill semantic workers after a successful live response; cached responses do
  not establish provider health. Keep the initial failure-wave bound.
- On the 4050 profile, use CUDA for DINO/LabPics with the same FP32 inference;
  only an actual CUDA OOM retries on CPU. Park auxiliary weights in host RAM
  between stages when at least 8 GiB is available, otherwise release them.
- SAM2 ordinal decoding retrieves only sampled images. This preserves requested
  frame selection without treating a mask as physical-action confirmation.
- Preserve originals and completed transcription chunks independently of ASR,
  final quality acceptance and formal publication. Retry NAS delivery after
  generating partial reports, including failure and quality-attention exits.
  Discover and preserve each physical recording independently: an unavailable
  source or failed copy keeps its failure identity while the other originals
  continue to be saved. The stage retains a failed status instead of claiming
  all recordings were archived. Busy-source and copy-failure cases are covered
  by deterministic tests; this is not a real-ASR accuracy receipt.
- Export visual and speech references in a portable JSON graph. Speaker identity
  remains unknown; aligned time and temporal association do not prove an action.
- Keep product reports focused on operations and evidence limitations. Technical
  receipts and token/resource metrics remain in structured records.

Deterministic tests cover scheduling, cache residency, OOM fallback, frame
selection, route gaps, optional speech failures, original-audio archival,
reference joins, report presentation, status updates and stage recovery.
These are implemented/test evidence. Windows 4050 speed, real speech accuracy,
actual NAS recovery on the customer machine and full-video quality remain
PARTIAL_EVIDENCE or NOT_PROVEN until the corresponding real execution receipts.

## Local 3090 Ti capacity experiment

Existing batch-64 engines and sixteen predecoded real frames per role were reused
for 15-second inference capacity trials. This is neither ten distinct sources nor
end-to-end preprocessing. Prediction receipts match warmup within each trial;
there is no cross-batch human accuracy baseline.

| Contexts / batch | Throughput (frames/s) | Test-process VRAM after run (MiB) |
| --- | ---: | ---: |
| 4 / 16 | 1034.223 | 6590 |
| 8 / 32 | 961.850 | 13160 |
| 10 / 64 | 957.1 | 17456 |

The idle application separately held about 4020 MiB. Increasing memory occupancy
did not increase throughput in these trials. This does not establish the hardware's
absolute maximum or predict 4050 throughput. Production engine bindings remain
unchanged by these trials.

## Outstanding real-video quality gate

Run `upload-dfb0808eefe2` took 652.63 seconds and retained only a 5.55-second group
and one key event from approximately 163-second inputs. Its model boundary review
covered only the final small candidate window. It also recorded a no-audio semantic
validation failure. Clear action-frame selection, full-timeline recall, object
identity and dual-view correspondence are not accepted as correct. UI/report and
archive fixes do not constitute a successful rerun of this experiment.
