# DEV-041 Read-Only Three-Dataset Acceptance

This command turns the durable JSON ledgers from the six-view, A, and Z runs
into one quality, performance, Token, hardware, network, output, and promotion
matrix. It never opens source MP4, clock CSV, extracted image, key clip, PDF, or
model API payloads.

```powershell
$env:PYTHONPATH = (Resolve-Path 'src').Path
python -m labvision_evidence.cli accept-three-datasets `
  --archive-root 'Y:\VisionCortexExperimentArchive' `
  --spec 'configs\acceptance\dev041-three-dataset.json' `
  --output 'C:\VisionCortex-Acceptance\DEV-041-three-dataset-acceptance.json'
```

The command writes JSON plus a sibling Markdown report outside the NAS archive.
It exits `0` only when all three latest runs pass every applicable gate. It exits
`1` for an incomplete, failed, unpromoted, or quality/performance-regressed run.

Candidate discovery compares the formal archive with every staging run for the
same English archive name and selects the newest durable `pipeline_status`
timestamp. Therefore an old successful formal archive cannot hide a newer
failed staging run. When the final execution handoff contains exact run paths,
pin them explicitly:

```powershell
python -m labvision_evidence.cli accept-three-datasets `
  --archive-root 'Y:\VisionCortexExperimentArchive' `
  --candidate 'six_view=Y:\...\collection-six' `
  --candidate 'a=Y:\...\collection-a' `
  --candidate 'z=Y:\...\collection-z' `
  --output 'C:\VisionCortex-Acceptance\DEV-041-pinned.json'
```

The six-view dataset receives the reviewed five-group, G2/G4, 80-event, closed
formal-anchor, and 1,200-second gates. A and Z never inherit those priors; they
must produce their own natural bounded groups and event distribution. All three
still require aligned first-/third-person evidence, object-aware semantic
filenames, current/next-step understanding, searchable references, continuous
state receipts, branded reports, exact timing/Token ledgers, and verified atomic
promotion.
