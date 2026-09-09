# RTX 4050 alignment and failure-message repair

**Goal:** Preserve recorded RGB clock coverage and report alignment failures accurately.

**Design:** Count eligible recorded RGB rows before choosing an evenly distributed,
bounded set of samples including both endpoints. Retain source CSV row numbers,
clock-validity checks, and existing quality thresholds. Classify explicit alignment
gate failures separately from memory exhaustion; mere mentions of CUDA/GPU/memory
must not assert insufficient resources.

**Scope:** `src/visioncortex/alignment.py`, `src/visioncortex/web/app.js`,
`tests/test_alignment_enhancements.py`, `tests/test_web_library_reliability.py`.
Use existing Python 3.12 dependencies and Node; no new application dependencies.

## Execution and verification

- [x] Run existing alignment and web characterization suites as baseline.
- [x] Add mixed RGB/depth sampling, endpoint/budget, dual-view gate, and error
  classification regression tests. Confirm failures against the original code.
- [x] In `read_timestamp_csv`, count eligible rows with the same CSV parser and
  RGB predicate used for selection. For `n` eligible rows and `k=min(n,max_points)`,
  select indices `i*(n-1)//(k-1)` for `i` in `range(k)`. Keep memory bounded and
  retain the existing error when fewer than two recorded rows are available.
- [x] Recognize alignment gate errors and explicit memory exhaustion in
  `friendlyFailureReason`; give actual OOM precedence over generic SAM2/frame text.
- [x] Run relevant suites, `node --check src/visioncortex/web/app.js`,
  `ruff check src tests`, and `python -m compileall -q src tests`.
- [x] Validate current real CSV sampling and alignment in a separate output
  location, without changing historical receipts or relaxing quality gates.
- [x] Review diff and evidence. Transfer only the two tested source changes into
  the installed-version repair checkout, preserving unrelated version differences.
- [ ] Commit the repair snapshot locally and apply through the existing verified,
  backup-producing updater after the application is idle and closed.
- [ ] Verify installed source identity and user-visible failure classification.

## Evidence boundaries

The development-authority remote is unavailable. Work remains a local repair,
with no stable promotion or claim of accepted development CI. Deterministic
tests prove contracts; replayed or fresh alignment proves that stage only.
Complete real-video analysis and stable-release readiness require their own
receipts. Do not include input media, runtime output, model binaries or secrets
in Git. Keep 0.65 alignment confidence and 1000 ms role overlap requirements.

## Verified results before installation

- Six added regression cases failed on the baseline and passed with the fix.
- Relevant deterministic suites: 251 passed; isolated reviewer subset: 9 passed.
- Installed-version repair checkout: 17 focused regression/update tests passed.
- Ruff, Python compilation, JavaScript syntax and diff whitespace checks passed.
- Fresh alignment using both existing real input pairs passed the unchanged
  alignment gate. One pair retains a partial-timeline coverage warning. Input
  file sizes and SHA-256 values were rechecked against upload receipts.
- No full model pipeline or stable promotion is claimed.
