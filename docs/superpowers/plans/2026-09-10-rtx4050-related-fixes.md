# RTX 4050 related fixes integration plan

**Goal:** Integrate the reviewed fixes at `83d72d17904359518ae56cad86fe6035f7fffe72` into local `4c725590f0af23b5d47bd5a3673055a77a682e9b`, preserving the local engine, timestamp and cache repairs, then update the installed desktop using the existing verified source updater.

**Architecture:** Merge the shared history in an isolated local branch. Keep stage recovery, explicit source coverage gaps, bounded semantic scheduling, auxiliary model residency and recording/report delivery changes together with their tests. Preserve local Windows and engine identity behavior. Do not tune models, relax evidence gates, publish to GitHub or perform a paid real-video run.

**Tech stack:** Existing Python 3.12 offline runtime, existing external pytest support, Node.js, Git and the repository source updater. No dependency installation or model/engine rebuild is planned.

## Tasks

- [x] Verify the clean local checkout, installed SHA, target SHA, remote identities and repository policy. The development remote is not accessible with the current account (HTTP 404); keep this integration local and do not treat it as stable promotion.
- [x] Run baseline tests for `test_alignment_enhancements.py`, `test_temporal_segmentation.py`, `test_rtx4050_engine_reuse.py` and `test_cache_paths.py` against the local base.
- [x] Merge the immutable target, resolving any overlapping tests while retaining `tools/rtx4050_portable.py`, recorded-RGB sampling in `alignment.py`, SAM2 `work_namespace`/`settings` identity, and the alignment failure message in `web/app.js`.
- [x] Run merged tests for source coverage, stage recovery, scheduling, auxiliary model residency, recordings, reports, Web behavior, local/no-NAS paths and the updater. If integration failures appear, reproduce them and make the smallest coherent correction before rerunning the affected tests.
- [x] Run Ruff, Python compilation, Web syntax, desktop Node tests, diff and repository policy checks. Review the actual merged diff and all preserved local contracts.
- [ ] Commit only named source/test/documentation paths with the configured user identity. Retain both parent SHAs and the verification results.
- [ ] Check for active installed application processes/tasks. When idle, invoke the existing source updater with the clean committed checkout; retain its backup and verify the installed source receipt and changed-file hashes. Do not terminate an active analysis to install an update.
- [ ] Report branch/SHA, test outcomes, backup path and installation status. Real-video quality, Windows GPU performance and stable-release readiness remain `NOT_PROVEN` until their own execution receipts exist.

## Verified integration evidence

The automatic merge preserves both histories and all local repairs without textual conflicts. Before merging, 70 local engine/timestamp/SAM2/cache tests passed; the upstream s06 coverage test failed on the old source because the coverage helper was absent. After merging, 32 focused modules returned **667 passed, 1 skipped in 196.43 seconds**. The skip requires an optional external validation archive. Tests used the existing bundled Python 3.12.10, external pytest support, a short E: temporary root and the supported Starlette httpx fallback; only the existing Starlette deprecation category was filtered in the local invocation. No application dependency was changed.

Ruff, Python compilation, Web syntax, 33 desktop Node tests, Git diff checks and the tracked repository/credential policy check passed (481 tracked files before this plan). Independent static integration review found no concrete blocking regressions and confirmed preservation of engine reuse, RGB sampling, SAM2 namespace/settings identity and Web failure classification. The installed engine cache mapping exists; model binaries have not been rebuilt.

Commit and installation completion are recorded by the updater's SHA-bound receipt and the local diagnostic handoff, rather than predicted in this pre-installation document. No real-video quality, actual GPU performance, NAS recovery or stable-release claim is made by these deterministic checks.
