# RTX 4050 engine reuse implementation plan

> For agentic workers: use superpowers:executing-plans to implement this approved change.

Goal: retain compatible TensorRT engines across ordinary source updates and skip the export subprocess on a verified cache hit.

Architecture: scope the cache identity to GPU/runtime identity, bundled Python file hashes, the engine builder CLI file, the two model weights and export/autotune parameters. Keep full package verification and actual engine/model hash checks. Adopt a legacy directory only when its old signature can be reproduced from the current configuration/hardware and a compatible, hash-verified update backup manifest. Record that mapping locally; never move engines or alter their build receipts. Missing compatibility evidence means a new cache identity; invalid engine receipts stop startup.

Tech stack: Python standard library, existing portable launcher, pytest; no new dependencies.

Constraints: local development repair, no stable publication; no video quality claims; retain the existing alignment repair and startup lifecycle fixes. Existing model smoke and package integrity gates remain active. Cache hits still require process initialization on every launch.

- [x] Add failing tests in tests/test_rtx4050_engine_reuse.py for source-only updates, changes to GPU/runtime/model/build settings, compatible legacy adoption, invalid receipts, missing engines and startup subprocess selection.
- [x] Modify tools/rtx4050_portable.py with scoped identity, bounded legacy adoption, strict receipt checks and an explicit reuse/build branch. Write an engine preparation timing receipt under Runtime/Logs.
- [x] Run focused tests and the existing portable deployment, updater and desktop lifecycle suites; Ruff, compile checks and diff review. Request independent review.
- [x] Commit named maintenance paths and apply using the existing source updater with automatic backup. Verify installed hashes and exercise the installed cache path without exporting engines.
- [x] Where feasible, launch the actual desktop application to verify reuse messages and timing. Report implemented/test/runtime evidence separately; full video quality and release acceptance remain NOT_PROVEN.

Verification: 108 related tests passed with the bundled Python 3.12.10 runtime. Ruff and Python compilation passed. Starlette testclient uses its supported httpx fallback; its deprecation category was filtered for this local run without installing dependencies. A short temporary directory on E: avoids unrelated C: low-free-space and Windows MAX_PATH test environment failures. Independent review identified malformed optional updater receipts/manifests; both top-level and nested cases now have passing regressions. Auxiliary smoke receipts are also bound to the package manifest; previous receipts run once after this update.

Installed functional commit: 81e44d7860cd42ecbd21e75a63dde49e2d617e16. Source update backup: Runtime/Updates/source-5drbo9p0 in the desktop installation. Two actual desktop launches reached the visible workbench. First launch: 445.045 seconds including one-time auxiliary smoke checks. Subsequent launch: 304.685 seconds, including 239.365 seconds of package verification and 0.353 seconds of engine receipt verification; auxiliary smoke receipts were reused without modification. Both original TensorRT engine hashes and modification times remained unchanged. Detailed local evidence: E:/VisionCortex-RTX4050-Desktop-Offline-20260907-R6/Runtime/Updates/engine-reuse-20260909/result.md. At that local acceptance checkpoint, GitHub publication and real-video analysis had not been performed.

## New GitHub branch integration

The user requested publication on a new `codex/rtx4050-engine-reuse-20260909` branch in `RealityLoopAI/VisionCortex`. This candidate includes the customer-update branch through `ece199a3b639d8939beee4ec647322e49caff01a`, preserving its package integrity cache, desktop watchdog, source updater, product changes and native frame provenance. The earlier local alignment/error-classification repair and scoped engine reuse are retained. The existing customer-update branch and stable main are not publication targets.

Combined-tree verification: 392 focused Python tests and 33 desktop Node tests passed; Ruff, Python compilation, Web syntax and repository/credential checks passed. Independent scoped merge review found no concrete regression. The above installed-machine timings describe maintenance commit `81e44d7`, before upstream package integrity caching was integrated; they must not be presented as timings for this combined branch. Combined on-machine startup, full real-video quality, cross-platform CI and stable-release readiness remain NOT_PROVEN pending their own receipts.

## Synchronizing the Windows cache repair

The user approved merging customer update `e921f99037920792cd1a62aa94aa21cb49b9cb7e` into this optimization branch and updating the existing installation with the official manifest-bound updater. Preserve the existing RGB timestamp sampling, scoped engine reuse and precise alignment/OOM messages. The upstream update supplies flat Windows work/SAM2 cache paths, accurate path-length errors, stage versions, activity/step evidence and optional speech recovery.

Review found that moving SAM2 receipts to the shared cache root also removed the previous work-directory identity boundary. The full cache digest now includes the caller's work namespace and complete SAM2 settings; the physical path remains short and keeps the full 64-character digest. Five regression cases first reproduced incorrect reuse across namespace/settings changes, then passed after the fix, including predictor execution counts and unchanged-identity reuse. A Windows-only test assertion was corrected to accept the native drive separator without resolving the mapped drive.

Local verification used bundled Python 3.12.10, existing pytest support and bundled FFmpeg, with an E: temporary root and the existing Starlette testclient deprecation filter. The 27-module run returned 709 passed, 1 skipped and the subsequently corrected Windows drive assertion failure; the complete cache/SAM2 follow-up returned 12 passed, including the five new cases. Desktop Node tests returned 33 passed. Ruff, source/test compilation and repository credential/runtime-output checks passed. Independent review found no remaining blocking issue after the cache identity fix.

These are implementation and deterministic-test results, not a real-video completion receipt. Installation retains runtime data and engines and creates an automatic source backup. The changed MLLM adapter identity requires a fresh AI connection validation; no existing connection receipt is rewritten to bypass that gate. Windows startup, the original s04 analysis and stable release readiness retain their own outstanding verification gates.
