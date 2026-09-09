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

Installed functional commit: 81e44d7860cd42ecbd21e75a63dde49e2d617e16. Source update backup: Runtime/Updates/source-5drbo9p0 in the desktop installation. Two actual desktop launches reached the visible workbench. First launch: 445.045 seconds including one-time auxiliary smoke checks. Subsequent launch: 304.685 seconds, including 239.365 seconds of package verification and 0.353 seconds of engine receipt verification; auxiliary smoke receipts were reused without modification. Both original TensorRT engine hashes and modification times remained unchanged. Detailed local evidence: E:/VisionCortex-RTX4050-Desktop-Offline-20260907-R6/Runtime/Updates/engine-reuse-20260909/result.md. No GitHub publication or real-video analysis was performed for this optimization.
