# VisionCortex Agent Guide

## Scope and authority

- This file applies to the entire repository and supplements user-level instructions.
- Treat the checked-out code, configuration, tests, and current Git state as implementation truth. Historical work-item documents are evidence for their recorded SHA only.
- Read `README.md` and `docs/DUAL-REPOSITORY-RELEASE-POLICY.md` before changing delivery, runtime, evidence, or release behavior.
- For RTX 4060 production runs, also read `docs/RTX4060-Codex-真实六路全链路执行任务书.md` and use `docs/RTX4060-真实六路运行回传模板.md`.

## Project contract

- New recorder-native NAS processing must follow `docs/DEVICE-DAY-ARCHIVE-CONTRACT.zh-CN.md` and `docs/contracts/device-day-v1.schema.json`. The device/day five-directory layout and v1 semantics are frozen by the user; do not redesign or rename them without an explicit user change. Preserve historical archive readers. The user now authorizes per-slice removal of the capture video only by atomic replacement with a native soft link to its verified MetaVideo original, after that slice’s preprocessing and index are durably published; never wait for multimodal/report completion. Do not unlink first. Production remains disabled until native NAS links and capture-reader compatibility are verified; use only owned temporary files for capability tests. Never delete audio, CSV or other capture data under this authorization.

- VisionCortex is a fail-closed, multi-view wet-lab video evidence pipeline. Preserve traceability from source media through detections, cross-view decisions, derived materials, reports, and receipts.
- Separate these claims explicitly: implemented behavior, deterministic-test evidence, actual model invocation, real-video quality evidence, browser-visible delivery, and stable-release readiness.
- Use `PROVEN`, `PARTIAL_EVIDENCE`, or `NOT_PROVEN` when the available evidence does not justify an unconditional conclusion. State the missing gate for partial or unproven claims.
- Synthetic media and `dry-run` outputs validate structure and contracts, not real-experiment accuracy. An HTTP `200`, a generated artifact, or a passing unit test alone is not end-to-end proof.
- Open-vocabulary boxes, SAM2 masks, LabPics masks, or model consensus cannot independently confirm a physical action. Data marked `pseudo_labels_not_ground_truth` must never be presented as human ground truth.

## Repository and release workflow

- The user changed the repository policy on 2026-09-15: `https://github.com/kealan-Jun/VisionCortex.git` and `https://github.com/RealityLoopAI/VisionCortex.git` are equal synchronization targets for the same codebase. There is no development-only or stable-only repository and no one-way promotion requirement.
- Synchronize the same reviewed local commit to both `main` branches and any explicitly shared working branch. Existing remote aliases (`development` and `origin`) are historical names, not different authority levels.
- Before consequential Git work, verify the remote URLs, default branch, current branch, local SHA, remote SHA, upstream, and working-tree status.
- Synchronization is complete only after querying both remotes and confirming identical target SHAs. Preserve remote commits, other branches and tags; reconcile divergence before pushing. Never force-push or use `--mirror` to erase remote history as part of routine synchronization.
- Apply `docs/DUAL-REPOSITORY-RELEASE-POLICY.md`. Code synchronization does not deploy services or assert release readiness. Formal releases still need deterministic checks, relevant real-video receipts for CV changes, and documented limitations.
- The RTX 4060 machine is a frozen execution node. Give it an immutable synchronized SHA from either repository; it may execute production video and return evidence, but it must not edit, commit, test, or tune code unless a later task explicitly changes that authority.
- Preserve unrelated work. In a dirty tree, stage only named paths; never use `git add -A` or `git add .`.

## Data, runtime, and security boundaries

- Do not commit model binaries, TensorRT engines, raw video, NAS data, runtime output, caches, failed staging material, or credentials.
- Supply credentials through environment variables or an approved untracked secret store. Never print, log, copy into fixtures, or expose credential values; CI may identify only the path and credential class.
- Treat NAS access, GPU/model startup, dependency installation, engine rebuilds, and full-media reads as material runtime actions. A read-only audit must not perform them.
- Before a real run, revalidate checkout SHA, interpreter and dependency environment, configuration, input manifest, storage roots, model/engine identity, and output destination. Do not infer readiness from the existence of a `.venv` or engine file.
- Local/no-NAS profiles must not inherit NAS paths. Keep local input, cache, staging, runtime, and archive roots within the configured local runtime root.
- Do not rewrite frozen NAS paths or copy source media as a convenience. Use repository configuration and resolvers, and report a visibility or authorization failure rather than weakening the input contract.

## Change workflow

1. Inspect the applicable instructions, Git state, canonical docs, relevant source, configuration, and tests.
2. Classify the task as read-only audit, development change, repository synchronization, frozen-node execution, release deployment, or live verification; keep actions inside that boundary.
3. Make the smallest coherent change that addresses the root problem. Avoid unrelated refactors, broad formatting, speculative features, and one-off helper artifacts.
4. Run the smallest relevant local checks, then rely on the required cross-platform CI and runtime gates for stronger claims.
5. Review the diff and status for unintended files, generated output, credential shapes, and scope drift before handoff or publication.
6. Report the exact branch/SHA, checks run, evidence obtained, and remaining `PARTIAL_EVIDENCE` or `NOT_PROVEN` boundaries.

## Verification guide

- Lightweight development install: `python -m pip install -e ".[dev]"`.
- Focused Python check: `python -m pytest tests/<relevant_test>.py`.
- Python source changes: run relevant tests plus `ruff check src tests` and `python -m compileall -q src tests` when available.
- Web JavaScript changes: run `node --check src/visioncortex/web/app.js` plus the relevant Python/Web characterization tests.
- Ubuntu deployment-script changes: run `bash -n` on the changed scripts.
- Documentation-only changes: run `git diff --check` and validate referenced paths/commands. Do not install dependencies or run code tests locally unless the documentation changes an executable contract.
- Both repositories run the same deterministic GitHub Actions suite, including Ubuntu Python 3.11/3.12, Windows Python 3.11, repository policy, and no-NAS contracts. Report actual CI status separately from successful Git synchronization.
- Do not silently fix unrelated failures. Separate regressions caused by the change from pre-existing or environment-specific failures.

## Code review rules

- Flag any claim stronger than its receipt, especially synthetic-as-real, test-as-runtime, API-as-browser, or historical-as-current evidence.
- Flag synchronization claims when the two target SHAs differ, overwritten remote history, or formal releases missing applicable evidence gates.
- Flag credential exposure and tracked runtime/model/media artifacts, including secret-bearing reachable history.
- Flag code changes performed on a frozen execution node or unrequested mutation during a read-only audit.
- Flag local/no-NAS configurations that can inherit NAS storage, and any code path that promotes pseudo-labels to ground truth.

- Device-day frame semantics are frozen by the user's clarification: `key_frames` / `keyframes` are reserved for selected events in the established five physical-action classes. Uniform samples for inactive/irrelevant intervals are `scene_frames` with `frame_kind: scene_sample`; inactive intervals must have an empty `key_frames` list. Keep original audio and actual STT outputs in their own device/day. Never substitute another day's audio or execution evidence for a day with no audio.

- User-final archive spelling is PascalCase, with no spaces or ordering numbers, for both folders and files: MetaVideo, ProcessedClips, MultimodalUnderstanding, LaboratoryDailyReport, Comment. Preserve date/camera identities and media timestamps. Earlier sentence-case or numbered spellings are historical migration inputs only.
