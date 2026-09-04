# VisionCortex Agent Guide

## Scope and authority

- This file applies to the entire repository and supplements user-level instructions.
- Treat the checked-out code, configuration, tests, and current Git state as implementation truth. Historical work-item documents are evidence for their recorded SHA only.
- Read `README.md` and `docs/DUAL-REPOSITORY-RELEASE-POLICY.md` before changing delivery, runtime, evidence, or release behavior.
- For RTX 4060 production runs, also read `docs/RTX4060-Codex-真实六路全链路执行任务书.md` and use `docs/RTX4060-真实六路运行回传模板.md`.

## Project contract

- VisionCortex is a fail-closed, multi-view wet-lab video evidence pipeline. Preserve traceability from source media through detections, cross-view decisions, derived materials, reports, and receipts.
- Separate these claims explicitly: implemented behavior, deterministic-test evidence, actual model invocation, real-video quality evidence, browser-visible delivery, and stable-release readiness.
- Use `PROVEN`, `PARTIAL_EVIDENCE`, or `NOT_PROVEN` when the available evidence does not justify an unconditional conclusion. State the missing gate for partial or unproven claims.
- Synthetic media and `dry-run` outputs validate structure and contracts, not real-experiment accuracy. An HTTP `200`, a generated artifact, or a passing unit test alone is not end-to-end proof.
- Open-vocabulary boxes, SAM2 masks, LabPics masks, or model consensus cannot independently confirm a physical action. Data marked `pseudo_labels_not_ground_truth` must never be presented as human ground truth.

## Repository and release workflow

- Development authority: `https://github.com/kealan-Jun/VisionCortex.git`. Feature work, pull requests, CI, diagnostics, execution evidence, and release candidates belong here.
- Stable authority: `https://github.com/RealityLoopAI/VisionCortex.git`. It accepts only commits already accepted in the development repository; do not perform feature development or reverse merges there.
- Before consequential Git work, verify the remote URLs, default branch, current branch, local SHA, remote SHA, upstream, and working-tree status.
- Stable promotion must preserve the exact tested commit: development SHA equals stable `main` SHA and, when releasing, the peeled stable tag SHA. Do not cherry-pick, amend, rebase, or edit during promotion.
- Apply every relevant promotion gate in `docs/DUAL-REPOSITORY-RELEASE-POLICY.md`. CV-affecting work needs a real-video quality receipt; deterministic tests alone are insufficient.
- The RTX 4060 machine is a frozen execution node. Give it an immutable development SHA; it may execute production video and return evidence, but it must not edit, commit, test, or tune code unless a later task explicitly changes that authority.
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
2. Classify the task as read-only audit, development change, frozen-node execution, release promotion, or live verification; keep actions inside that boundary.
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
- Stable promotion still requires the repository's complete deterministic GitHub Actions suite, including Ubuntu Python 3.11/3.12, Windows Python 3.11, repository policy, and no-NAS contracts.
- Do not silently fix unrelated failures. Separate regressions caused by the change from pre-existing or environment-specific failures.

## Code review rules

- Flag any claim stronger than its receipt, especially synthetic-as-real, test-as-runtime, API-as-browser, or historical-as-current evidence.
- Flag stable commits that are not identical to an accepted development commit, or releases missing applicable promotion gates.
- Flag credential exposure and tracked runtime/model/media artifacts, including secret-bearing reachable history.
- Flag code changes performed on a frozen execution node or unrequested mutation during a read-only audit.
- Flag local/no-NAS configurations that can inherit NAS storage, and any code path that promotes pseudo-labels to ground truth.
