# VisionCortex Agent Guide

## Scope and authority

- This file applies to the entire repository and supplements user-level instructions.
- Treat the checked-out code, configuration, tests, and current Git state as implementation truth. Historical work-item documents are evidence for their recorded SHA only.
- Read `README.md` and `docs/RELEASE-POLICY.md` before changing delivery, runtime, evidence, or release behavior.
- Production runs use a task-specified immutable commit, the relevant `deployment/<hardware>/README.md`, and privately supplied site configuration and evidence requirements. Keep concrete task books and return records out of Git.

## Architecture and task scope

- Start from `docs/README.md` and `docs/reference/architecture.zh-CN.md` for current responsibilities. Select the relevant source and contract; historical task books do not define today's implementation.
- Use native coding, file and local test tools for ordinary development. Apply skills for an explicitly requested workflow or a concrete capability needed by this task. Read relevant references on demand. Maintain customizations in project/global guidance, not generated plugin caches.
- Continue authorized reversible work. Optional skill advice does not introduce approval gates or expand the task. Offline code changes, mocks and documentation require no new API key; preserve the user's selected provider. Actual external requests, secret creation/writes and live services retain their applicable safeguards.
- Give each algorithm one implementation owner. Keep old public imports and script entry points as narrow compatibility adapters. Inject explicit service/callable dependencies when old monkeypatch behavior must survive; do not synchronize module globals or turn a facade into a hidden implementation.
- Keep `analysis`, `media`, `evidence`, `application`, `web_api`, and training responsibilities separate. HTTP handlers compose application services; CLI and machine workers use public services without importing the HTTP application. The durable run queue and device-day queue have different contracts.
- Before moving executable sources, verify recursive package-relative source identities, stage impact and training trace coverage. Version changed identity recipes; preserve historical receipt readers and never alias an unknown execution change to an accepted hash.
- Before broad refactors, check whether live workers use this checkout. If they can reload its code/configuration, develop in an isolated worktree from the verified SHA and preserve their checkout, environment and data. Repository synchronization alone does not authorize service restarts or live rollout; production execution uses an immutable validated checkout.
- Web domain modules own uploads, archives, materials and tasks through explicit ports. State has one owning store; retain cancellation, stale-response protection, release-bound media URLs and player cleanup when moving code.
- Local profiles inherit a neutral hardware base, with local roots and explicit provider/runtime choices. Compare effective production configuration before and after inheritance changes. Do not weaken frozen NAS paths or dependency-cycle checks.
- Package source from a fixed Git SHA. Verify README local-resource closure, nested Python/web modules, launcher paths and platform-specific evidence gates. Keep formal, candidate and portable package policies distinct.
- During documentation cleanup, inventory tracked root Markdown files and incoming references. Maintain hardware-specific operations in the matching `deployment/<hardware>/README.md`; historical handoffs remain at verified immutable Git identities and in private recoverable backups. Update current links and verify them against checked-out scripts/configuration, preserve immutable historical originals without exposing internal evidence links in customer navigation, and never present historical metrics as current validation. Keep a root compatibility document only for a verified caller and state its purpose.
- `tools/packaging/` intentionally has no `__init__.py`: historical launchers add `tools` to `sys.path`, so a regular top-level `packaging` package would shadow the third-party dependency. Preserve this import boundary when organizing tools.
- Place new tests in domain directories using `tests/repo_paths.py`; move existing tests only with their fixture/import dependencies. Test behavior across boundaries and compatibility ports, rather than counting functions or asserting a file's old physical location.
- Use a configured local workspace for bounded scratch work and reuse a verified interpreter. Check space before large tests, copies or installs. Bulk storage follows the site's verified-mount contract; never relocate active data or environments opportunistically.

## Project contract

- New recorder-native NAS processing must follow `docs/DEVICE-DAY-ARCHIVE-CONTRACT.zh-CN.md` and `docs/contracts/device-day-v1.schema.json`. The device/day five-directory layout and v1 semantics are frozen by the user; do not redesign or rename them without an explicit user change. Preserve historical archive readers. The user now authorizes per-slice removal of the capture video only by atomic replacement with a native soft link to its verified MetaVideo original, after that slice’s preprocessing and index are durably published; never wait for multimodal/report completion. Do not unlink first. Production remains disabled until native NAS links and capture-reader compatibility are verified; use only owned temporary files for capability tests. Never delete audio, CSV or other capture data under this authorization.

- VisionCortex is a fail-closed, multi-view wet-lab video evidence pipeline. Preserve traceability from source media through detections, cross-view decisions, derived materials, reports, and receipts.
- Separate these claims explicitly: implemented behavior, deterministic-test evidence, actual model invocation, real-video quality evidence, browser-visible delivery, and stable-release readiness.
- Use `PROVEN`, `PARTIAL_EVIDENCE`, or `NOT_PROVEN` when the available evidence does not justify an unconditional conclusion. State the missing gate for partial or unproven claims.
- Synthetic media and `dry-run` outputs validate structure and contracts, not real-experiment accuracy. An HTTP `200`, a generated artifact, or a passing unit test alone is not end-to-end proof.
- Open-vocabulary boxes, SAM2 masks, LabPics masks, or model consensus cannot independently confirm a physical action. Data marked `pseudo_labels_not_ground_truth` must never be presented as human ground truth.

## Repository and release workflow

- Synchronize the same reviewed commit to the configured synchronization targets and explicitly shared branches. Resolve targets from trusted local Git configuration; do not embed maintainer identities or internal repository roles in public guidance.
- Before consequential Git work, verify URLs, default branch, current branch, local and target SHAs, upstream, and working-tree status. Complete synchronization requires querying all targets and confirming identical target SHAs.
- Preserve remote commits, other branches and tags; reconcile divergence before pushing. Never force-push or use `--mirror` for routine synchronization.
- When a synchronization target has an explicitly authorized new Git history baseline, begin work from a fresh checkout of that target. Apply reviewed file changes onto its history; never merge, rebase or push retired ancestry into it.
- Keep frozen execution checkouts on their assigned immutable SHA. They are not a source for repository synchronization; preserve their runtime receipts and use the current authoritative checkout for development and pushes.
- Apply `docs/RELEASE-POLICY.md`. Synchronization does not deploy services or establish release readiness. CI is manual-only; routine code management must not trigger or wait for it.
- Execution nodes use task-specified immutable code and private site configuration. They execute and return evidence without editing, testing or tuning code unless explicitly authorized.
- Preserve unrelated work. In a dirty tree, stage only named paths; never use `git add -A` or `git add .`.

## Public defaults and first-clone acceptance

- Public defaults must start a bounded local workflow without NAS, model startup or automatic collection. Site paths, recorder identities, service addresses, private task books, runtime evidence and credentials belong to private deployment configuration/storage and must not enter Git.
- Validate a fresh fixed-SHA checkout independently of the maintainer's environment: documented installation, check-only startup, actual local Web startup, static resources and clean shutdown. Record actual results; do not infer them from existing environments or previous logs.
- Customer documentation and packages include installation, operation, configuration and applicable acceptance requirements. Exclude internal maintenance instructions, runtime journals and historical remote projections from customer navigation and delivery.
- Inventory tracked file types during privacy review. Decode PDF and office/archive contents and inspect screenshots; raw text searches cannot clear compressed or binary documents. Remove stale generated copies or regenerate them from reviewed sources, and check both the source tree and customer packages.
- Check hosting metadata, tags, historical commits and pull-request references separately from current files. A clean default branch does not establish that old repository content is no longer accessible.
- Moving internal documents out of the current tree does not erase existing Git history. Preserve originals at verified immutable identities and private recoverable backups; never rewrite evidence or claim historical removal without a separately authorized procedure.

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
4. Run the smallest relevant local checks for implementation changes. Routine Git synchronization does not require running CI. Use applicable runtime evidence for stronger deployment claims.
5. Review the diff and status for unintended files, generated output, credential shapes, and scope drift before handoff or publication. Include newly added files in whitespace, syntax and resource checks; an unstaged `git diff` alone omits untracked files.
6. Report the exact branch/SHA, checks run, evidence obtained, and remaining `PARTIAL_EVIDENCE` or `NOT_PROVEN` boundaries.

## Verification guide

- Lightweight development install: `python -m pip install -e ".[dev]"`.
- Focused Python check: `python -m pytest tests/<relevant_test>.py`.
- Python source changes: run relevant tests plus `ruff check src tests` and `python -m compileall -q src tests` when available.
- Keep deterministic model-port tests independent of the optional GPU stack. Mock the library boundary for fake-model contract tests; use an explicit skip for tests requiring real optional tensor/model interfaces. A missing declared core dependency is an environment issue to resolve, never a reason to weaken evidence assertions.
- Web JavaScript changes: run `node --check` on every changed JS module plus the relevant Python/Web characterization and module-interface tests.
- Ubuntu deployment-script changes: run `bash -n` on the changed scripts.
- Documentation-only changes: run `git diff --check` and validate referenced paths/commands. Do not install dependencies or run code tests locally unless the documentation changes an executable contract.
- Keep the GitHub Actions workflow manual-only. Pushes and pull requests do not trigger it. Do not launch CI for routine synchronization; if explicitly requested later, report its actual result separately from successful Git synchronization.
- Do not silently fix unrelated failures. Separate regressions caused by the change from pre-existing or environment-specific failures.

## Code review rules

- Flag any claim stronger than its receipt, especially synthetic-as-real, test-as-runtime, API-as-browser, or historical-as-current evidence.
- Flag synchronization claims when the two target SHAs differ, overwritten remote history, or formal releases missing applicable evidence gates.
- Flag credential exposure and tracked runtime/model/media artifacts, including secret-bearing reachable history.
- Flag code changes performed on a frozen execution node or unrequested mutation during a read-only audit.
- Flag local/no-NAS configurations that can inherit NAS storage, and any code path that promotes pseudo-labels to ground truth.

- Device-day frame semantics are frozen by the user's clarification: `key_frames` / `keyframes` are reserved for selected events in the established five physical-action classes. Uniform samples for inactive/irrelevant intervals are `scene_frames` with `frame_kind: scene_sample`; inactive intervals must have an empty `key_frames` list. Keep original audio and actual STT outputs in their own device/day. Never substitute another day's audio or execution evidence for a day with no audio.

- User-final archive spelling is PascalCase, with no spaces or ordering numbers, for both folders and files: MetaVideo, ProcessedClips, MultimodalUnderstanding, LaboratoryDailyReport, Comment. Preserve date/camera identities and media timestamps. Earlier sentence-case or numbered spellings are historical migration inputs only.
