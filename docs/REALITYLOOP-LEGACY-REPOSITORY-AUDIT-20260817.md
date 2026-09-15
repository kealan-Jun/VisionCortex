# RealityLoopAI/VisionCortex legacy repository audit

> 2026-09-15 规则更新：本文中的旧仓库分工仅为历史记录。两个 GitHub 仓库现为平级同步目标，不再执行开发仓向稳定仓的单向晋升；当前规则以 [双仓同步与发布规则](DUAL-REPOSITORY-RELEASE-POLICY.md) 为准。

Audit date: 2026-08-17

Target: `RealityLoopAI/VisionCortex` (private)

Development source: `kealan-Jun/VisionCortex` (private)

## Executive conclusion

The RealityLoopAI repository is not a safe base for continued development. Its two published branch tips belong to an older, unrelated history and must not be merged into the validated evidence pipeline. A small number of engineering ideas are useful, but the implementations are either tightly coupled, very large, or not validated on the current six-view data.

The safe strategy is:

1. keep `kealan-Jun/VisionCortex` as the only development source;
2. port only small deterministic capabilities with focused tests;
3. promote an exact tested commit and immutable tag to `RealityLoopAI/VisionCortex`;
4. never merge commits back from the release repository;
5. archive and then remove the credential-bearing legacy refs after the Ark credential is rotated.

This audit did not merge or execute legacy application code.

## Repository evidence

| Item | Evidence |
|---|---:|
| Legacy default branch | `main` at `35a8ce1aff4b61b82130ac8d1ec92ea75a3cb57e` |
| Legacy baseline branch | `codex/visioncortex-production-baseline` at `5cf2c7b48bbba1d36630b52e6308a70f9a4fecd0` |
| Legacy tag | `v0.1.0-alpha.1`, peeled commit `e0896a81b2f3286e386e4a05d1c1ed80e9cc1910` |
| Legacy `main` commits/files | 99 commits / 2,688 tree files |
| Legacy baseline commits/files | 157 commits / 372 tree files |
| Historical unique blobs | 5,045 blobs / approximately 151 MB |
| Relationship to current code | no common ancestor |
| Current frozen validation baseline | `e67a4d8c0b518fe42e654294a22e8bbe82e0f112` |
| Current frozen tag | `dev036-index-collections-20260817` |
| Current pre-audit regression baseline | 128 tests passed |

The old `main` contains generated artifacts, one-off analysis scripts and a very large product-chain workflow. The baseline branch is cleaner but still descends from the dirty history and therefore does not remove historical credential exposure.

## Security finding

Historical legacy commits contain a Volcengine Ark credential-shaped value in ten one-off Python scripts. The audit intentionally records only paths and fingerprints, never the credential text. Rewriting `main` alone is insufficient while an old branch or tag still reaches those commits.

Required response:

1. revoke/rotate the exposed Ark credential;
2. keep a local, access-controlled legacy bundle for recovery;
3. replace the release repository default branch with the clean tested release history;
4. remove old remote branches and tags after explicit history-purge approval;
5. retain a CI scanner that reports only file paths and credential classes.

Verified local recovery bundle created before any remote rewrite:

- path: `D:\VisionCortexLegacyAudit\RealityLoopAI-VisionCortex-legacy-20260817.bundle`
- size: 38,546,546 bytes
- SHA-256: `f466687e0bfd56d8bd86e56262b73be28185f4734292af5dd6de518aa981ecb9`
- coverage: legacy `main`, legacy production-baseline branch and legacy alpha tag, with complete history

The bundle contains credential-bearing history. It must remain local and access-controlled and must not be attached to an issue, release or NAS collaboration log.

## Reuse decisions

| Legacy capability | Decision | Reason |
|---|---|---|
| Filter-bound opaque cursors | migrated as a small generic helper | Prevents a cursor created under one query from being reused under another; independently testable. |
| Physical lifecycle ledger concept | migrated as a much smaller physical-change projection | The current event JSON already has `state_before`/`state_after`. Only explicit known differences are indexed; no missing state is inferred. |
| Pinned GitHub Actions and repository secret gate | migrated in a reduced workflow | Useful supply-chain practice without importing the old 1,000+ line workflow. |
| Immutable media bindings | not copied | Current archive/index already stores artifact SHA-256, sizes, sidecars, atomic writes and promotion receipts. Copying a second binding system would duplicate authority. |
| Run provenance | not copied | Current evidence, decision, archive and run-metrics ledgers already cover the required provenance. The old module is coupled to its runtime types. |
| Durable task store and leases | concept retained for later modular work | The legacy store is about 4,961 lines. Current `pipeline_status.json` already makes in-progress states recoverable, but post-restart run listing can be improved separately. |
| Adaptive resource controller | not copied | Current performance behavior is validated through the 4060 ledgers. The old controller has no proof on the current persistent-decoder/Fine path. |
| First-person corrective localization | not copied | About 2,752 lines and overlaps the already validated progressive Fine implementation. |
| Cross-view assignment | not copied | About 2,377 lines, depends on explicit workstation configuration, and is not validated on the current data. |
| Full lifecycle/episode decoder | not copied | About 1,399 lines and could change grouping/boundaries. This audit explicitly freezes those decisions. |
| Old YOLO/preprocessing pipeline | rejected | It would invalidate the current five-group, boundary and key-event evidence baseline. |
| Old Web product | rejected as a code dependency | The current Web and archive structure are the product authority; visual ideas can be reviewed separately. |

## Capabilities added without changing CV decisions

- strict, namespace- and filter-bound pagination for key-event searches;
- stable cursor pagination for collection discovery;
- `physical_change_registry.jsonl` and a rebuildable SQLite `physical_changes` table;
- `/api/physical-changes` with archive, object, action, experiment and time filters;
- a reduced CI workflow with pinned actions, complete deterministic tests and path-only credential findings.

These are derived/read layers. They do not change Motion, Fine, cross-view association, experiment grouping, boundaries, key-event selection, media export or MLLM calls.

## Current validated capability and open production issues

DEV-036 confirmed that the existing six-view archive report refresh produced the V2 daily report, V1 professional PDF, 10/10 report evaluation and 18 evidence images with zero model Token usage.

DEV-036 also identified two separate input/runtime blockers that are not addressed by this repository audit:

1. one camera collection contains segment FPS drift from 26.471 to 30.0 and fails the current preflight contract;
2. one persistent Fine decode ended with `expected=814`, `actual=813` at FFmpeg EOF.

Both require a new development commit, focused deterministic tests and a new one-shot real-video authorization. They must not be hidden by relaxing quality gates globally.

## Modularity assessment

The current implementation is much smaller than the legacy monorepo but still has several oversized modules: `pipeline.py`, `archive.py`, `api.py`, `video_io.py`, `grouping.py`, `detection.py`, `actions.py` and `indexing.py`. Refactoring them during performance or CV acceptance work would create unnecessary risk.

Recommended sequence:

1. freeze behavior with characterization tests and receipts;
2. extract pure helpers first (pagination and physical-change projection are the first examples);
3. split storage/API projections next because they do not determine CV quality;
4. refactor detection/grouping only after the 20-minute and quality gates pass together on more than one collection.

## Publication recommendation

After local and CI validation, publish the tested development commit to `kealan-Jun/VisionCortex` first. Promote the exact same commit and an immutable stable tag to `RealityLoopAI/VisionCortex`. Do not cherry-pick, rebuild or edit files in the release repository.
