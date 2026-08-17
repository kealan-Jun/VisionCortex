# VisionCortex dual-repository release policy

## Repository roles

### Development authority

`https://github.com/kealan-Jun/VisionCortex.git`

- all feature branches and pull requests;
- deterministic unit/integration tests;
- 4060 task freezes and evidence feedback;
- issue tracking and development diagnostics;
- release candidate tags.

### Stable release authority

`https://github.com/RealityLoopAI/VisionCortex.git`

- only exact commits already accepted in the development repository;
- a minimal `main` branch plus immutable stable tags/releases;
- no direct feature development;
- no reverse merges into the development repository;
- no runtime data, NAS artifacts, keys or failed staging output.

## Promotion gates

A commit is eligible for stable promotion only when all applicable gates are recorded:

1. development worktree and remote SHA match;
2. complete deterministic test suite passes;
3. repository credential/runtime-output policy passes;
4. CV-affecting changes have a real-video quality receipt;
5. performance claims identify whether they cover preprocessing or the complete chain;
6. MLLM stages include input/output/total Token usage;
7. archive promotion is complete and SHA-256 receipts pass;
8. open limitations are written into the release notes.

Pure derived-index, report-template or Web-read changes do not require an expensive full-video rerun when characterization tests prove they cannot alter CV decisions. They still require the full deterministic test suite.

## Promotion invariant

The stable release commit must be byte-for-byte identical to the tested development commit:

```text
development commit SHA == stable main SHA == stable tag peeled SHA
```

No cherry-picking or editing is allowed during promotion.

## Credential and history policy

- credentials are supplied by environment or an untracked local secret store;
- CI may report the path and credential class, never the value;
- a repository with reachable secret-bearing history cannot be called clean;
- legacy recovery bundles stay local and access-controlled;
- credential rotation is required even after history is rewritten.

## 4060 operating rule

Every 4060 task receives an immutable commit SHA from the development repository. The 4060 machine may run production video and return evidence, but it must not edit, commit, test or tune the code unless a later task explicitly changes that authority.
