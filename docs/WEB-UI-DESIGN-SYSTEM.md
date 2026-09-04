# VisionCortex Web UI design system

This document defines the product-facing rules for the local VisionCortex web application. It applies to experiment creation, progress, archives, key materials, reports, and system status. It does not change evidence or model contracts.

## Product principles

1. Lead with the user's next decision. Technical receipts remain available, but they are collapsed by default.
2. Preserve evidence meaning. “画面确认”, “模型提示”, and “证据不足” must remain visually and verbally distinct.
3. Keep archive identity immutable. Friendly names, owners, tags, and notes are workspace metadata; the original archive identifier must remain visible and unchanged.
4. Use progressive disclosure. Lists show an overview, while video, objects, provenance, and metrics open on demand.
5. Fail closed. Missing media, reports, or verification records must use an explicit state instead of an empty surface or a success-like presentation.

## Visual tokens

- Primary brand: `#1e4a52`
- Secondary brand: `#2f747b`
- Page background: `#f3f7f6`
- Main text: `#18343a`
- Observed evidence: `#238878`
- Model inference: `#6274c4`
- Uncertain evidence: `#c58b39`
- Destructive or failed state: `#b84d45`
- Control radius: `9px`
- Card radius: `14px`
- Hero radius: `18px`
- Base spacing sequence: `4, 8, 12, 16, 24px`

Use borders and tonal surfaces before adding shadows. Reserve the deepest brand fill for the primary action and the active navigation item.

## Typography and language

- Page title: 27–36px, semibold.
- Section title: 17–22px, semibold.
- Body: 12–14px with 1.55–1.7 line height.
- Metadata: 9–11px. Monospace is limited to identifiers, paths, timestamps, and machine values.
- Prefer task language such as “打开实验文件” and “查看分析进度”. Avoid exposing pipeline stage names in primary interfaces.
- Never describe model inference as an observed fact or human ground truth.

## Component rules

### Navigation

- Desktop uses the global sidebar and a four-item experiment result navigation.
- The result navigation becomes compact while scrolling.
- Mobile keeps four primary destinations plus “更多” in the bottom navigation. Low-frequency result and system destinations live in the More sheet.

### Experiment library

- Search covers the friendly name, archive identifier, owner, and tags.
- Time, status, owner, and tag filters can be combined.
- List view is the density default; card view is an optional visual scan mode.
- Status text must come from the recorded pipeline state.

### Key materials and video

- Material cards are overview-first. Only one material detail is expanded during continuous review.
- The available archive media mode must be stated honestly. Do not render a first-person or third-person switch unless independent media URLs exist.
- Standard review controls: previous/next material, 0.5×–2× speed, loop, fullscreen, start/peak/end markers, and documented keyboard shortcuts.

### Feedback states

- Neutral: no data or no matching result.
- Progress: valid work is still being generated.
- Success: a recorded gate or operation completed.
- Error: the user needs a retry, return path, or another concrete recovery action.
- Every non-success state includes a plain-language explanation. Recoverable states include an action.

## Responsive behavior

- `>1180px`: full desktop information density.
- `851–1180px`: two-column summaries; technical tables may collapse secondary columns.
- `561–850px`: single-column content with compact controls.
- `<=560px`: bottom navigation, single-column forms and cards, no fixed result navigation, and touch targets of at least 40px for primary actions.

## Accessibility checklist

- Maintain visible keyboard focus.
- Use native buttons, links, selects, details, and dialogs.
- Provide text labels for icons and status colors.
- Keep loading, selection, and error regions announced with `aria-live`, `role=status`, or `role=alert` as appropriate.
- Do not rely on hover to expose a required action.
