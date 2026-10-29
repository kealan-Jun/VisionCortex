"""Internal software operations, not an Agent tool or Feishu UI contract."""


def command_methods(runtime):
    def worker_call(method):
        def invoke(**body):
            if 'now' in body:
                raise ValueError('Worker leases use the server clock')
            return method(**body)
        return invoke
    return {
        'timeline-operator-scan': runtime.operator_scans.record,
        'timeline-comment-transition': runtime.comment_transitions.apply,
        'timeline-operator-context': runtime.operator_scans.context,
        'timeline-label-event': runtime.timeline.label_event,
        'timeline-resolve-qr': runtime.timeline.resolve_qr,
        'timeline-create-run': runtime.timeline.create_run,
        'timeline-link-runs': runtime.timeline.link_runs,
        'timeline-bind': runtime.timeline.bind,
        'timeline-bind-qr': runtime.timeline.bind_qr,
        'timeline-add-activity': runtime.timeline.add_activity,
        'timeline-associate-activity': runtime.timeline.associate_activity,
        'timeline-claim-task': worker_call(runtime.tasks.claim),
        'timeline-submit-result': worker_call(runtime.tasks.submit),
        'timeline-reanalyze': runtime.tasks.reanalyze,
        'timeline-register-ocr-frame': runtime.readouts.register_frame,
        'timeline-register-temporal-ocr': runtime.tasks.register_temporal_ocr,
        'timeline-register-video-qr': runtime.tasks.register_video_qr,
        'timeline-register-clip': runtime.clips.register,
        'timeline-metrics': runtime.metrics.day,
        'timeline-register-panel-link': runtime.dynamic_context.register_panel_link,
        'timeline-observe-context': runtime.dynamic_context.observe,
        'timeline-observe-context-batch': runtime.dynamic_context.append_batch,
        'timeline-context-window': runtime.dynamic_context.query_window,
        'timeline-resolve-material-label': runtime.material_trace.resolve_label,
        'timeline-observe-material': runtime.material_trace.append_observation,
        'timeline-material-observation-history': runtime.material_trace.history,
        'timeline-material-history': runtime.material_trace.material_history,
    }
