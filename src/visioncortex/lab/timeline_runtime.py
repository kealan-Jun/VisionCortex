"""Opt-in automatic timeline housekeeping; model execution is a separate runner."""
from .timeline import Timeline
from .timeline_policy import TimelinePolicies
from .timeline_tasks import TimelineTasks
from .timeline_views import TimelineViews
from .timeline_readout import TimelineReadouts
from .timeline_runner import TimelineRunner
from .timeline_publication import TimelinePublication
from .retained_clips import RetainedClips
from .timeline_metrics import TimelineMetrics
from .dynamic_context import DynamicContext
from .material_trace import MaterialTrace
from .operator_scan import OperatorScans
from .comment_transition import CommentTransitions
from .scan_intake import ScanIntake


class TimelineRuntime:
    def __init__(self, core, settings):
        self.settings = settings
        self.timeline = Timeline(core, TimelinePolicies.load(settings.timeline_policy_path))
        self.dynamic_context = DynamicContext(core)
        self.material_trace = MaterialTrace(core)
        self.operator_scans = OperatorScans(core)
        self.comment_transitions = CommentTransitions(core)
        self.scan_intake = ScanIntake(core, settings)
        self.tasks = TimelineTasks(self.timeline)
        self.views = TimelineViews(self.timeline)
        self.readouts = TimelineReadouts(self.timeline)
        self.clips=RetainedClips(self.timeline,settings)
        self.runner = TimelineRunner(self, settings)
        self.publication = TimelinePublication(self, settings)
        self.metrics = TimelineMetrics(self)

    def tick(self, now=None):
        if self.settings.processing_profile != 'timeline-v1':
            return dict(enabled=False)
        return dict(enabled=True, scans=self.scan_intake.tick(), association=self.timeline.reconcile_pending(), closure=self.timeline.close_due(now), **self.views.refresh_dirty())

    def status(self):
        if self.settings.processing_profile != 'timeline-v1':
            return dict(enabled=False)
        with self.timeline.core.db.transaction(readonly=True) as conn:
            tasks = conn.execute('SELECT kind,status,count(*) AS count FROM hj_timeline_task WHERE customer_id=%s GROUP BY kind,status ORDER BY kind,status', (self.timeline.customer,)).fetchall()
            records = conn.execute('SELECT record_status,count(*) AS count FROM hj_experiment_record WHERE customer_id=%s GROUP BY record_status', (self.timeline.customer,)).fetchall()
            attention = conn.execute("SELECT id,media_id,kind,analysis_revision,attempts,error_code FROM hj_timeline_task WHERE customer_id=%s AND status='needs_attention' ORDER BY created_at,id LIMIT 100", (self.timeline.customer,)).fetchall()
        return dict(enabled=True, tasks=tasks, records=records, attention=attention,
                    scan_intake=self.scan_intake.status(),
                    local_providers_configured=self.settings.timeline_providers_path is not None,
                    publication_enabled=self.settings.timeline_publish, model_quality='NOT_PROVEN')
