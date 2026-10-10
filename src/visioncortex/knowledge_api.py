"""Compatibility installer for knowledge, task-event and recorder HTTP contracts."""
from .web_api.knowledge import Question
from .web_api.task_events import Acknowledgement
from .web_api.receiver import UploadCompletion


def install_routes(app, settings):
    from .web_api import knowledge, task_events, receiver
    knowledge.install_routes(app, settings)
    task_events.install_routes(app, settings)
    receiver.install_routes(app, settings)


__all__ = ["Question", "Acknowledgement", "UploadCompletion", "install_routes"]
