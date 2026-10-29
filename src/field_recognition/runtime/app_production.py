"""Compatibility helpers for explicitly constructed managed runtime instances."""


def integrate_production(core, app_instance):
    if core.get("app") is not app_instance or not core.get("RUNTIME_ENABLED"):
        raise RuntimeError(
            "Prepare migrations and start a managed API/worker role; runtime patching is not supported"
        )
    return core


def get_production_core(runtime=None, *, config=None, service_role="api", **runtime_options):
    """Return an existing managed instance, or explicitly construct a new one."""
    if runtime is not None:
        if config is not None or runtime_options:
            raise TypeError("Pass either an existing runtime or factory settings")
        return integrate_production(runtime, runtime["app"])
    if config is None and not runtime_options:
        raise TypeError(
            "Pass an explicit runtime or configuration; use create_runtime to construct one"
        )
    from field_recognition.api.application import create_app
    from field_recognition.runtime.factory import create_runtime

    core = create_runtime(
        config, service_role=service_role, production_enabled=True, **runtime_options
    )
    return integrate_production(core, create_app(core))
