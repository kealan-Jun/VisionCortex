"""Managed runtime deployment instructions; never patch a running application."""


def enable_production_mode():
    raise RuntimeError("Use prepare_runtime.py and deployment/field-recognition.target")


def patch_app_at_startup():
    raise RuntimeError(
        "Runtime patching was removed; production boundaries are installed by app.py"
    )


if __name__ == "__main__":
    print(
        "Prepare with prepare_runtime.py, then install deployment/field-recognition*.service and field-recognition.target. See docs/ProductionRuntime.md."
    )
