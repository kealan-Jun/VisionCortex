from __future__ import annotations

import argparse
import json
import signal
import threading
from pathlib import Path

from .components import ROOT, verify_components
from .config import Settings, load_env
from .db import Database
from .domain import Core, clean
from .ingest import Ingest
from .processing import Worker
from .projections import Projections
from .retention import Retention
from .rules import Rules
from .cache import CachePruner
from .delivery import NativeDelivery
from .live import ReceiverSupervisor
from .material_batch import MaterialBatch
from .equipment import equipment_details
from .timeline_runtime import TimelineRuntime


def runtime(settings):
    db = Database(settings.database_url)
    with db.transaction(readonly=True) as conn:
        row = conn.execute("SELECT id FROM customer_instance WHERE code=%s", (settings.customer_code,)).fetchone()
        if not row:
            raise ValueError("Customer instance not initialized; run init explicitly")
    return Core(db, row["id"])


def main(argv=None):
    parser = argparse.ArgumentParser(prog="visioncortex.lab")
    parser.add_argument("--env-file", type=Path)
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("migrate")
    init = commands.add_parser("init")
    init.add_argument("--scope", choices=["customer", "development", "fixture"], default="development")
    commands.add_parser("verify-components")
    commands.add_parser("verify-rules")
    commands.add_parser("publish")
    commands.add_parser("tick")
    runner = commands.add_parser("run")
    runner.add_argument("--role", choices=["all", "ingest", "worker", "index", "retention", "cache", "delivery", "receiver", "timeline", "timeline-model", "timeline-publication", "producer-ingress"], default="all")
    commands.add_parser("schedule-models")
    commands.add_parser("status")
    equipment = commands.add_parser('equipment-details')
    equipment.add_argument('uuid')
    equipment.add_argument('--as-of')
    batch_summary = commands.add_parser("material-batch-summary")
    batch_summary.add_argument("kind", choices=["source-lot", "process-batch"])
    batch_summary.add_argument("id")
    dataset = commands.add_parser("dataset")
    dataset.add_argument("action", choices=["validate", "prepare"])
    dataset.add_argument("manifest", type=Path)
    dataset.add_argument("image_root", type=Path)
    dataset.add_argument("--output", type=Path)
    execute = commands.add_parser("command")
    execute.add_argument("name")
    execute.add_argument("request", type=Path)
    serve = commands.add_parser("serve")
    serve.add_argument("--host", default="127.0.0.1")
    serve.add_argument("--port", type=int, default=8197)
    args = parser.parse_args(argv)
    if args.command == "verify-rules":
        print(json.dumps(Rules().identity()))
        return
    if args.command == "verify-components":
        manifest = verify_components()
        print(json.dumps({"verified": True, "components": [{"name": c["name"], "commit": c["source_commit"]} for c in manifest["components"]], "models_included": False}))
        return
    if args.env_file:
        load_env(args.env_file)
    settings = Settings.from_env()
    settings.prepare()
    db = Database(settings.database_url)
    if args.command == "migrate":
        print(json.dumps({"applied": db.migrate(ROOT / "migrations/lab")}))
        return
    if args.command == "init":
        print(json.dumps(Core.customer_instance(db, settings.customer_code, scope=args.scope)))
        return
    core = runtime(settings)
    if args.command == 'equipment-details':
        print(json.dumps(equipment_details(core, args.uuid, as_of=args.as_of), ensure_ascii=False))
        return
    if args.command == "material-batch-summary":
        batches = MaterialBatch(core)
        result = batches.source_lot(args.id) if args.kind == "source-lot" else batches.process_batch(args.id)
        print(json.dumps(clean(result), ensure_ascii=False))
        return
    timeline = TimelineRuntime(core, settings)
    if args.command == "serve":
        from .api import create_app
        import uvicorn
        uvicorn.run(create_app(core, settings), host=args.host, port=args.port)
        return
    if args.command == "schedule-models":
        result = {"created": Worker(core, settings).schedule_models()}
    elif args.command == "publish":
        result = Retention(core, settings).tick()
    elif args.command == "dataset":
        from .training import validate_dataset, prepare_dataset
        if args.action == "prepare":
            if args.output is None or not args.output.absolute().is_relative_to(settings.runtime_root):
                raise ValueError("Dataset output must use the bounded local project runtime")
            result = prepare_dataset(args.manifest, args.image_root, args.output, customer_code=settings.customer_code)
        else:
            result = validate_dataset(args.manifest, args.image_root, customer_code=settings.customer_code)
    elif args.command == "status":
        with db.transaction(readonly=True) as conn:
            jobs = conn.execute("SELECT status,count(*) AS count FROM processing_job WHERE customer_id=%s GROUP BY status", (core.customer,)).fetchall()
        result = dict(customer_id=core.customer, jobs=jobs, timeline=timeline.status(), retention=Retention(core, settings).status(), processing_rules=Rules().identity(), **Projections(core).status())
    elif args.command == "command":
        names = {"assign-camera-person": "assign_camera_person", "reserve-equipment-catalogue": "reserve_equipment_catalogue", "issue-equipment": "issue_equipment", "activate-equipment": "activate_equipment", "retire-equipment": "retire_equipment", "issue-identity": "issue_identity", "register": "register", "register-material": "register_material", "assign-tag": "assign_tag", "bind-container": "bind_container", "equipment-use": "equipment_use", "operate": "operate", "correct-operation": "correct_operation", "correct-fact": "correct_fact", "confirm-candidate": "confirm_candidate", "confirm-usage": "confirm_usage", "reserve": "reserve"}
        if args.name not in names:
            from .timeline_commands import command_methods
            methods = command_methods(timeline)
            if args.name not in methods:
                raise ValueError("Unknown business command")
            result = methods[args.name](**json.loads(args.request.read_text()))
        else:
            result = getattr(core, names[args.name])(**json.loads(args.request.read_text()))
    else:
        ingest, worker, index = Ingest(core, settings), Worker(core, settings), Projections(core)
        stop = threading.Event()
        for sig in (signal.SIGINT, signal.SIGTERM):
            signal.signal(sig, lambda *_: stop.set())
        if args.command == "run":
            def loop(role, action):
                while not stop.is_set():
                    try:
                        value = action()
                        if value and (not isinstance(value, dict) or value.get("processed") or value.get("errors") or value.get("indexed") or value.get("job_id") or value.get('task_id') or value.get('files') or value.get('reports') or value.get('closure', {}).get('archived')):
                            print(json.dumps(clean(dict(role=role, result=value)), ensure_ascii=False), flush=True)
                    except Exception as exc:
                        print(json.dumps({"role": role, "status": "needs_attention", "error_code": type(exc).__name__}), flush=True)
                    stop.wait(settings.poll_seconds)
            receiver = ReceiverSupervisor(core,settings)
            actions = {"ingest": ingest.discover_once, "worker": worker.run_once, "index": index.tick, "retention": Retention(core, settings).tick, "cache": CachePruner(core, settings).tick, "delivery":NativeDelivery(core,settings).tick, 'receiver':receiver.tick, 'timeline': timeline.tick, 'timeline-model': timeline.runner.run_once, 'timeline-publication': timeline.publication.tick}
            if args.role == 'producer-ingress':
                from .producer_ingress import ProducerIngress
                actions = {'producer-ingress': ProducerIngress(core, settings).tick}
            threads = [threading.Thread(target=loop, args=(role, action), name="vc-lab-" + role, daemon=True) for role, action in actions.items() if args.role in {"all", role}]
            for thread in threads:
                thread.start()
            while not stop.wait(0.5):
                pass
            for thread in threads:
                thread.join(timeout=5)
            receiver.close()
            return
        while True:
            result = dict(ingest=ingest.discover_once(), delivery=NativeDelivery(core,settings).tick(), worker=worker.run_once(), index=index.tick(), retention=Retention(core, settings).tick(), cache=CachePruner(core, settings).tick(), timeline_model=timeline.runner.run_once(), timeline=timeline.tick(), timeline_publication=timeline.publication.tick())
            if args.command == "tick":
                break
            if result["ingest"]["processed"] or result["ingest"]["errors"] or result["worker"] or result["index"]["indexed"]:
                print(json.dumps(clean(result), ensure_ascii=False), flush=True)
            if stop.wait(settings.poll_seconds):
                break
        if args.command == "run":
            return
    print(json.dumps(clean(result), ensure_ascii=False))
