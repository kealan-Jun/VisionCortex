from __future__ import annotations

import json
from pathlib import Path

import pytest

from labvision_evidence import hardware_acceptance


def test_hardware_tuning_selects_fewest_worker_profile_within_two_percent(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    throughputs = {1: 540.0, 2: 545.0, 3: 510.0}
    gpu_peaks = {1: 91.0, 2: 96.0, 3: 99.0}

    def fake_run(
        output: Path,
        _config: dict,
        _media_paths: list[Path],
        *,
        duration_seconds: float,
        workers_per_role: int,
    ) -> Path:
        assert duration_seconds == 10.0
        output.mkdir(parents=True)
        receipt = output / "hardware-acceptance.json"
        receipt.write_text(
            json.dumps(
                {
                    "passed": True,
                    "workload": {
                        "aggregate_tensor_rt_frames_per_second": throughputs[
                            workers_per_role
                        ]
                    },
                    "peaks": {
                        "gpu_compute_percent": gpu_peaks[workers_per_role],
                        "gpu_memory_used_mib": 7000 + workers_per_role * 1000,
                    },
                }
            ),
            encoding="utf-8",
        )
        return receipt

    monkeypatch.setattr(hardware_acceptance, "run_hardware_acceptance", fake_run)

    receipt = hardware_acceptance.tune_hardware_acceptance(
        tmp_path / "tuning",
        {},
        [tmp_path / "clip.mp4"] * 6,
        duration_seconds=10.0,
    )

    payload = json.loads(receipt.read_text(encoding="utf-8"))
    assert payload["selected_workers_per_role"] == 1
    assert payload["selected_aggregate_tensor_rt_frames_per_second"] == 540.0
    assert payload["hardware_compute_saturated"] is True
    assert payload["nas_accessed"] is False
    assert payload["source_copy_bytes"] == 0


def test_hardware_acceptance_rejects_nas_media_before_probe(tmp_path: Path) -> None:
    with pytest.raises(RuntimeError, match="local non-NAS media"):
        hardware_acceptance.run_hardware_acceptance(
            tmp_path / "output",
            {},
            [Path("/home/x1/桌面/nas/video.mp4")] * 6,
            duration_seconds=10.0,
        )


def test_hardware_acceptance_rejects_network_filesystem(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setattr(hardware_acceptance, "_mount_filesystem_type", lambda _: "cifs")

    with pytest.raises(RuntimeError, match="local non-NAS media"):
        hardware_acceptance._require_local_media(tmp_path / "video.mp4")


def test_hardware_tuning_rejects_unreceipted_existing_output(tmp_path: Path) -> None:
    output = tmp_path / "tuning"
    output.mkdir()

    with pytest.raises(FileExistsError, match="already exists"):
        hardware_acceptance.tune_hardware_acceptance(
            output,
            {},
            [tmp_path / "clip.mp4"] * 6,
            duration_seconds=10.0,
        )
