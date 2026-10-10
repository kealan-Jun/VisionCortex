"""Offline tests of explicit site selection and instance ownership guards."""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest
import yaml

from repo_paths import ROOT


SCRIPTS = ROOT / 'deployment' / 'rtx4060'
RUNTIME = SCRIPTS / 'Runtime-Config.psm1'
POWERSHELL = shutil.which('powershell.exe') or shutil.which('pwsh')
requires_powershell = pytest.mark.skipif(
    POWERSHELL is None, reason='PowerShell parser and module mocks are unavailable'
)


def _configuration_probe(profile: Path) -> subprocess.CompletedProcess[str]:
    source = RUNTIME.read_text(encoding='utf-8').split("$code = @'\n", 1)[1].split("\n'@", 1)[0]
    env = {key: value for key, value in os.environ.items() if not key.startswith('VISIONCORTEX_')}
    env['PYTHONPATH'] = str(ROOT / 'src')
    return subprocess.run(
        [sys.executable, '-B', '-c', source, str(ROOT), str(profile)],
        cwd=ROOT, env=env, capture_output=True, text=True, check=False,
    )


def test_node_scripts_refuse_the_unconfigured_public_production_template():
    result = _configuration_probe(ROOT / 'configs/rtx4060-laptop-production.yaml')
    assert result.returncode != 0
    assert 'explicit configured private production site' in result.stderr


def test_node_configuration_uses_only_selected_private_paths_and_registration(tmp_path):
    profile = tmp_path / 'site.yaml'
    profile.write_text(yaml.safe_dump({
        'project': {'run_purpose': 'production', 'site_configuration_required': False},
        'runtime': {'local_only': False},
        'storage': {
            'local_runtime_root': str(tmp_path / 'runtime'),
            'local_cache_root': str(tmp_path / 'cache'),
            'archive_root': str(tmp_path / 'archive'),
            'index_csv': str(tmp_path / 'index.csv'),
        },
        'models': {
            'first_person': str(tmp_path / 'first.pt'),
            'third_person': str(tmp_path / 'third.pt'),
            'sha256_by_role': {'first_person': 'a' * 64, 'third_person': 'b' * 64},
        },
        'fixed_benchmark': {
            'enabled': True, 'experiment_id': 'EXAMPLE-SESSION',
            'archive_name': 'ExampleSession', 'source_count': 6,
        },
    }))
    result = _configuration_probe(profile)
    assert result.returncode == 0, result.stderr
    value = json.loads(result.stdout)
    assert value['config_path'] == str(profile)
    assert value['runtime_root'] == str(tmp_path / 'runtime')
    assert value['archive_root'] == str(tmp_path / 'archive')
    assert value['models']['first_person'] == str(tmp_path / 'first.pt')
    assert value['model_hashes']['third_person'] == 'b' * 64
    assert value['fixed_benchmark']['archive_name'] == 'ExampleSession'
    assert value['fixed_benchmark']['enabled'] is True
    assert not (tmp_path / 'runtime').exists()
    assert not (tmp_path / 'archive').exists()


def _run_powershell(command: str) -> dict:
    result = subprocess.run(
        [POWERSHELL, '-NoLogo', '-NoProfile', '-NonInteractive', '-Command', command],
        check=True, capture_output=True, text=True,
    )
    return json.loads(result.stdout.strip())


@requires_powershell
def test_all_node_scripts_parse_without_executing_services_or_probes():
    command = "\n".join([
        "$ErrorActionPreference = 'Stop'",
        f"$root = '{str(SCRIPTS).replace(chr(39), chr(39) * 2)}'",
        "$errorsFound = @()",
        "Get-ChildItem -LiteralPath $root -File | Where-Object { $_.Extension -in '.ps1','.psm1' } | ForEach-Object {",
        "  $tokens = $null; $parseErrors = $null",
        "  $null = [System.Management.Automation.Language.Parser]::ParseFile($_.FullName, [ref]$tokens, [ref]$parseErrors)",
        "  $errorsFound += @($parseErrors | ForEach-Object { $_.Message })",
        "}",
        "@{errors = @($errorsFound)} | ConvertTo-Json -Compress",
    ])
    assert _run_powershell(command)['errors'] == []


@requires_powershell
@pytest.mark.parametrize('safety', [
    "@{provable=$true;active_tasks=0}",
    "@{provable=$true;active_tasks=1}",
    "@{provable=$false;active_tasks=0}",
    "@{provable=$true}",
    "@{provable='true';active_tasks=0}",
])
def test_shutdown_requires_provably_idle_numeric_task_state(safety):
    command = "\n".join([
        "$ErrorActionPreference = 'Stop'",
        f"Import-Module '{RUNTIME}' -Force",
        f"$health = [pscustomobject]@{{shutdown_safety = {safety}; execution_queue = @{{gpu_busy=$false}}}}",
        "$accepted=$true; try { Assert-VisionCortexShutdownSafe -Health $health } catch { $accepted=$false }",
        "@{accepted=$accepted} | ConvertTo-Json -Compress",
    ])
    assert _run_powershell(command)['accepted'] is (safety == '@{provable=$true;active_tasks=0}')


@requires_powershell
def test_process_and_health_mocks_reject_recycled_or_foreign_instances():
    command = "\n".join([
        "$ErrorActionPreference = 'Stop'",
        f"Import-Module '{RUNTIME}' -Force",
        "& (Get-Module Runtime-Config) {",
        "  $root = [IO.Path]::GetTempPath()",
        "  $cfg = [pscustomobject]@{project_root=$root; python=(Join-Path $root 'python.exe'); config_path=(Join-Path $root 'site.yaml'); default_config_path=(Join-Path $root 'default.yaml'); archive_root=(Join-Path $root 'archive'); settings_sha256='fixture-sha'}",
        "  $script:fakeProcess = [pscustomobject]@{ExecutablePath=$cfg.python; CreationDate=[datetime]'2030-01-01T00:00:00Z'; CommandLine=('-m visioncortex serve --config "' + $cfg.config_path + '"')}",
        "  $owner=[pscustomobject]@{schema_version='visioncortex-web-owner/1';pid=123;port=8000;project_root=$root;python=$cfg.python;config_path=$cfg.config_path;archive_root=$cfg.archive_root;settings_sha256=$cfg.settings_sha256;created_utc=$script:fakeProcess.CreationDate.ToUniversalTime().ToString('o')}",
        "  function script:Get-CimInstance { param($ClassName,$Filter,$ErrorAction) return $script:fakeProcess }",
        "  function script:Get-NetTCPConnection { param($LocalPort,$State,$ErrorAction) return [pscustomobject]@{OwningProcess=123} }",
        "  $null = Assert-VisionCortexProcessOwnership -Configuration $cfg -Owner $owner -Port 8000",
        "  $owner.created_utc='2030-01-02T00:00:00Z'",
        "  $recycledRejected=$false; try { $null=Assert-VisionCortexProcessOwnership -Configuration $cfg -Owner $owner -Port 8000 } catch { $recycledRejected=$true }",
        "  $health=[pscustomobject]@{status='ok';archive_root=$cfg.archive_root}",
        "  $automation=[pscustomobject]@{pid=123;configuration=@{config_path=$cfg.config_path;default_config_path=$cfg.default_config_path;settings_sha256=$cfg.settings_sha256}}",
        "  Assert-VisionCortexHealthOwnership -Configuration $cfg -Owner $owner -Health $health -Automation $automation",
        "  $health.archive_root=Join-Path $root 'other-archive'",
        "  $foreignRejected=$false; try { Assert-VisionCortexHealthOwnership -Configuration $cfg -Owner $owner -Health $health -Automation $automation } catch { $foreignRejected=$true }",
        "  @{recycled_rejected=$recycledRejected; foreign_archive_rejected=$foreignRejected} | ConvertTo-Json -Compress",
        "}",
    ])
    assert _run_powershell(command) == {'recycled_rejected': True, 'foreign_archive_rejected': True}
