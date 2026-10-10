function Get-VisionCortexRuntimeConfiguration {
    param(
        [Parameter(Mandatory = $true)][string]$ProjectRoot,
        [Parameter(Mandatory = $true)][string]$ConfigPath,
        [string]$PythonPath = ''
    )
    $project = (Resolve-Path -LiteralPath $ProjectRoot).Path
    $python = if ([string]::IsNullOrWhiteSpace($PythonPath)) {
        Join-Path $project '.venv\Scripts\python.exe'
    } else { $PythonPath }
    $config = if ([IO.Path]::IsPathRooted($ConfigPath)) { $ConfigPath } else { Join-Path $project $ConfigPath }
    foreach ($required in @($python, $config)) {
        if (-not (Test-Path -LiteralPath $required -PathType Leaf)) {
            throw "Required local runtime file is missing: $required"
        }
    }
    $code = @'
import json, sys
from pathlib import Path
import visioncortex
from visioncortex.config import load_config, _resolve_default_config
from visioncortex.automation_readiness import configuration_digest
from visioncortex.benchmark_registration import registration
from visioncortex.model_registry import load_model_registry
root = Path(sys.argv[1]).resolve()
profile = Path(sys.argv[2]).resolve()
if not Path(visioncortex.__file__).resolve().is_relative_to(root / 'src'):
    raise SystemExit('The interpreter does not load this checkout source.')
cfg = load_config(profile)
project = cfg.get('project') or {}
if project.get('run_purpose') != 'production' or project.get('site_configuration_required'):
    raise SystemExit('An explicit configured private production site is required.')
def full(value):
    p = Path(value).expanduser()
    return str((root / p).resolve() if not p.is_absolute() else p.resolve())
storage, models = cfg['storage'], cfg['models']
hashes = dict(models.get('sha256_by_role') or {})
registry = models.get('registry_path')
if registry:
    payload = load_model_registry(Path(full(registry)))
    for role in ('first_person', 'third_person'):
        entry = payload['models'][role]
        if full(entry['path']) != full(models[role]):
            raise SystemExit('The private model registry path does not match the selected model.')
        hashes[role] = entry['sha256']
print(json.dumps({
    'project_root':str(root), 'python':str(Path(sys.executable).resolve()),
    'config_path':str(profile), 'default_config_path':str(_resolve_default_config(profile)),
    'settings_sha256':configuration_digest(cfg),
    'runtime_root':full(storage['local_runtime_root']),
    'cache_root':full(storage['local_cache_root']),
    'archive_root':full(storage['archive_root']), 'index_csv':full(storage['index_csv']),
    'models':{role:full(models[role]) for role in ('first_person','third_person')},
    'model_hashes':hashes, 'fixed_benchmark':registration(cfg),
    'mllm_enabled':bool(cfg.get('mllm',{}).get('enabled')),
}, ensure_ascii=True))
'@
    $oldPythonPath = $env:PYTHONPATH
    Push-Location -LiteralPath $project
    try {
        $env:PYTHONPATH = Join-Path $project 'src'
        $result = & $python -B -c $code $project $config
        if ($LASTEXITCODE -ne 0) { throw 'Private runtime configuration validation failed.' }
        $value = ($result -join "`n") | ConvertFrom-Json
        if ($value.runtime_root.StartsWith('\\')) { throw 'Runtime state must be on a local filesystem.' }
        $runtimeDrive = [IO.DriveInfo]::new([IO.Path]::GetPathRoot($value.runtime_root))
        if ($runtimeDrive.DriveType -in @([IO.DriveType]::Network, [IO.DriveType]::Unknown, [IO.DriveType]::NoRootDirectory)) {
            throw 'Runtime state must use a verified local drive, including when archive drives are mapped.'
        }
        return $value
    }
    finally {
        Pop-Location
        if ($null -eq $oldPythonPath) { Remove-Item Env:PYTHONPATH -ErrorAction SilentlyContinue }
        else { $env:PYTHONPATH = $oldPythonPath }
    }
}

function Test-VisionCortexSamePath {
    param([string]$First, [string]$Second)
    if ([string]::IsNullOrWhiteSpace($First) -or [string]::IsNullOrWhiteSpace($Second)) { return $false }
    return [IO.Path]::GetFullPath($First).TrimEnd('\', '/').Equals(
        [IO.Path]::GetFullPath($Second).TrimEnd('\', '/'), [StringComparison]::OrdinalIgnoreCase)
}

function Assert-VisionCortexProcessOwnership {
    param(
        [Parameter(Mandatory = $true)]$Configuration,
        [Parameter(Mandatory = $true)]$Owner,
        [Parameter(Mandatory = $true)][int]$Port
    )
    if ($Owner.schema_version -ne 'visioncortex-web-owner/1' -or $Owner.port -ne $Port -or
        -not (Test-VisionCortexSamePath $Owner.project_root $Configuration.project_root) -or
        -not (Test-VisionCortexSamePath $Owner.python $Configuration.python) -or
        -not (Test-VisionCortexSamePath $Owner.config_path $Configuration.config_path) -or
        -not (Test-VisionCortexSamePath $Owner.archive_root $Configuration.archive_root) -or
        $Owner.settings_sha256 -ne $Configuration.settings_sha256) {
        throw 'The recorded service belongs to another checkout or configuration.'
    }
    $ownerProcessId = 0
    if (-not [int]::TryParse([string]$Owner.pid, [ref]$ownerProcessId) -or $ownerProcessId -le 0) {
        throw 'The owner receipt does not contain a valid process identity.'
    }
    $process = Get-CimInstance Win32_Process -Filter "ProcessId = $ownerProcessId" -ErrorAction Stop
    if ($null -eq $process -or
        -not (Test-VisionCortexSamePath $process.ExecutablePath $Configuration.python) -or
        $process.CreationDate.ToUniversalTime().ToString('o') -ne $Owner.created_utc) {
        throw 'The service process identity changed; refusing to claim a recycled PID.'
    }
    $configToken = '(?i)(?:^|\s)--config\s+(?:"' + [regex]::Escape($Configuration.config_path) + '"|' +
        [regex]::Escape($Configuration.config_path) + ')(?:\s|$)'
    if ($process.CommandLine -notmatch '(?i)(?:^|\s)-m\s+visioncortex\s+serve(?:\s|$)' -or
        $process.CommandLine -notmatch $configToken) {
        throw 'The process command does not match this service configuration.'
    }
    $listeners = @(Get-NetTCPConnection -LocalPort $Port -State Listen -ErrorAction Stop)
    if ($listeners.Count -eq 0 -or @($listeners | Where-Object { $_.OwningProcess -ne $Owner.pid }).Count -gt 0) {
        throw 'The listening socket is not owned by the recorded service.'
    }
    return $process
}

function Assert-VisionCortexHealthOwnership {
    param(
        [Parameter(Mandatory = $true)]$Configuration,
        [Parameter(Mandatory = $true)]$Owner,
        [Parameter(Mandatory = $true)]$Health,
        [Parameter(Mandatory = $true)]$Automation
    )
    if ($Health.status -ne 'ok' -or $Automation.pid -ne $Owner.pid -or
        -not (Test-VisionCortexSamePath $Health.archive_root $Configuration.archive_root) -or
        -not (Test-VisionCortexSamePath $Automation.configuration.config_path $Configuration.config_path) -or
        -not (Test-VisionCortexSamePath $Automation.configuration.default_config_path $Configuration.default_config_path) -or
        $Automation.configuration.settings_sha256 -ne $Configuration.settings_sha256) {
        throw 'The service health belongs to another configuration or archive root.'
    }
}

function Assert-VisionCortexShutdownSafe {
    param([Parameter(Mandatory = $true)]$Health)
    $safety = $Health.shutdown_safety
    if ($null -eq $safety -or $safety.provable -isnot [bool] -or $safety.provable -ne $true -or
        ($safety.active_tasks -isnot [int] -and $safety.active_tasks -isnot [long]) -or
        $safety.active_tasks -ne 0 -or $Health.execution_queue.gpu_busy -eq $true) {
        throw 'Active task state is not provably idle; refusing to stop the service.'
    }
}

Export-ModuleMember -Function Get-VisionCortexRuntimeConfiguration, Test-VisionCortexSamePath, Assert-VisionCortexProcessOwnership, Assert-VisionCortexHealthOwnership, Assert-VisionCortexShutdownSafe
