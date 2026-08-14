$ErrorActionPreference = 'Stop'
$ProjectRoot = (Resolve-Path -LiteralPath (Join-Path $PSScriptRoot '..\..')).Path
$Python = Join-Path $ProjectRoot '.venv\Scripts\python.exe'
$Config = Join-Path $ProjectRoot 'configs\rtx4060-laptop-production.yaml'

if (-not (Test-Path -LiteralPath $Python)) { throw 'Not installed. Run the setup script first.' }
if (-not (Test-Path -LiteralPath 'Y:\experiment_record_index.csv')) { throw 'Y: NAS or the index CSV is unavailable.' }
$UserKey = [Environment]::GetEnvironmentVariable('ARK_API_KEY', 'User')
if ([string]::IsNullOrWhiteSpace($env:ARK_API_KEY) -and -not [string]::IsNullOrWhiteSpace($UserKey)) { $env:ARK_API_KEY = $UserKey }
if ([string]::IsNullOrWhiteSpace($env:ARK_API_KEY)) { throw 'ARK_API_KEY is not configured.' }

$env:VISIONCORTEX_COARSE_DECODE_LANES = 'cuda,cuda,cuda,cuda,cpu,cpu'
$env:VISIONCORTEX_SOURCE_WORKERS = '6'
$env:VISIONCORTEX_DECODE_QUEUE_DEPTH = '32'
$env:VISIONCORTEX_CPU_DECODE_THREADS = '8'
$env:VISIONCORTEX_YOLO_INFERENCE_WORKERS = '2'
$env:VISIONCORTEX_MATERIALIZATION_WORKERS = '2'
$env:VISIONCORTEX_IO_WORKERS = '4'
$env:VISIONCORTEX_TENSORRT = 'required'

Set-Location -LiteralPath $ProjectRoot
& $Python -m labvision_evidence run-fixed-benchmark --config $Config
