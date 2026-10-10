param([string]$PackageRoot)
$ErrorActionPreference = 'Stop'
try {
    if (-not $PackageRoot) {
        Add-Type -AssemblyName System.Windows.Forms
        $picker = New-Object System.Windows.Forms.FolderBrowserDialog
        $picker.Description = 'Select the extracted VisionCortex folder containing VisionCortex.exe'
        if ($picker.ShowDialog() -ne [System.Windows.Forms.DialogResult]::OK) { exit 0 }
        $PackageRoot = $picker.SelectedPath
        $picker.Dispose()
    }
    $root = (Resolve-Path -LiteralPath $PackageRoot).Path.TrimEnd('\')
    if (-not (Test-Path -LiteralPath (Join-Path $root 'VisionCortex.exe'))) {
        throw 'Select the application folder containing VisionCortex.exe.'
    }
    $spec = Get-Content -LiteralPath (Join-Path $PSScriptRoot 'update.json') -Raw | ConvertFrom-Json
    $manifestHash = (Get-FileHash -LiteralPath (Join-Path $root 'SHA256SUMS.json') -Algorithm SHA256).Hash.ToLowerInvariant()
    if ($manifestHash -ne $spec.base_manifest_sha256 -and $manifestHash -ne $spec.updated_manifest_sha256) {
        throw 'Wrong application version. No files changed.'
    }
    # Query failure is an error; never kill a user's application or analysis task.
    $active = @(Get-CimInstance Win32_Process -ErrorAction Stop | Where-Object {
        $_.ExecutablePath -and $_.ExecutablePath.StartsWith($root + '\', [StringComparison]::OrdinalIgnoreCase)
    })
    if ($active.Count -gt 0) { throw 'Close VisionCortex and wait for its analysis processes to exit, then try again.' }
    $python = Join-Path $root 'python\python.exe'
    if ((Get-FileHash -LiteralPath $python -Algorithm SHA256).Hash.ToLowerInvariant() -ne $spec.python_sha256) {
        throw 'Bundled Python checksum mismatch. No files changed.'
    }
    & $python -I -S -B (Join-Path $PSScriptRoot 'apply-update.py') --root $root
    if ($LASTEXITCODE -ne 0) { throw 'Update was not completed. Keep the output above for diagnosis.' }
    Write-Host 'Update applied. Start VisionCortex.exe from the application folder.'
    Write-Host 'Existing data, engines and encrypted keys were preserved.'
} catch {
    Write-Host ('ERROR: ' + $_.Exception.Message)
    exit 1
}
