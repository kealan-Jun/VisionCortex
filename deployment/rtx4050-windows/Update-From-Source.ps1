param([string]$PackageRoot)
$ErrorActionPreference = 'Stop'
try {
    if (-not $PackageRoot) {
        Add-Type -AssemblyName System.Windows.Forms
        $picker = New-Object System.Windows.Forms.FolderBrowserDialog
        $picker.Description = 'Select the installed folder containing VisionCortex.exe'
        try {
            if ($picker.ShowDialog() -ne [System.Windows.Forms.DialogResult]::OK) { exit 0 }
            $PackageRoot = $picker.SelectedPath
        } finally { $picker.Dispose() }
    }
    $root = (Resolve-Path -LiteralPath $PackageRoot).Path.TrimEnd('\')
    if (-not (Test-Path -LiteralPath (Join-Path $root 'VisionCortex.exe'))) { throw 'Select the installed application folder.' }
    $active = @(Get-CimInstance Win32_Process -ErrorAction Stop | Where-Object {
        $_.ExecutablePath -and $_.ExecutablePath.StartsWith($root + '\', [StringComparison]::OrdinalIgnoreCase)
    })
    if ($active.Count -gt 0) { throw 'Close VisionCortex and wait for its analysis processes to exit before updating.' }
    $manifest = Get-Content -LiteralPath (Join-Path $root 'SHA256SUMS.json') -Raw -Encoding UTF8 | ConvertFrom-Json
    $pythonEntry = @($manifest.files | Where-Object { $_.path -eq 'python/python.exe' })
    $python = Join-Path $root 'python\python.exe'
    if ($pythonEntry.Count -ne 1 -or (Get-FileHash -LiteralPath $python -Algorithm SHA256).Hash.ToLowerInvariant() -ne $pythonEntry[0].sha256) {
        throw 'Bundled Python checksum mismatch. No files changed.'
    }
    $source = (Resolve-Path -LiteralPath (Join-Path $PSScriptRoot '../..')).Path
    & $python -I -S -B (Join-Path $source 'tools/update_rtx4050_source.py') --root $root --source $source
    if ($LASTEXITCODE -ne 0) { throw 'Source update was not completed. Keep the diagnostic output.' }
    Write-Host 'Source update applied. Start VisionCortex.exe from the installed application folder.'
} catch {
    Write-Host ('ERROR: ' + $_.Exception.Message)
    exit 1
}
