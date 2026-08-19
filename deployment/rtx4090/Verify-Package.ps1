param(
    [string]$PackageRoot = (Resolve-Path -LiteralPath (Join-Path $PSScriptRoot '..\..')).Path
)

$ErrorActionPreference = 'Stop'
$Root = (Resolve-Path -LiteralPath $PackageRoot).Path
$ManifestPath = Join-Path $Root 'SHA256SUMS.json'
if (-not (Test-Path -LiteralPath $ManifestPath -PathType Leaf)) {
    throw "Package manifest is missing: $ManifestPath"
}

$Manifest = Get-Content -LiteralPath $ManifestPath -Raw -Encoding UTF8 | ConvertFrom-Json
if ($Manifest.schema_version -ne 1) { throw 'Unsupported package manifest schema.' }
if ([string]::IsNullOrWhiteSpace($Manifest.source_commit)) { throw 'Source commit is missing.' }

$Verified = 0
foreach ($Entry in $Manifest.files) {
    $Relative = [string]$Entry.path
    if ([string]::IsNullOrWhiteSpace($Relative) -or [System.IO.Path]::IsPathRooted($Relative)) {
        throw "Unsafe manifest path: $Relative"
    }
    $FullPath = [System.IO.Path]::GetFullPath((Join-Path $Root $Relative))
    $Boundary = $Root.TrimEnd('\') + '\'
    if (-not $FullPath.StartsWith($Boundary, [StringComparison]::OrdinalIgnoreCase)) {
        throw "Manifest path escapes the package root: $Relative"
    }
    if (-not (Test-Path -LiteralPath $FullPath -PathType Leaf)) {
        throw "Package file is missing: $Relative"
    }
    $File = Get-Item -LiteralPath $FullPath
    if ($File.Length -ne [long]$Entry.size_bytes) {
        throw "Package file size mismatch: $Relative"
    }
    $Actual = (Get-FileHash -LiteralPath $FullPath -Algorithm SHA256).Hash.ToLowerInvariant()
    if ($Actual -ne ([string]$Entry.sha256).ToLowerInvariant()) {
        throw "Package checksum mismatch: $Relative"
    }
    $Verified++
}

Write-Host "Package verification passed: $Verified files; source $($Manifest.source_commit)."

