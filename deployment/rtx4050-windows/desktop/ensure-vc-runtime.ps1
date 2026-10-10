param([Parameter(Mandatory=$true)][string]$PackageRoot)
$ErrorActionPreference = 'Stop'
$Installer = Join-Path $PackageRoot 'vendor\vc_redist.x64.exe'
$Manifest = Get-Content -LiteralPath (Join-Path $PackageRoot 'SHA256SUMS.json') -Raw | ConvertFrom-Json
$Entry = @($Manifest.files | Where-Object { $_.path -eq 'vendor/vc_redist.x64.exe' })
if ($Entry.Count -ne 1 -or (Get-FileHash -LiteralPath $Installer -Algorithm SHA256).Hash -ne $Entry[0].sha256) { exit 2 }
$Signature = Get-AuthenticodeSignature -LiteralPath $Installer
if ($Signature.Status -ne 'Valid' -or $Signature.SignerCertificate.Subject -notmatch 'Microsoft Corporation') { exit 3 }
$Process = Start-Process -FilePath $Installer -ArgumentList '/install', '/passive', '/norestart' -Verb RunAs -Wait -PassThru
exit $Process.ExitCode
