param([switch]$CheckOnly, [switch]$NoBrowser, [switch]$ResetApiKey, [ValidateRange(1,65535)][int]$Port = 8000)
$ErrorActionPreference = 'Stop'
[Console]::OutputEncoding = New-Object System.Text.UTF8Encoding($false)
$env:PYTHONIOENCODING = 'utf-8'
$env:PYTHONUTF8 = '1'
$env:PYTHONDONTWRITEBYTECODE = '1'
$Root = $PSScriptRoot
$Python = Join-Path $Root 'python\python.exe'
if (-not (Test-Path -LiteralPath $Python -PathType Leaf)) {
    throw 'This launcher must run from the fully extracted RTX 4050 package.'
}
Set-Location -LiteralPath $Root

# Hold a package-specific mutex throughout setup and the foreground server.
$Hasher = [System.Security.Cryptography.SHA256]::Create()
$Identity = [BitConverter]::ToString($Hasher.ComputeHash([Text.Encoding]::UTF8.GetBytes($Root.ToLowerInvariant()))).Replace('-', '')
$Hasher.Dispose()
$Mutex = New-Object System.Threading.Mutex($false, "Local\VisionCortex-$Identity")
$Locked = $false
$SavedApiKey = $env:ARK_API_KEY
try {
    try { $Locked = $Mutex.WaitOne(0) } catch [System.Threading.AbandonedMutexException] { $Locked = $true }
    if (-not $Locked) { throw 'This package is already running or preparing. Use its existing window.' }

    # The offline redistributable is used only when the system runtime is missing.
    $SavedErrorPreference = $ErrorActionPreference
    try {
        $ErrorActionPreference = 'Continue'
        & $Python -I -c "import ctypes; ctypes.WinDLL('msvcp140.dll'); ctypes.WinDLL('vcruntime140_1.dll')" 2>$null
        $RuntimeProbeExit = $LASTEXITCODE
    } finally { $ErrorActionPreference = $SavedErrorPreference }
    if ($RuntimeProbeExit -ne 0) {
        if ($CheckOnly) { throw 'Microsoft VC++ x64 runtime is missing. Start normally to install the bundled runtime offline.' }
        $Installer = Join-Path $Root 'vendor\vc_redist.x64.exe'
        $Sums = Get-Content -LiteralPath (Join-Path $Root 'SHA256SUMS.json') -Raw | ConvertFrom-Json
        $Entry = @($Sums.files | Where-Object { $_.path -eq 'vendor/vc_redist.x64.exe' })
        if ($Entry.Count -ne 1 -or (Get-FileHash -LiteralPath $Installer -Algorithm SHA256).Hash -ne $Entry[0].sha256) {
            throw 'Bundled Microsoft runtime checksum failed.'
        }
        $Signature = Get-AuthenticodeSignature -LiteralPath $Installer
        if ($Signature.Status -ne 'Valid' -or $Signature.SignerCertificate.Subject -notmatch 'Microsoft Corporation') {
            throw 'Bundled Microsoft runtime signature verification failed.'
        }
        Write-Host 'Installing the bundled Microsoft VC++ runtime offline; Windows may ask for administrator permission.'
        $Install = Start-Process -FilePath $Installer -ArgumentList '/install', '/passive', '/norestart' -Wait -PassThru
        if ($Install.ExitCode -eq 3010) { throw 'Microsoft runtime installed. Restart Windows, then double-click Start-VisionCortex.bat again.' }
        if ($Install.ExitCode -ne 0 -and $Install.ExitCode -ne 1638) { throw "Microsoft runtime installation failed: $($Install.ExitCode)" }
    }

    if (-not $CheckOnly) {
        # DPAPI binds this secret to the current Windows user and computer.
        # Keep it outside the transportable package; never write a plaintext key.
        $SecretRoot = Join-Path $env:LOCALAPPDATA 'VisionCortex\Secrets'
        $SecretFile = Join-Path $SecretRoot 'ark-api-key.dpapi'
        if ([string]::IsNullOrWhiteSpace($env:ARK_API_KEY) -or $ResetApiKey) {
            if ((Test-Path -LiteralPath $SecretFile) -and -not $ResetApiKey) {
                $SecureKey = Get-Content -LiteralPath $SecretFile -Raw | ConvertTo-SecureString
            } else {
                $SecureKey = Read-Host 'Enter your ARK_API_KEY (hidden; used for online video analysis)' -AsSecureString
                if ($SecureKey.Length -eq 0) { throw 'An ARK_API_KEY is required for full analysis.' }
                New-Item -ItemType Directory -Path $SecretRoot -Force | Out-Null
                $SecureKey | ConvertFrom-SecureString | Set-Content -LiteralPath $SecretFile -Encoding ascii
            }
            $Pointer = [Runtime.InteropServices.Marshal]::SecureStringToBSTR($SecureKey)
            try { $env:ARK_API_KEY = [Runtime.InteropServices.Marshal]::PtrToStringBSTR($Pointer) }
            finally { [Runtime.InteropServices.Marshal]::ZeroFreeBSTR($Pointer); $SecureKey.Dispose() }
        }
    }
    $Arguments = @('-B', (Join-Path $Root 'tools\rtx4050_portable.py'), '--port', "$Port")
    if ($CheckOnly) { $Arguments += '--check-only' }
    if ($NoBrowser) { $Arguments += '--no-browser' }
    & $Python @Arguments
    if ($LASTEXITCODE -ne 0) { throw 'VisionCortex did not start. Keep the error text and Runtime logs for diagnosis.' }
}
finally {
    $env:ARK_API_KEY = $SavedApiKey
    if ($Locked) { $Mutex.ReleaseMutex() }
    $Mutex.Dispose()
}
