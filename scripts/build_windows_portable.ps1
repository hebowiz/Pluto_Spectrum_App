param(
    [string]$Version = "dev",
    [switch]$SkipInstall
)

$ErrorActionPreference = "Stop"
$repoRoot = Split-Path -Parent $PSScriptRoot
$python = Join-Path $repoRoot ".venv\Scripts\python.exe"
if (-not (Test-Path -LiteralPath $python)) {
    throw "Virtual environment not found: $python"
}

if (-not $SkipInstall) {
    & $python -m pip install -r (Join-Path $repoRoot "requirements.txt")
    if ($LASTEXITCODE -ne 0) { throw "Runtime dependency installation failed" }
    & $python -m pip install -r (Join-Path $repoRoot "requirements-build.txt")
    if ($LASTEXITCODE -ne 0) { throw "Build dependency installation failed" }
}

$cacheRoot = Join-Path $repoRoot ".build-cache\libiio-v0.25"
$archivePath = Join-Path $cacheRoot "libiio-0.25-windows.zip"
$expandedPath = Join-Path $cacheRoot "expanded"
$runtimePath = Join-Path $cacheRoot "runtime"
New-Item -ItemType Directory -Force -Path $cacheRoot | Out-Null
if (-not (Test-Path -LiteralPath $archivePath)) {
    Invoke-WebRequest -TimeoutSec 120 `
        -Uri "https://github.com/analogdevicesinc/libiio/releases/download/v0.25/libiio-0.25-gb6028fd-windows.zip" `
        -OutFile $archivePath
}
if (-not (Test-Path -LiteralPath (Join-Path $expandedPath "Windows-VS-2022-x64\libiio.dll"))) {
    if (Test-Path -LiteralPath $expandedPath) {
        Remove-Item -LiteralPath $expandedPath -Recurse -Force
    }
    Expand-Archive -LiteralPath $archivePath -DestinationPath $expandedPath
}

New-Item -ItemType Directory -Force -Path $runtimePath | Out-Null
$officialRuntime = Join-Path $expandedPath "Windows-VS-2022-x64"
$runtimeFiles = @(
    "libiio.dll",
    "libusb-1.0.dll",
    "libxml2.dll",
    "libserialport-0.dll",
    "msvcp140.dll",
    "vcruntime140.dll"
)
foreach ($name in $runtimeFiles) {
    Copy-Item -LiteralPath (Join-Path $officialRuntime $name) -Destination $runtimePath -Force
}

$licensePath = Join-Path $cacheRoot "libiio-COPYING.txt"
if (-not (Test-Path -LiteralPath $licensePath)) {
    Invoke-WebRequest -TimeoutSec 60 `
        -Uri "https://raw.githubusercontent.com/analogdevicesinc/libiio/v0.25/COPYING.txt" `
        -OutFile $licensePath
}

$distRoot = Join-Path $repoRoot "dist"
$buildRoot = Join-Path $repoRoot "build"
$releaseRoot = Join-Path $repoRoot "release"
$specRoot = Join-Path $buildRoot "spec"
foreach ($path in @($distRoot, $buildRoot, $releaseRoot)) {
    if (Test-Path -LiteralPath $path) {
        Remove-Item -LiteralPath $path -Recurse -Force
    }
    New-Item -ItemType Directory -Force -Path $path | Out-Null
}
New-Item -ItemType Directory -Force -Path $specRoot | Out-Null

$runtimeHook = Join-Path $repoRoot "packaging\hooks\runtime_hook_libiio.py"
$notices = Join-Path $repoRoot "packaging\THIRD_PARTY_NOTICES.txt"
$applications = @(
    @{ Name = "Pluto_RTSA"; Entry = "packaging\entrypoints\pluto_rtsa.py"; Hidden = @() },
    @{ Name = "Pluto_VSA"; Entry = "packaging\entrypoints\pluto_vsa.py"; Hidden = @(
        "PySide6.QtWebEngineCore", "PySide6.QtWebEngineWidgets", "PySide6.QtWebChannel"
    ) },
    @{ Name = "Pluto_VSG"; Entry = "packaging\entrypoints\pluto_vsg_entry.py"; Hidden = @() }
)

foreach ($application in $applications) {
    $arguments = @(
        "-m", "PyInstaller",
        "--noconfirm", "--clean", "--onedir", "--windowed",
        "--name", $application.Name,
        "--paths", $repoRoot,
        "--distpath", $distRoot,
        "--workpath", (Join-Path $buildRoot $application.Name),
        "--specpath", $specRoot,
        "--runtime-hook", $runtimeHook,
        "--add-data", "$notices;licenses",
        "--add-data", "$licensePath;licenses"
    )
    foreach ($name in $runtimeFiles) {
        $arguments += @("--add-binary", "$(Join-Path $runtimePath $name);libiio")
    }
    foreach ($module in $application.Hidden) {
        $arguments += @("--hidden-import", $module)
    }
    $arguments += (Join-Path $repoRoot $application.Entry)
    & $python @arguments
    if ($LASTEXITCODE -ne 0) { throw "PyInstaller failed for $($application.Name)" }

    $executable = Join-Path $distRoot "$($application.Name)\$($application.Name).exe"
    $smokeReport = Join-Path $buildRoot "$($application.Name)-smoke.json"
    Remove-Item -LiteralPath $smokeReport -Force -ErrorAction SilentlyContinue
    $env:PLUTO_APP_SMOKE_TEST = "1"
    $env:PLUTO_APP_SMOKE_REPORT = $smokeReport
    try {
        $process = Start-Process -FilePath $executable -Wait -PassThru -WindowStyle Hidden
        if ($process.ExitCode -ne 0) { throw "Frozen smoke test failed for $($application.Name)" }
        if (-not (Test-Path -LiteralPath $smokeReport)) {
            throw "Frozen smoke test produced no report for $($application.Name)"
        }
        $report = Get-Content -LiteralPath $smokeReport -Raw | ConvertFrom-Json
        if ($report.application -ne $application.Name -or $report.libiio_version[1] -ne 25) {
            throw "Frozen smoke report is invalid for $($application.Name): $($report | ConvertTo-Json -Compress)"
        }
    } finally {
        Remove-Item Env:PLUTO_APP_SMOKE_TEST -ErrorAction SilentlyContinue
        Remove-Item Env:PLUTO_APP_SMOKE_REPORT -ErrorAction SilentlyContinue
    }

    $archive = Join-Path $releaseRoot "$($application.Name)-$Version-windows-x64.zip"
    Add-Type -AssemblyName System.IO.Compression.FileSystem
    [System.IO.Compression.ZipFile]::CreateFromDirectory(
        (Join-Path $distRoot $application.Name),
        $archive,
        [System.IO.Compression.CompressionLevel]::Optimal,
        $true
    )
    if (-not (Test-Path -LiteralPath $archive) -or (Get-Item -LiteralPath $archive).Length -eq 0) {
        throw "Portable archive was not created for $($application.Name)"
    }
}

$checksumPath = Join-Path $releaseRoot "SHA256SUMS.txt"
$checksumLines = Get-ChildItem -LiteralPath $releaseRoot -Filter "*.zip" |
    Sort-Object Name |
    ForEach-Object {
        $hash = (Get-FileHash -LiteralPath $_.FullName -Algorithm SHA256).Hash.ToLowerInvariant()
        "$hash  $($_.Name)"
    }
Set-Content -LiteralPath $checksumPath -Value $checksumLines -Encoding ascii
if (-not $checksumLines -or $checksumLines.Count -ne $applications.Count) {
    throw "Expected $($applications.Count) archives but generated $($checksumLines.Count)"
}
Write-Host "Portable packages created in $releaseRoot"
