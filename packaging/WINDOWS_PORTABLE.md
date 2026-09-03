# Windows portable build and GitHub Release

The Windows deliverables are four **one-folder portable ZIP files**:

- `Pluto_RTSA-<version>-windows-x64.zip`
- `Pluto_VSA-<version>-windows-x64.zip`
- `Pluto_VSG-<version>-windows-x64.zip`
- `Pluto_CAL-<version>-windows-x64.zip`

Each archive contains one application EXE, Python, Qt, numerical libraries,
and the official libiio v0.25 Windows runtime. The target PC does not need a
separate Python or libiio installation. A one-folder build is intentional:
Qt WebEngine and native SDR DLL loading are more reliable and start faster
than a self-extracting one-file executable.

## Local build

From a PowerShell prompt in the repository root:

```powershell
.\scripts\build_windows_portable.ps1 -Version dev
```

The script downloads the official libiio v0.25 archive into the ignored
`.build-cache` directory, builds all four applications, runs frozen smoke
tests, and creates ZIP files under `release`.

To rebuild without downloading libiio again, leave `.build-cache` in place.

## Release workflow

Push a version tag to build and publish a GitHub Release:

```powershell
git tag v0.1.0-alpha.1
git push origin v0.1.0-alpha.1
```

The `release-windows.yml` workflow builds on a clean Windows runner and
uploads the four portable archives plus SHA-256 checksums. It can also be
started manually to produce workflow artifacts without publishing a Release.

## Smoke-test mode

The packaging entry points support `PLUTO_APP_SMOKE_TEST=1`. In that mode each
frozen EXE imports its application, libiio, and (for VSA) Qt WebEngine, writes
a machine-readable report to `PLUTO_APP_SMOKE_REPORT`, and exits without
opening hardware or a GUI. The build fails unless every report confirms the
bundled libiio v0.25 runtime.
