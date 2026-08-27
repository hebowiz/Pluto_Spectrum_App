@echo off
setlocal EnableExtensions EnableDelayedExpansion

set "PROJECT_DIR=%~dp0"
set "PYTHON_EXE=%PROJECT_DIR%.venv\Scripts\python.exe"

if not exist "%PYTHON_EXE%" (
    echo [ERROR] Virtual environment Python was not found:
    echo   %PYTHON_EXE%
    echo.
    echo Create it first with:
    echo   python -m venv .venv
    exit /b 1
)

rem If iio already imports successfully, keep the current environment unchanged.
"%PYTHON_EXE%" -c "import iio" >nul 2>nul
if not errorlevel 1 exit /b 0

set "LIBIIO_DIR="

rem Prefer an optional project-local portable runtime when present.
if exist "%PROJECT_DIR%runtime\libiio\bin\libiio.dll" (
    set "LIBIIO_DIR=%PROJECT_DIR%runtime\libiio\bin"
)

rem Search PATH next.
if not defined LIBIIO_DIR (
    for /f "delims=" %%F in ('where libiio.dll 2^>nul') do (
        if not defined LIBIIO_DIR set "LIBIIO_DIR=%%~dpF"
    )
)

rem Search common Analog Devices installation roots. Use WHERE /R so the
rem launcher also works when the exact subdirectory differs between PCs.
if not defined LIBIIO_DIR if exist "%ProgramFiles%\Analog Devices" (
    for /f "delims=" %%F in ('where /r "%ProgramFiles%\Analog Devices" libiio.dll 2^>nul') do (
        if not defined LIBIIO_DIR set "LIBIIO_DIR=%%~dpF"
    )
)

if not defined LIBIIO_DIR if defined ProgramFiles(x86) if exist "%ProgramFiles(x86)%\Analog Devices" (
    for /f "delims=" %%F in ('where /r "%ProgramFiles(x86)%\Analog Devices" libiio.dll 2^>nul') do (
        if not defined LIBIIO_DIR set "LIBIIO_DIR=%%~dpF"
    )
)

if not defined LIBIIO_DIR if exist "%ProgramFiles%\libiio" (
    for /f "delims=" %%F in ('where /r "%ProgramFiles%\libiio" libiio.dll 2^>nul') do (
        if not defined LIBIIO_DIR set "LIBIIO_DIR=%%~dpF"
    )
)

if not defined LIBIIO_DIR (
    echo [ERROR] libiio.dll was not found.
    echo.
    echo The Python package pylibiio is installed, but its native Windows
    echo runtime is not available to this process.
    echo.
    echo Install a matching libiio runtime or place a portable copy under:
    echo   %PROJECT_DIR%runtime\libiio\bin
    echo.
    echo Expected Python binding: pylibiio 0.25
    exit /b 2
)

set "PATH=%LIBIIO_DIR%;%PATH%"

rem Verify that Python can now load the native runtime and its dependencies.
"%PYTHON_EXE%" -c "import iio; print('libiio runtime:', iio.version)"
if errorlevel 1 (
    echo.
    echo [ERROR] libiio.dll was found but could not be loaded.
    echo Runtime directory:
    echo   %LIBIIO_DIR%
    echo.
    echo Check DLL architecture and dependent DLLs such as libusb.
    exit /b 3
)

rem Export the adjusted PATH back to the caller before ENDLOCAL restores it.
endlocal & set "PATH=%PATH%" & exit /b 0
