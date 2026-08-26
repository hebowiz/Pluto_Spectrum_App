@echo off
setlocal

set "PROJECT_DIR=%~dp0"
set "PYTHON_EXE=%PROJECT_DIR%.venv\Scripts\python.exe"

if not exist "%PYTHON_EXE%" (
    echo Virtual environment Python was not found:
    echo   %PYTHON_EXE%
    echo.
    echo Create it first with:
    echo   python -m venv .venv
    pause
    exit /b 1
)

pushd "%PROJECT_DIR%"

"%PYTHON_EXE%" -m pluto_vsg
set "EXIT_CODE=%ERRORLEVEL%"

popd

if not "%EXIT_CODE%"=="0" (
    echo.
    echo Pluto VSG exited with code %EXIT_CODE%.
    pause
)

exit /b %EXIT_CODE%
