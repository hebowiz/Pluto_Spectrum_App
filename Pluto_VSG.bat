@echo off
setlocal

set "PROJECT_DIR=%~dp0"
set "VENV_ACTIVATE=%PROJECT_DIR%.venv\Scripts\activate.bat"

if not exist "%VENV_ACTIVATE%" (
    echo Virtual environment was not found:
    echo   %VENV_ACTIVATE%
    echo.
    echo Create it first with:
    echo   python -m venv .venv
    pause
    exit /b 1
)

pushd "%PROJECT_DIR%"

call "%VENV_ACTIVATE%"

python -m pluto_vsg
set "EXIT_CODE=%ERRORLEVEL%"

popd

if not "%EXIT_CODE%"=="0" (
    echo.
    echo Pluto VSG exited with code %EXIT_CODE%.
    pause
)

exit /b %EXIT_CODE%