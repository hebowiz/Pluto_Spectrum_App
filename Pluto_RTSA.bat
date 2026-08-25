@echo off
setlocal
cd /d "%~dp0"

set "APP_PYTHON=%~dp0.venv\Scripts\python.exe"
if not exist "%APP_PYTHON%" goto missing_venv

"%APP_PYTHON%" -m pluto_sa.main
set "APP_EXIT_CODE=%ERRORLEVEL%"
if not "%APP_EXIT_CODE%"=="0" goto launch_failed
exit /b 0

:missing_venv
echo [Pluto RTSA] Python virtual environment was not found:
echo   %APP_PYTHON%
echo Install the project dependencies into .venv before launching.
pause
exit /b 1

:launch_failed
echo.
echo [Pluto RTSA] Application exited with code %APP_EXIT_CODE%.
pause
exit /b %APP_EXIT_CODE%
