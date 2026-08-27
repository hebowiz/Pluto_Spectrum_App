@echo off
setlocal

set "PROJECT_DIR=%~dp0"
set "PYTHON_EXE=%PROJECT_DIR%.venv\Scripts\python.exe"

call "%PROJECT_DIR%_Pluto_Runtime.bat"
set "RUNTIME_EXIT=%ERRORLEVEL%"
if not "%RUNTIME_EXIT%"=="0" (
    echo.
    echo Pluto runtime setup failed with code %RUNTIME_EXIT%.
    pause
    exit /b %RUNTIME_EXIT%
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
