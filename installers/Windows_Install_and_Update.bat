@echo off
setlocal
cd /d "%~dp0" || goto :failed

echo WARNING. For this auto installer to work you need to have installed Python 3.12.10, Git, FFmpeg, cuDNN, CUDA, Visual C++ Tools
echo This tutorial shows how to install requirements (1-time only) with 1-clik installer scripts : https://youtu.be/ptuLL0C72Ms
echo.

set "APP_DIR=%CD%\SECourses_Video_Captioner_Pro"
set "VENV_DIR=%APP_DIR%\venv"
set "VENV_PYTHON=%VENV_DIR%\Scripts\python.exe"
set UV_SKIP_WHEEL_FILENAME_CHECK=1
set UV_LINK_MODE=copy

if exist "%APP_DIR%\.git" (
    echo Updating SECourses Video Captioner Pro...
    pushd "%APP_DIR%" || goto :failed
    git reset --hard && git pull
    if errorlevel 1 (
        popd
        goto :failed
    )
    popd
) else (
    if exist "%APP_DIR%" (
        echo ERROR: "%APP_DIR%" exists but is not a Git checkout.
        goto :failed
    )
    git clone https://github.com/FurkanGozukara/SECourses_Video_Captioner_Pro "%APP_DIR%"
    if errorlevel 1 goto :failed
)

if exist "%VENV_PYTHON%" (
    "%VENV_PYTHON%" -c "import sys; raise SystemExit(0 if sys.version_info[:2] == (3, 12) else 1)"
    if errorlevel 1 (
        echo Rebuilding the virtual environment with Python 3.12...
        rmdir /s /q "%VENV_DIR%"
    ) else (
        echo Reusing the existing Python 3.12 virtual environment.
    )
)

if not exist "%VENV_PYTHON%" (
    py -3.12 -c "import sys; raise SystemExit(0 if sys.version_info[:2] == (3, 12) else 1)" >nul 2>&1
    if errorlevel 1 (
        python -c "import sys; raise SystemExit(0 if sys.version_info[:2] == (3, 12) else 1)" >nul 2>&1
        if errorlevel 1 (
            echo ERROR: A 64-bit Python 3.12 installation was not found.
            goto :failed
        )
        python -m venv "%VENV_DIR%"
    ) else (
        py -3.12 -m venv "%VENV_DIR%"
    )
    if errorlevel 1 goto :failed
)

call "%VENV_DIR%\Scripts\activate.bat" || goto :failed
python -m pip install --upgrade pip || goto :failed
pip install uv || goto :failed

cd /d "%~dp0" || goto :failed
uv pip install -r video_caption_requirements.txt || goto :failed

pushd "%APP_DIR%" || goto :failed
python -m vcap.core.cuda_runtime --repair
if errorlevel 1 (
    popd
    goto :failed
)
popd

echo.
echo SECourses Video Captioner Pro installation/update completed.
echo Models download automatically on first use, or use Windows_Download_Models_and_or_Resume.bat.
pause
exit /b 0

:failed
echo.
echo ERROR: Installation or update failed. Review the messages above.
pause
exit /b 1
