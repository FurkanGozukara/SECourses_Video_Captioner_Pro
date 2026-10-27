@echo off
setlocal

set "BASE_DIR=%~dp0SECourses_Video_Captioner_Pro"
if not exist "%BASE_DIR%\venv\Scripts\activate.bat" (
    echo.
    echo ERROR: Virtual environment not found at:
    echo   %BASE_DIR%\venv
    echo Run Windows_Install_and_Update.bat first.
    echo.
    pause
    exit /b 1
)

if not exist "%BASE_DIR%\temp" mkdir "%BASE_DIR%\temp"
if not exist "%BASE_DIR%\models" mkdir "%BASE_DIR%\models"

set "TEMP=%BASE_DIR%\temp"
set "TMP=%BASE_DIR%\temp"
set "MODELS_DIR=%BASE_DIR%\models"
set "VCAP_MODELS_DIR=%BASE_DIR%\models"
set "HF_HOME=%BASE_DIR%\models"
set "TRANSFORMERS_CACHE=%BASE_DIR%\models"
set "PYTORCH_CUDA_ALLOC_CONF=garbage_collection_threshold:0.8"
set "PYTHONUTF8=1"
set "PYTHONIOENCODING=utf-8"
set "PYTHONUNBUFFERED=1"
set "HF_HUB_ENABLE_HF_TRANSFER=0"
set "CUDA_DEVICE_ORDER=PCI_BUS_ID"
set "CUDA_MODULE_LOADING=LAZY"

cd /d "%BASE_DIR%" || exit /b 1
call venv\Scripts\activate.bat || exit /b 1
python secourses_app.py %*
set "APP_EXIT=%ERRORLEVEL%"
pause
exit /b %APP_EXIT%
