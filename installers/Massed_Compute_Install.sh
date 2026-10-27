export UV_SKIP_WHEEL_FILENAME_CHECK=1
export UV_LINK_MODE=copy

git clone https://github.com/FurkanGozukara/SECourses_Video_Captioner_Pro

cd SECourses_Video_Captioner_Pro

git reset --hard

git pull

REQUIRED_PY_VERSION="3.12.13"
REQUIRED_PY_TUPLE="(3, 12, 13)"

VENV_DIR="venv"
VENV_PYTHON="$VENV_DIR/bin/python"
VENV_READY=0
CAN_INSTALL_SYSTEM_PACKAGES=1
SUDO=()

python_is_current() {
    [ -n "$1" ] || return 1
    "$1" -c "import sys; raise SystemExit(0 if sys.version_info[:2] == (3, 12) and sys.version_info.releaselevel == 'final' and sys.version_info[:3] >= $REQUIRED_PY_TUPLE else 1)" 2>/dev/null
}

if [ "$(id -u)" -ne 0 ]; then
    if command -v sudo >/dev/null 2>&1; then
        SUDO=(sudo)
    else
        CAN_INSTALL_SYSTEM_PACKAGES=0
    fi
fi

echo "Ensuring Git, xz, CMake, build tools, and libcurl are installed..."
if ! command -v git >/dev/null 2>&1 \
    || ! command -v xz >/dev/null 2>&1 \
    || ! command -v cmake >/dev/null 2>&1 \
    || ! command -v c++ >/dev/null 2>&1 \
    || ! command -v dpkg-query >/dev/null 2>&1 \
    || ! dpkg-query -W -f='${Status}' libcurl4-openssl-dev 2>/dev/null | grep -q "install ok installed"; then
    if [ "$CAN_INSTALL_SYSTEM_PACKAGES" -ne 1 ]; then
        echo "Root access or sudo is required to install Git, FFmpeg, and llama.cpp build tools. Exiting..."
        exit 1
    fi

    if ! command -v apt-get >/dev/null 2>&1; then
        echo "apt-get is required to install Git, FFmpeg, and llama.cpp build tools. Exiting..."
        exit 1
    fi

    if ! "${SUDO[@]}" apt-get update \
        || ! "${SUDO[@]}" apt-get install -y xz-utils git cmake build-essential libcurl4-openssl-dev; then
        echo "Failed to install Git, FFmpeg, and llama.cpp build tools. Exiting..."
        exit 1
    fi
fi

FFMPEG_URL="https://huggingface.co/MonsterMMORPG/Wan_GGUF/resolve/main/ffmpeg-n9.0-latest-linux64-gpl-9.0.tar.xz"
echo "Installing FFmpeg n9.0 as the system default..."
FFMPEG_TMP="$(mktemp -d)"
if ! curl -fL -o "$FFMPEG_TMP/ffmpeg.tar.xz" "$FFMPEG_URL" \
    || ! tar -xJf "$FFMPEG_TMP/ffmpeg.tar.xz" -C "$FFMPEG_TMP" \
    || ! "${SUDO[@]}" install -m 755 "$FFMPEG_TMP"/ffmpeg-*/bin/ffmpeg "$FFMPEG_TMP"/ffmpeg-*/bin/ffprobe /usr/local/bin/ \
    || ! "${SUDO[@]}" ln -sf /usr/local/bin/ffmpeg /usr/local/bin/ffprobe /usr/bin/; then
    echo "Failed to install FFmpeg. Exiting..."
    exit 1
fi
rm -rf "$FFMPEG_TMP"

if ! command -v git >/dev/null 2>&1; then
    echo "Git installation could not be verified. Exiting..."
    exit 1
fi
if ! command -v ffmpeg >/dev/null 2>&1 || ! command -v ffprobe >/dev/null 2>&1; then
    echo "FFmpeg installation could not be verified. Exiting..."
    exit 1
fi
if ! command -v cmake >/dev/null 2>&1 || ! command -v c++ >/dev/null 2>&1; then
    echo "CMake and C++ build tool installation could not be verified. Exiting..."
    exit 1
fi
echo "Using $(git --version)"
echo "Using $(ffmpeg -version | head -n 1)"
echo "Using $(cmake --version | head -n 1)"

LEGACY_PYTHON_PREF="/etc/apt/preferences.d/deadsnakes-python3.12"
if [ -e "$LEGACY_PYTHON_PREF" ]; then
    if [ "$CAN_INSTALL_SYSTEM_PACKAGES" -eq 1 ]; then
        "${SUDO[@]}" rm -f "$LEGACY_PYTHON_PREF"
    else
        echo "Warning: cannot remove ignored APT preference file $LEGACY_PYTHON_PREF without root access."
    fi
fi

echo "Checking for an existing Python $REQUIRED_PY_VERSION (or newer 3.12.x) virtual environment..."

if [ -x "$VENV_PYTHON" ] && python_is_current "$VENV_PYTHON"; then
    echo "Reusing existing $("$VENV_PYTHON" --version 2>&1) virtual environment."
    VENV_READY=1
else
    if [ -x "$VENV_PYTHON" ]; then
        echo "Existing virtual environment is $("$VENV_PYTHON" --version 2>&1); it will be rebuilt on Python $REQUIRED_PY_VERSION or newer."
    else
        echo "No existing Python 3.12 virtual environment was found."
    fi

    if [ "$CAN_INSTALL_SYSTEM_PACKAGES" -ne 1 ]; then
        echo "Root access or sudo is required to install Python $REQUIRED_PY_VERSION. Exiting..."
        exit 1
    fi

    if command -v python3.12 >/dev/null 2>&1 && ! python3.12 -c 'import sys; raise SystemExit(0 if sys.version_info.releaselevel == "final" else 1)' 2>/dev/null; then
        echo "Removing the existing prerelease Python 3.12 packages..."
        if ! "${SUDO[@]}" apt-get remove -y python3.12 python3.12-venv python3.12-dev python3.12-tk; then
            echo "Failed to remove the prerelease Python 3.12 packages. Exiting..."
            exit 1
        fi
        if ! "${SUDO[@]}" apt-get autoremove -y; then
            echo "Failed to clean obsolete Python packages. Exiting..."
            exit 1
        fi
    fi

    PYTHON_BIN=""

    echo "Adding deadsnakes PPA..."
    if "${SUDO[@]}" apt-get install -y software-properties-common \
        && "${SUDO[@]}" add-apt-repository -y ppa:deadsnakes/ppa; then

        "${SUDO[@]}" tee /etc/apt/preferences.d/deadsnakes-python312.pref >/dev/null <<'EOF'
Package: python3.12 python3.12-*
Pin: release o=LP-PPA-deadsnakes
Pin-Priority: 1000
EOF

        if ! "${SUDO[@]}" apt-get update; then
            echo "Warning: failed to refresh APT package information; using the cached package lists."
        fi

        echo "Available Python 3.12 versions:"
        apt-cache policy python3.12

        echo "Installing the latest stable Python 3.12 from APT..."
        if ! "${SUDO[@]}" apt-get install -y python3.12 python3.12-venv python3.12-dev python3.12-tk; then
            echo "Warning: the APT Python 3.12 packages could not be installed."
        fi
    else
        echo "Warning: the deadsnakes PPA could not be added."
    fi

    if python_is_current python3.12; then
        PYTHON_BIN="python3.12"
        echo "APT provided $(python3.12 --version 2>&1)."
    else
        if command -v python3.12 >/dev/null 2>&1; then
            echo "APT only offers $(python3.12 --version 2>&1); Python $REQUIRED_PY_VERSION is not packaged for this Ubuntu release."
        fi

        echo "Falling back to a uv-managed standalone CPython $REQUIRED_PY_VERSION..."
        if ! command -v uv >/dev/null 2>&1; then
            curl -LsSf https://astral.sh/uv/install.sh | sh
        fi
        export PATH="$HOME/.local/bin:$PATH"

        if command -v uv >/dev/null 2>&1 && uv python install "$REQUIRED_PY_VERSION"; then
            PYTHON_BIN="$(uv python find "$REQUIRED_PY_VERSION" 2>/dev/null)"
        fi

        if ! python_is_current "$PYTHON_BIN"; then
            echo "Failed to obtain Python $REQUIRED_PY_VERSION. Exiting..."
            exit 1
        fi
        echo "Using uv-managed $("$PYTHON_BIN" --version 2>&1)."
    fi

    echo "Recreating the virtual environment with $("$PYTHON_BIN" --version 2>&1)..."
    if ! "$PYTHON_BIN" -m venv --clear "$VENV_DIR"; then
        echo "Failed to recreate the Python 3.12 virtual environment. Exiting..."
        exit 1
    fi

    if [ ! -x "$VENV_PYTHON" ] || ! python_is_current "$VENV_PYTHON"; then
        echo "Failed to create the Python $REQUIRED_PY_VERSION virtual environment. Exiting..."
        exit 1
    fi

    VENV_READY=1
fi

if [ "$VENV_READY" -ne 1 ]; then
    echo "Python $REQUIRED_PY_VERSION virtual environment setup failed. Exiting..."
    exit 1
fi

if [ "$CAN_INSTALL_SYSTEM_PACKAGES" -eq 1 ]; then
    if ! "${SUDO[@]}" apt-get install -y python3.12-tk; then
        echo "Warning: python3.12-tk could not be installed; the local folder picker will be unavailable."
    fi
fi

source "$VENV_DIR/bin/activate"

python -m pip install --upgrade pip

python -m pip install uv


cd ..

if ! uv pip install -r video_caption_requirements.txt; then
    echo "Failed to install application requirements. Exiting..."
    exit 1
fi

cd SECourses_Video_Captioner_Pro

python -m vcap.core.cuda_runtime --repair || exit 1

python -m vcap.models.llamacpp_install --build-if-needed || echo "WARNING: llama.cpp build failed; GGUF variants can be installed later from the System & Models tab."

echo "Models will be downloaded individually when first selected in the app."

echo .
echo .
echo .
echo .
echo SECourses Video Captioner Pro installed check out all messages and save them before close to verify any errors or not later
