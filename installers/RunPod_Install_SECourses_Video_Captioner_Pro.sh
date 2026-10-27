cd /workspace

export UV_SKIP_WHEEL_FILENAME_CHECK=1
export UV_LINK_MODE=copy

SUDO=()
if [ "$(id -u)" -ne 0 ]; then
    if command -v sudo >/dev/null 2>&1; then
        SUDO=(sudo)
    else
        echo "Root access or sudo is required to install Git, FFmpeg, and llama.cpp build tools. Exiting..."
        exit 1
    fi
fi

echo "Ensuring Git, xz, CMake, build tools, and libcurl are installed..."
if ! command -v git >/dev/null 2>&1 \
    || ! command -v xz >/dev/null 2>&1 \
    || ! command -v cmake >/dev/null 2>&1 \
    || ! command -v c++ >/dev/null 2>&1 \
    || ! command -v dpkg-query >/dev/null 2>&1 \
    || ! dpkg-query -W -f='${Status}' libcurl4-openssl-dev 2>/dev/null | grep -q "install ok installed"; then
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

git clone https://github.com/FurkanGozukara/SECourses_Video_Captioner_Pro

cd SECourses_Video_Captioner_Pro

git reset --hard

git pull

echo "Provisioning the latest stable Python 3.12 with uv..."

VENV_DIR="venv"
VENV_PYTHON="$VENV_DIR/bin/python"
PYTHON_REQUEST="3.12"
PYTHON_VERSION_CMD='import sys; print(".".join(map(str, sys.version_info[:3])))'

export UV_PYTHON_PREFERENCE=only-managed

if ! command -v uv >/dev/null 2>&1; then
    echo "Installing uv..."
    if command -v curl >/dev/null 2>&1; then
        curl -LsSf https://astral.sh/uv/install.sh | sh
    elif command -v wget >/dev/null 2>&1; then
        wget -qO- https://astral.sh/uv/install.sh | sh
    else
        echo "Neither curl nor wget is available to install uv. Exiting..."
        exit 1
    fi
fi

export PATH="$HOME/.local/bin:$PATH"

if ! command -v uv >/dev/null 2>&1; then
    echo "Failed to install uv. Exiting..."
    exit 1
fi

if ! uv python install "$PYTHON_REQUEST"; then
    echo "Failed to download Python $PYTHON_REQUEST. Exiting..."
    exit 1
fi

TARGET_PYTHON="$(uv python find "$PYTHON_REQUEST")"
if [ -z "$TARGET_PYTHON" ] || [ ! -x "$TARGET_PYTHON" ]; then
    echo "Could not locate the managed Python $PYTHON_REQUEST interpreter. Exiting..."
    exit 1
fi

PYTHON_FULL_VERSION="$("$TARGET_PYTHON" -c "$PYTHON_VERSION_CMD")"
echo "Using Python $PYTHON_FULL_VERSION from $TARGET_PYTHON"

if [ -x "$VENV_PYTHON" ] && [ "$("$VENV_PYTHON" -c "$PYTHON_VERSION_CMD" 2>/dev/null)" = "$PYTHON_FULL_VERSION" ]; then
    echo "Reusing existing Python $PYTHON_FULL_VERSION virtual environment."
else
    if [ -e "$VENV_DIR" ]; then
        echo "Existing virtual environment is not on Python $PYTHON_FULL_VERSION; recreating it..."
        rm -rf "$VENV_DIR"
    else
        echo "No existing virtual environment was found; creating one..."
    fi

    if ! uv venv --python "$TARGET_PYTHON" --seed "$VENV_DIR"; then
        echo "Failed to create the Python $PYTHON_FULL_VERSION virtual environment. Exiting..."
        exit 1
    fi
fi

if [ ! -x "$VENV_PYTHON" ]; then
    echo "Python 3.12 virtual environment setup failed. Exiting..."
    exit 1
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
