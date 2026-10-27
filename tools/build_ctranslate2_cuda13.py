"""Build upstream CTranslate2 4.8.4 for our shared CUDA 13.0 runtime.

Run with the project venv. Source must be a recursive checkout of the pinned
upstream commit; toolkit must contain CUDA 13.0 headers, nvcc, and libraries.
The wheel deliberately excludes NVIDIA libraries supplied by PyTorch.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

COMMIT = "5de572ef60b67a0276c5989ee73d0072808ff4c3"
VERSION = "4.8.4+cu130"


def run(args: list[str], *, cwd: Path | None = None, env=None) -> str:
    print(subprocess.list2cmdline(args), flush=True)
    result = subprocess.run(args, cwd=cwd, env=env, check=True, stdout=sys.stdout, stderr=sys.stderr,
                            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
    return str(result.returncode)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--toolkit", type=Path, required=True)
    parser.add_argument("--cudnn-include", type=Path, required=True)
    parser.add_argument("--cudnn-library", type=Path, required=True)
    parser.add_argument("--jobs", type=int, default=8)
    parser.add_argument("--architectures", default="7.5;8.0;8.6;8.9;9.0;10.0;10.3;12.0;12.1+PTX")
    parser.add_argument("--configure-only", action="store_true")
    parser.add_argument("--package-only", action="store_true", help="Reuse the already-built native library")
    args = parser.parse_args()
    source, toolkit = args.source.resolve(), args.toolkit.resolve()
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=source, text=True).strip()
    if commit != COMMIT:
        raise SystemExit(f"Expected upstream {COMMIT}, got {commit}")
    nvcc = toolkit / "bin" / ("nvcc.exe" if os.name == "nt" else "nvcc")
    nvcc_version = subprocess.check_output([str(nvcc), "--version"], text=True)
    if "release 13.0," not in nvcc_version:
        raise SystemExit("Use CUDA 13.0 nvcc to match the shipped cu130 runtime")
    build = source / "build-cu130"
    install = source / "install-cu130"
    output = source / "wheelhouse-cu130"
    output.mkdir(exist_ok=True)
    (source / "python/ctranslate2/version.py").write_text(f'__version__ = "{VERSION}"\n', encoding="utf-8")
    # FindCUDA in Ubuntu's CMake predates Blackwell. Give it one known
    # architecture, then pass the remaining targets directly to nvcc.
    architectures = args.architectures.split(";")
    nvcc_flags = ["-Xfatbin=-compress-all"]
    for arch in architectures[1:]:
        number = arch.removesuffix("+PTX").replace(".", "")
        nvcc_flags.append(f"--generate-code=arch=compute_{number},code=sm_{number}")
        if arch.endswith("+PTX"):
            nvcc_flags.append(f"--generate-code=arch=compute_{number},code=compute_{number}")
    command = [
        "cmake", "-S", str(source), "-B", str(build),
        "-DCMAKE_POLICY_VERSION_MINIMUM=3.5", "-DCMAKE_BUILD_TYPE=Release",
        f"-DCMAKE_INSTALL_PREFIX={install.as_posix()}", "-DBUILD_CLI=OFF", "-DBUILD_TESTS=OFF",
        "-DCMAKE_OBJECT_PATH_MAX=128",
        "-DWITH_MKL=OFF", "-DWITH_DNNL=OFF", "-DWITH_RUY=ON",
        "-DOPENMP_RUNTIME=COMP", "-DWITH_CUDA=ON", "-DWITH_CUDNN=ON",
        "-DCUDA_DYNAMIC_LOADING=ON", "-DCUDA_USE_STATIC_CUDA_RUNTIME=OFF",
        f"-DCUDA_TOOLKIT_ROOT_DIR={toolkit.as_posix()}", f"-DCUDA_NVCC_EXECUTABLE={nvcc.as_posix()}",
        f"-DCUDA_ARCH_LIST={architectures[0]}", f"-DCUDA_NVCC_FLAGS={';'.join(nvcc_flags)}",
        f"-DCUDNN_INCLUDE_DIR={args.cudnn_include.resolve().as_posix()}",
        f"-DCUDNN_LIBRARIES={args.cudnn_library.resolve().as_posix()}",
    ]
    if os.name == "nt":
        command += ["-G", "Ninja", f"-DCUDA_CUDART_LIBRARY={(toolkit / 'lib/x64/cudart.lib').as_posix()}"]
    else:
        command += ["-G", "Ninja", f"-DCUDA_CUDART_LIBRARY={toolkit / 'lib/libcudart.so.13'}"]
    if not args.package_only:
        run(command)
    if args.configure_only:
        return
    if not args.package_only:
        run(["cmake", "--build", str(build), "--config", "Release", "--target", "install", "--parallel", str(args.jobs)])
    if os.name == "nt":
        shutil.copy2(install / "bin/ctranslate2.dll", source / "python/ctranslate2/ctranslate2.dll")
    shutil.copy2(source / "README.md", source / "python/README.md")
    licenses = source / "python/ctranslate2/licenses"
    licenses.mkdir(exist_ok=True)
    shutil.copy2(source / "LICENSE", licenses / "CTranslate2-LICENSE")
    for path in (source / "third_party").rglob("LICENSE*"):
        if path.is_file() and ".git" not in path.parts:
            shutil.copy2(path, licenses / "__".join(path.relative_to(source).parts))
    if os.name != "nt":
        for path in (Path("/usr/share/doc/libgomp1/copyright"), Path("/usr/share/common-licenses/GPL-3")):
            if path.is_file():
                shutil.copy2(path, licenses / ("GCC-runtime-" + path.name))
    setup = source / "python/setup.py"
    setup_text = setup.read_text(encoding="utf-8")
    setup_text = setup_text.replace("package_data = {}", 'package_data = {"ctranslate2": ["licenses/*"]}')
    setup_text = setup_text.replace('package_data["ctranslate2"] = ["*.dll"]', 'package_data["ctranslate2"] = ["*.dll", "licenses/*"]')
    setup.write_text(setup_text, encoding="utf-8")
    environment = dict(os.environ, CTRANSLATE2_ROOT=str(install), CMAKE_BUILD_PARALLEL_LEVEL=str(args.jobs))
    run([sys.executable, "setup.py", "bdist_wheel", "--dist-dir", str(output)], cwd=source / "python", env=environment)
    if os.name != "nt":
        environment["PATH"] = str(Path(sys.executable).parent) + os.pathsep + environment.get("PATH", "")
        environment["LD_LIBRARY_PATH"] = str(install / "lib")
        run([sys.executable, "-m", "auditwheel", "repair", "--plat", "manylinux_2_35_x86_64",
             "--exclude", "libcudnn.so.9", "--exclude", "libcudart.so.13", "--wheel-dir", str(output / "repaired"),
             str(output / f"ctranslate2-{VERSION}-cp312-cp312-linux_x86_64.whl")], env=environment)
    manifest = {"commit": commit, "version": VERSION, "nvcc": nvcc_version,
                "architectures": args.architectures, "configure": command}
    (output / "build_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
