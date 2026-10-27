"""Build and verify the installer ZIP using the Git-managed downloader source."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
import zipfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from vcap import VERSION

INSTALLER_FILES = (
    "Massed_Compute_Install.sh", "Massed_Compute_Instructions_READ.txt",
    "Models_Downloader.py", "RunPod_Install_SECourses_Video_Captioner_Pro.sh",
    "Runpod_SimplePod_Video_Captioner_Pro_Instructions.txt", "video_caption_requirements.txt",
    "Windows_Download_Models_and_or_Resume.bat", "Windows_Install_and_Update.bat",
    "Windows_Run_Video_Captioner_Pro.bat",
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--distribution-dir", type=Path, default=ROOT.parent)
    args = parser.parse_args()
    distribution = args.distribution_dir.resolve()
    target = distribution / f"SECourses_Video_Captioner_Pro_v{VERSION.replace('.', '_')}.zip"
    sources = {}
    for name in INSTALLER_FILES:
        path = ROOT / name if name == "Models_Downloader.py" else distribution / name
        sources[name] = path.read_bytes()
    with zipfile.ZipFile(target, "x", compression=zipfile.ZIP_DEFLATED, compresslevel=9) as package:
        for name, content in sources.items():
            package.writestr(name, content)
    with zipfile.ZipFile(target) as package:
        assert package.testzip() is None
        assert package.namelist() == list(INSTALLER_FILES)
        for name, expected in sources.items():
            assert package.read(name) == expected, name
    manifest = {
        "version": VERSION, "path": str(target), "bytes": target.stat().st_size,
        "sha256": hashlib.sha256(target.read_bytes()).hexdigest(),
        "members": {name: {"bytes": len(content), "sha256": hashlib.sha256(content).hexdigest()}
                    for name, content in sources.items()},
    }
    evidence = ROOT / "temp" / "release_builds"
    evidence.mkdir(parents=True, exist_ok=True)
    (evidence / f"v{VERSION}_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({key: value for key, value in manifest.items() if key != "members"}, indent=2))


if __name__ == "__main__":
    main()
