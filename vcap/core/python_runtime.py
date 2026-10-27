"""Select child Python executables without escaping their virtual environment."""

from __future__ import annotations

import os
from pathlib import Path
import sys


def project_python(app_dir: Path, override: str | None = None) -> Path:
    """Keep the venv executable's lexical path, including Linux symlinks.

    Resolving ``venv/bin/python`` to its underlying interpreter loses the
    adjacent ``pyvenv.cfg`` and therefore the venv's installed dependencies.
    File existence may follow symlinks; the executable passed to Popen must not.
    """
    if override:
        value = os.fspath(override).strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in {"'", '"'}:
            value = value[1:-1]
        return Path(os.path.abspath(os.path.expandvars(os.path.expanduser(value))))
    candidates = (
        app_dir / "venv" / ("Scripts/python.exe" if os.name == "nt" else "bin/python"),
        Path(sys.executable),
    )
    return next((path.absolute() for path in candidates if path.is_file()), Path(sys.executable).absolute())
