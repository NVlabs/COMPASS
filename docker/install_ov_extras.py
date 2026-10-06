"""Install Isaac Lab OV extras into the Python interpreter running this script."""

import importlib.util
import os
import stat
import subprocess
import sys
import tempfile
import tomllib
from pathlib import Path


def prepare_runtime_directories():
    """Allow docker/run.sh's supplemental GID 1234 to write OVRTX runtime data."""
    runtime = Path(importlib.util.find_spec("ovrtx").origin).parent / "bin"
    for relative in ("cache", "mdl/omniverse_exts"):
        directory = runtime / relative
        directory.mkdir(parents=True, exist_ok=True)
        for path in [directory, *directory.rglob("*")]:
            if path.is_symlink():
                continue
            os.chown(path, -1, 1234)
            mode = path.stat().st_mode | stat.S_IRGRP | stat.S_IWGRP
            if path.is_dir():
                mode |= stat.S_IXGRP | stat.S_ISGID
            path.chmod(mode)


def main():
    isaaclab = Path(os.environ["ISAACLAB_PATH"])
    with (isaaclab / "pyproject.toml").open("rb") as f:
        config = tomllib.load(f)

    extras = config["project"]["optional-dependencies"]
    requirements = list(dict.fromkeys(extras["ovrtx"] + extras["ovphysx"]))
    # Preserve Isaac Lab's compatibility overrides for Kit's shared dependencies.
    overrides = config["tool"]["uv"].get("override-dependencies", [])

    with tempfile.TemporaryDirectory() as temporary:
        override_file = Path(temporary) / "overrides.txt"
        override_file.write_text("\n".join(overrides) + "\n")

        subprocess.run(
            [
                "uv",
                "pip",
                "install",
                "--python",
                sys.executable,
                "--index",
                "https://pypi.nvidia.com",
                "--index-strategy",
                "unsafe-best-match",
                "--prerelease",
                "allow",
                "--overrides",
                str(override_file),
                *requirements,
            ],
            cwd=isaaclab,
            check=True,
        )

    # OVRTX creates derived-data, texture, and generated-material files at startup.
    # The package is installed as root, but local runs use the host UID.
    prepare_runtime_directories()


if __name__ == "__main__":
    main()
