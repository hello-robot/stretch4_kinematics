# Store the version here so:
# 1) we don't load heavy dependencies when querying the version
# 2) setuptools can extract it dynamically at build time

import os
import subprocess

BASE_VERSION = "2026.09.14"

def get_version() -> str:
    version = BASE_VERSION
    if os.environ.get("IS_EDITABLE_INSTALL") == "1":
        try:
            git_hash = subprocess.check_output(
                ["git", "rev-parse", "--short", "HEAD"],
                cwd=os.path.dirname(os.path.abspath(__file__)),
                stderr=subprocess.DEVNULL
            ).decode("utf-8").strip()
            if git_hash:
                version = f"{version}+{git_hash}"
        except Exception:
            pass
    return version

__version__ = get_version()
