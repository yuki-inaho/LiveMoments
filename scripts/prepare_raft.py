#!/usr/bin/env python3
"""Fetch the official, pinned RAFT source without replacing an existing checkout."""

import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from live_utils.config import runtime_dir

REVISION = "2888e15a51fa41140771d3f498ed8023cff098d1"
URL = "https://github.com/princeton-vl/RAFT.git"


def patch_normalization(path):
    text = path.read_text()
    lines = [f"        image{i} = 2 * (image{i} / 255.0) - 1.0" for i in (1, 2)]
    marker = "# LiveMoments inputs are already normalized to [-1, 1]."
    if marker in text and all(line not in text for line in lines):
        return
    if not all(text.count(line) == 1 for line in lines):
        raise RuntimeError("Unexpected RAFT normalization code; refusing to modify it")
    text = text.replace(lines[0], "        " + marker).replace(lines[1] + "\n", "")
    path.write_text(text)


def main():
    target = runtime_dir() / "src/RAFT"
    if target.exists():
        actual = subprocess.check_output(
            ["git", "-C", str(target), "rev-parse", "HEAD"], text=True
        ).strip()
        if actual != REVISION:
            raise RuntimeError(
                "Existing RAFT checkout has a different revision; move it explicitly before retrying"
            )
    else:
        target.parent.mkdir(parents=True, exist_ok=True)
        subprocess.run(["git", "clone", URL, str(target)], check=True)
        subprocess.run(["git", "-C", str(target), "checkout", "--detach", REVISION], check=True)
    patch_normalization(target / "core/raft.py")
    print("Pinned RAFT source prepared with the LiveMoments README normalization adjustment")


if __name__ == "__main__":
    main()
