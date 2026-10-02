#!/usr/bin/env python3
"""Manual offline GPU entry point with asset/data/free-memory guards."""

import argparse
import fcntl
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from live_utils.config import runtime_dir

RUNTIME = runtime_dir()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "config/inference_example.yml")
    parser.add_argument("--output-dir", type=Path, default=RUNTIME / "outputs")
    parser.add_argument("--output-name", default="livemoments")
    parser.add_argument("--gpu-index", type=int, default=0)
    parser.add_argument(
        "--min-free-mib",
        type=int,
        default=28000,
        help="Conservative minimum free VRAM; calibrate for your input size after a pilot.",
    )
    args = parser.parse_args()
    if args.min_free_mib < 1:
        parser.error("--min-free-mib must be positive")
    if args.gpu_index < 0:
        parser.error("--gpu-index must be nonnegative")
    if args.output_name in {"", ".", ".."} or Path(args.output_name).name != args.output_name:
        parser.error("--output-name must be a single directory name")
    if args.output_dir.exists() and (
        not args.output_dir.is_dir() or any(args.output_dir.iterdir())
    ):
        parser.error("--output-dir must be new or empty; refusing to overwrite results")
    cfg = args.config.resolve()
    env = os.environ.copy()
    env.update(
        HF_HOME=env.get("HF_HOME", str(RUNTIME / "hf-cache")),
        HF_HUB_OFFLINE="1",
        TRANSFORMERS_OFFLINE="1",
        LIVEMOMENTS_CONFIG=str(cfg),
        CUDA_VISIBLE_DEVICES=str(args.gpu_index),
    )
    cpu_env = env.copy()
    cpu_env["CUDA_VISIBLE_DEVICES"] = ""
    checked = subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts/preflight.py"),
            "--json",
            "--config",
            str(cfg),
            "--check-data",
        ],
        cwd=ROOT,
        env=cpu_env,
        text=True,
        capture_output=True,
    )
    if checked.returncode:
        print(checked.stdout, end="")
        print(checked.stderr, end="", file=sys.stderr)
        print("Inference blocked: complete missing assets/data shown above first.", file=sys.stderr)
        return checked.returncode
    RUNTIME.mkdir(parents=True, exist_ok=True)
    shared_lock = RUNTIME.parent / "gpu-inference.lock"
    with shared_lock.open("a") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            print(
                "Another restoration-tool inference currently holds the GPU lock.", file=sys.stderr
            )
            return 3
        result = subprocess.run(
            [
                "nvidia-smi",
                "--id",
                str(args.gpu_index),
                "--query-gpu=memory.free",
                "--format=csv,noheader,nounits",
            ],
            text=True,
            capture_output=True,
            check=True,
        )
        free_mib = int(result.stdout.splitlines()[0].strip())
        if free_mib < args.min_free_mib:
            print(
                f"GPU free {free_mib} MiB < required {args.min_free_mib} MiB. Try again when idle.",
                file=sys.stderr,
            )
            return 3
        args.output_dir.mkdir(parents=True, exist_ok=True)
        return subprocess.run(
            [
                sys.executable,
                str(ROOT / "infer/infer_LiveMoments.py"),
                "--device",
                "cuda",
                "--output_dir",
                str(args.output_dir.resolve()),
                "--output_dir_name",
                args.output_name,
            ],
            cwd=ROOT,
            env=env,
        ).returncode


if __name__ == "__main__":
    raise SystemExit(main())
