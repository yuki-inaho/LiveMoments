#!/usr/bin/env python3
"""Download pinned official assets, recording gated/missing assets explicitly."""

import argparse
import hashlib
import json
import os
import sys
import zipfile
from pathlib import Path
from urllib.request import urlopen

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from live_utils.config import runtime_dir

RUNTIME = runtime_dir()
os.environ.setdefault("HF_HOME", str(RUNTIME / "hf-cache"))

from huggingface_hub import HfApi, snapshot_download
from huggingface_hub.errors import GatedRepoError, HfHubHTTPError

REPOSITORIES = {
    "livemoments": ("Clara-7/LiveMoments", "f0ccce7bb606174cdf8c983df82b3eb805f328e4", None),
    "sd3": (
        "stabilityai/stable-diffusion-3-medium-diffusers",
        "ea42f8cef0f178587cf766dc8129abd379c90671",
        [
            "scheduler/scheduler_config.json",
            "vae/config.json",
            "vae/diffusion_pytorch_model.safetensors",
            "LICENSE",
            "README.md",
        ],
    ),
}


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--only", choices=["livemoments", "sd3", "raft", "all"], default="all")
    parser.add_argument(
        "--include-text-encoders",
        action="store_true",
        help="After SD3 access is authorized, also fetch components for CPU empty-prompt embeddings.",
    )
    args = parser.parse_args()
    model_root = RUNTIME / "models"
    model_root.mkdir(parents=True, exist_ok=True)
    manifest_path = RUNTIME / "model-manifest.json"
    manifest = (
        json.loads(manifest_path.read_text())
        if manifest_path.exists()
        else {
            "schema_version": 1,
            "gpu_inference_tested": False,
            "sources": {},
            "files": [],
            "missing_assets": [],
        }
    )
    for name, (repo, revision, patterns) in REPOSITORIES.items():
        if args.only != "all" and name != args.only:
            continue
        destination = model_root / name
        if name == "sd3" and args.include_text_encoders:
            patterns = patterns + [
                "model_index.json",
                "text_encoder/config.json",
                "text_encoder/model.safetensors",
                "text_encoder_2/config.json",
                "text_encoder_2/model.safetensors",
                "text_encoder_3/config.json",
                "text_encoder_3/model.safetensors.index.json",
                "text_encoder_3/model-00001-of-00002.safetensors",
                "text_encoder_3/model-00002-of-00002.safetensors",
                "tokenizer/*",
                "tokenizer_2/*",
                "tokenizer_3/*",
            ]
        try:
            print(f"Downloading official {repo}@{revision}", flush=True)
            info = HfApi().model_info(repo, revision=revision, files_metadata=True)
            snapshot_download(
                repo,
                revision=revision,
                local_dir=str(destination),
                allow_patterns=patterns,
                max_workers=2,
            )
            files = []
            for sibling in info.siblings:
                path = destination / sibling.rfilename
                if not path.is_file():
                    continue
                actual_sha = sha256(path)
                expected_sha = sibling.lfs.sha256 if sibling.lfs else None
                if expected_sha and actual_sha != expected_sha:
                    raise RuntimeError(f"SHA-256 mismatch for {repo}/{sibling.rfilename}")
                files.append(
                    {
                        "component": name,
                        "path": str(path),
                        "relative_path": sibling.rfilename,
                        "size_bytes": path.stat().st_size,
                        "sha256": actual_sha,
                        "upstream_lfs_sha256": expected_sha,
                        "upstream_lfs_verified": bool(expected_sha),
                    }
                )
            manifest["files"] = [f for f in manifest["files"] if f["component"] != name] + files
            manifest["sources"][name] = {
                "repository": repo,
                "revision": revision,
                "download_status": "complete",
                "local_path": str(destination),
            }
            print(f"{name}: {len(files)} files verified", flush=True)
        except (GatedRepoError, HfHubHTTPError) as error:
            status = error.response.status_code if error.response is not None else None
            if status not in (401, 403):
                raise
            manifest["sources"][name] = {
                "repository": repo,
                "revision": revision,
                "download_status": "blocked_authorization",
                "http_status": status,
                "local_path": str(destination),
                "required_action": "Use an account with accepted SD3 model terms; authenticate locally.",
            }
            print(f"{name}: HTTP {status}, authorized Hugging Face access required", flush=True)
        manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    archive = model_root / "raft-models.zip"
    raft_model = model_root / "raft" / "raft-sintel.pth"
    if args.only in {"all", "raft"} and not archive.is_file():
        temporary = archive.with_suffix(".zip.partial")
        with (
            urlopen(
                "https://dl.dropboxusercontent.com/s/4j4z58wuv8o0mfz/models.zip", timeout=60
            ) as source,
            temporary.open("wb") as target,
        ):
            while chunk := source.read(1024 * 1024):
                target.write(chunk)
        temporary.replace(archive)
    if archive.is_file():
        with zipfile.ZipFile(archive) as handle:
            wanted = next(
                (
                    p
                    for p in handle.namelist()
                    if p.endswith("/raft-sintel.pth") or p == "raft-sintel.pth"
                ),
                None,
            )
            if not wanted:
                raise RuntimeError("Official RAFT archive lacks raft-sintel.pth")
            raft_model.parent.mkdir(exist_ok=True)
            with handle.open(wanted) as source, raft_model.open("wb") as target:
                while chunk := source.read(1024 * 1024):
                    target.write(chunk)
        manifest["sources"]["raft"] = {
            "repository": "https://github.com/princeton-vl/RAFT",
            "revision": "2888e15a51fa41140771d3f498ed8023cff098d1",
            "download_url": "https://dl.dropboxusercontent.com/s/4j4z58wuv8o0mfz/models.zip",
            "archive_sha256": sha256(archive),
            "download_status": "complete",
            "upstream_checksum_available": False,
        }
        manifest["files"] = [f for f in manifest["files"] if f["component"] != "raft"] + [
            {
                "component": "raft",
                "path": str(raft_model),
                "relative_path": "raft-sintel.pth",
                "size_bytes": raft_model.stat().st_size,
                "sha256": sha256(raft_model),
                "upstream_lfs_verified": False,
            }
        ]
    missing = []
    required = {
        "LiveMoments target weights": model_root
        / "livemoments/transformer/diffusion_pytorch_model.safetensors",
        "LiveMoments reference weights": model_root
        / "livemoments/transformer_ref/diffusion_pytorch_model.safetensors",
        "LiveMoments motion encoder": model_root / "livemoments/motion_encoder.pt",
        "SD3 scheduler": model_root / "sd3/scheduler/scheduler_config.json",
        "SD3 VAE config": model_root / "sd3/vae/config.json",
        "SD3 VAE weights": model_root / "sd3/vae/diffusion_pytorch_model.safetensors",
        "prompt embeddings (not released upstream)": model_root / "prompt_embeds.pt",
        "pooled prompt embeddings (not released upstream)": model_root / "pooled_prompt_embeds.pt",
        "RAFT Sintel": raft_model,
    }
    for label, path in required.items():
        if not path.is_file():
            missing.append({"asset": label, "expected_path": str(path)})
    manifest["missing_assets"] = missing
    manifest["model_assets_complete"] = not missing
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps({"manifest": str(manifest_path), "missing_assets": missing}, indent=2))
    return 0 if manifest["model_assets_complete"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
