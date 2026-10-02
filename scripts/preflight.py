#!/usr/bin/env python3
"""Offline CPU readiness check. Never opens a CUDA context."""

import argparse
import importlib.metadata
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from live_utils.config import load_config, runtime_dir

RUNTIME = runtime_dir()
os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ.setdefault("HF_HOME", str(RUNTIME / "hf-cache"))
os.chdir(ROOT)
sys.path.insert(0, str(ROOT))


def inspect(config_path, check_data=False):
    import torch
    import torchvision
    from diffusers import (
        AutoencoderKL,
        FlowMatchEulerDiscreteScheduler,
        SD3Transformer2DModel,
    )
    from safetensors import safe_open

    from live_utils.cuda_env import CUDNN_LIBS, wheel_cudnn_dir
    from live_utils.motion_network import MotionEncoder
    from models.transformer_sd3_fusem import SD3TransformerFuseMotion2DModel

    cfg = load_config(config_path)
    raft_source = Path(cfg["raft_path"]) / "core/raft.py"
    normalization_disabled = raft_source.is_file() and all(
        f"image{i} = 2 * (image{i} / 255.0) - 1.0" not in raft_source.read_text() for i in (1, 2)
    )
    sys.path.insert(0, str(Path(cfg["raft_path"]) / "core"))
    from argparse import Namespace

    from raft import RAFT

    report = {
        "project": "LiveMoments",
        "source_revision": "39df3c3ef58be7d7856d233484d1f8c82cdd7eda",
        "python": sys.version.split()[0],
        "torch": torch.__version__,
        "torchvision": torchvision.__version__,
        "cuda_wheel": torch.version.cuda,
        "compiled_cuda_arches": torch._C._cuda_getArchFlags(),
        "versions": {
            k: importlib.metadata.version(k)
            for k in [
                "diffusers",
                "transformers",
                "peft",
                "accelerate",
                "numpy",
                "opencv-python-headless",
            ]
        },
        "cuda_context_initialized": torch.cuda.is_initialized(),
        "gpu_inference_tested": False,
        "cpu_imports_passed": True,
        "config": str(Path(config_path).resolve()),
        "models": [],
        "missing_assets": [],
        "data_checked": check_data,
        "raft_normalization_disabled": normalization_disabled,
        "wheel_cudnn_directory": str(wheel_cudnn_dir()),
        "wheel_cudnn_libraries_present": all(
            (wheel_cudnn_dir() / name).is_file() for name in CUDNN_LIBS
        ),
    }
    required_paths = {
        "SD3 scheduler": Path(cfg["SD3_weight_path"]) / "scheduler/scheduler_config.json",
        "SD3 VAE config": Path(cfg["SD3_weight_path"]) / "vae/config.json",
        "SD3 VAE weights": Path(cfg["SD3_weight_path"]) / "vae/diffusion_pytorch_model.safetensors",
        "LiveMoments target config": Path(cfg["pretrained_weight_path"])
        / "transformer/config.json",
        "LiveMoments target weights": Path(cfg["pretrained_weight_path"])
        / "transformer/diffusion_pytorch_model.safetensors",
        "LiveMoments reference config": Path(cfg["pretrained_weight_path"])
        / "transformer_ref/config.json",
        "LiveMoments reference weights": Path(cfg["pretrained_weight_path"])
        / "transformer_ref/diffusion_pytorch_model.safetensors",
        "motion encoder": Path(cfg["motion_encoder_weight_path"]),
        "RAFT Sintel": Path(cfg["raft_weight_path"]),
        "prompt embeddings (not released upstream)": Path(cfg["prompt_embeds_path"]),
        "pooled prompt embeddings (not released upstream)": Path(cfg["pooled_prompt_embeds_path"]),
    }
    for label, path in required_paths.items():
        if not path.is_file():
            report["missing_assets"].append({"asset": label, "path": str(path)})
    # Construct modules on meta so 8.5GB transformers are not allocated/read to RAM.
    for name, cls in [
        ("transformer", SD3TransformerFuseMotion2DModel),
        ("transformer_ref", SD3Transformer2DModel),
    ]:
        directory = Path(cfg["pretrained_weight_path"]) / name
        if (
            not (directory / "config.json").is_file()
            or not (directory / "diffusion_pytorch_model.safetensors").is_file()
        ):
            continue
        with torch.device("meta"):
            model = cls.from_config(cls.load_config(str(directory)))
        expected = {k: list(v.shape) for k, v in model.state_dict().items()}
        with safe_open(
            str(directory / "diffusion_pytorch_model.safetensors"), framework="pt", device="cpu"
        ) as handle:
            actual = {k: list(handle.get_slice(k).get_shape()) for k in handle.keys()}
        mismatches = [
            k for k in expected.keys() | actual.keys() if expected.get(k) != actual.get(k)
        ]
        report["models"].append(
            {
                "component": name,
                "checkpoint_tensors": len(actual),
                "state_keys_shapes_match": not mismatches,
                "mismatched_keys": mismatches[:10],
            }
        )
    if Path(cfg["motion_encoder_weight_path"]).is_file():
        state = torch.load(
            cfg["motion_encoder_weight_path"], map_location="cpu", weights_only=True, mmap=True
        )
        with torch.device("meta"):
            model = MotionEncoder()
        expected = {k: list(v.shape) for k, v in model.state_dict().items()}
        actual = {k: list(v.shape) for k, v in state.items()}
        report["models"].append(
            {
                "component": "motion_encoder",
                "checkpoint_tensors": len(actual),
                "state_keys_shapes_match": actual == expected,
            }
        )
    if Path(cfg["raft_weight_path"]).is_file():
        state = torch.load(
            cfg["raft_weight_path"], map_location="cpu", weights_only=True, mmap=True
        )
        actual = {k.removeprefix("module."): list(v.shape) for k, v in state.items()}
        with torch.device("meta"):
            model = RAFT(Namespace(small=False, mixed_precision=True, alternate_corr=False))
        expected = {k: list(v.shape) for k, v in model.state_dict().items()}
        report["models"].append(
            {
                "component": "raft_sintel",
                "checkpoint_tensors": len(actual),
                "state_keys_shapes_match": actual == expected,
            }
        )
    report["checkpoint_shapes_passed"] = all(m["state_keys_shapes_match"] for m in report["models"])
    vae_dir = Path(cfg["SD3_weight_path"]) / "vae"
    if (vae_dir / "config.json").is_file() and (
        vae_dir / "diffusion_pytorch_model.safetensors"
    ).is_file():
        with torch.device("meta"):
            vae = AutoencoderKL.from_config(AutoencoderKL.load_config(str(vae_dir)))
        expected = {k: list(v.shape) for k, v in vae.state_dict().items()}
        with safe_open(
            str(vae_dir / "diffusion_pytorch_model.safetensors"), framework="pt", device="cpu"
        ) as handle:
            actual = {k: list(handle.get_slice(k).get_shape()) for k in handle.keys()}
        mismatches = [
            k for k in expected.keys() | actual.keys() if expected.get(k) != actual.get(k)
        ]
        report["models"].append(
            {
                "component": "sd3_vae",
                "checkpoint_tensors": len(actual),
                "state_keys_shapes_match": not mismatches,
                "mismatched_keys": mismatches[:10],
            }
        )
        report["checkpoint_shapes_passed"] = all(
            m["state_keys_shapes_match"] for m in report["models"]
        )
    scheduler_dir = Path(cfg["SD3_weight_path"]) / "scheduler"
    if (scheduler_dir / "scheduler_config.json").is_file():
        scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(
            str(scheduler_dir), local_files_only=True
        )
        report["scheduler"] = {
            "class": type(scheduler).__name__,
            "num_train_timesteps": scheduler.config.num_train_timesteps,
            "timesteps_available": len(scheduler.timesteps),
            "sigmas_available": len(scheduler.sigmas),
            "cpu_load_passed": scheduler.config.num_train_timesteps > 923,
        }
    report["scheduler_ready"] = report.get("scheduler", {}).get("cpu_load_passed", False)
    report["prompt_tensors_valid"] = False
    if (
        Path(cfg["prompt_embeds_path"]).is_file()
        and Path(cfg["pooled_prompt_embeds_path"]).is_file()
    ):
        prompt = torch.load(cfg["prompt_embeds_path"], map_location="cpu", weights_only=True)
        pooled = torch.load(cfg["pooled_prompt_embeds_path"], map_location="cpu", weights_only=True)
        report["prompt_shapes"] = [list(prompt.shape), list(pooled.shape)]
        prompt_shape = list(prompt.squeeze().shape)
        report["prompt_tensors_valid"] = (
            len(prompt_shape) == 2
            and prompt_shape[0] > 0
            and prompt_shape[1] == 4096
            and list(pooled.squeeze().shape) == [2048]
            and bool(torch.isfinite(prompt).all())
            and bool(torch.isfinite(pooled).all())
        )
    report["blackwell_wheel_supported"] = "sm_120" in (report["compiled_cuda_arches"] or "")
    provenance_path = Path(cfg["prompt_embeds_path"]).parent / "empty-prompt-provenance.json"
    if provenance_path.is_file():
        report["prompt_embedding_provenance"] = str(provenance_path)
        report["author_exact_embedding_reproduction_claimed"] = False
    if check_data:
        from PIL import Image

        paths = {key: Path(cfg[key]) for key in ["lr_path", "ref_path", "ref_lr_path"]}
        missing_dirs = [str(path) for path in paths.values() if not path.is_dir()]
        report["missing_data_directories"] = missing_dirs
        data_errors = []
        count = 0
        if not missing_dirs:
            names = {
                key: {p.name for p in path.iterdir() if p.is_file()} for key, path in paths.items()
            }
            if not names["lr_path"] or any(names["lr_path"] != names[key] for key in names):
                data_errors.append(
                    "Three input directories must contain identical, nonempty filename sets."
                )
            else:
                count = len(names["lr_path"])
                scale = float(cfg["ref_res"]) / float(cfg["lr_res"])
                for name in sorted(names["lr_path"]):
                    with (
                        Image.open(paths["lr_path"] / name) as target,
                        Image.open(paths["ref_path"] / name) as ref,
                        Image.open(paths["ref_lr_path"] / name) as ref_lr,
                    ):
                        if target.size != ref_lr.size:
                            data_errors.append(f"{name}: target and LR reference sizes differ.")
                        scaled = tuple(int(side * scale) for side in target.size)
                        if ref.size != scaled:
                            data_errors.append(
                                f"{name}: HQ reference size {ref.size} differs from scaled target {scaled}."
                            )
        report["input_triplets"] = count
        report["data_errors"] = data_errors
        report["data_ready"] = not missing_dirs and not data_errors and count > 0
    report["environment_ready"] = (
        report["cpu_imports_passed"]
        and report["blackwell_wheel_supported"]
        and report["checkpoint_shapes_passed"]
        and report["wheel_cudnn_libraries_present"]
        and not report["cuda_context_initialized"]
    )
    report["model_assets_complete"] = not report["missing_assets"]
    report["cuda_context_initialized"] = torch.cuda.is_initialized()
    report["inference_ready"] = (
        report["environment_ready"]
        and normalization_disabled
        and report["model_assets_complete"]
        and report["prompt_tensors_valid"]
        and report["scheduler_ready"]
        and not report["cuda_context_initialized"]
        and report.get("data_ready", True)
    )
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--json", action="store_true")
    parser.add_argument("--config", default=str(ROOT / "config/inference_example.yml"))
    parser.add_argument("--check-data", action="store_true")
    parser.add_argument("--output")
    args = parser.parse_args()
    report = inspect(args.config, args.check_data)
    encoded = json.dumps(report, indent=2)
    if args.output:
        Path(args.output).write_text(encoded + "\n")
    print(encoded)
    return 0 if report["inference_ready"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
