#!/usr/bin/env python3
"""Derive genuine SD3 empty-prompt embeddings on CPU from authorized local weights.

The authors did not release their exact prompt tensors. This reproducible empty
prompt is a documented local choice, not a claim to reproduce author tensors.
"""

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path

os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from live_utils.config import runtime_dir

RUNTIME = runtime_dir()
os.environ.setdefault("HF_HOME", str(RUNTIME / "hf-cache"))


def file_sha(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sd3-dir", type=Path, default=RUNTIME / "models/sd3")
    parser.add_argument("--output-dir", type=Path, default=RUNTIME / "models")
    parser.add_argument("--cpu-threads", type=int, default=8)
    parser.add_argument(
        "--min-free-cpu-mib",
        type=int,
        default=32768,
        help="Minimum available host memory before loading float32 text encoders.",
    )
    args = parser.parse_args()
    if args.cpu_threads < 1 or args.min_free_cpu_mib < 1:
        parser.error("CPU threads and minimum available memory must be positive")
    required = [
        "text_encoder/model.safetensors",
        "text_encoder_2/model.safetensors",
        "text_encoder_3/model-00001-of-00002.safetensors",
        "text_encoder_3/model-00002-of-00002.safetensors",
        "text_encoder_3/model.safetensors.index.json",
        "tokenizer/tokenizer_config.json",
        "tokenizer_2/tokenizer_config.json",
        "tokenizer_3/tokenizer_config.json",
    ]
    missing = [str(args.sd3_dir / p) for p in required if not (args.sd3_dir / p).is_file()]
    if missing:
        print(
            json.dumps(
                {"status": "blocked_missing_authorized_text_encoder_assets", "missing": missing},
                indent=2,
            )
        )
        return 2
    memory_lines = Path("/proc/meminfo").read_text().splitlines()
    available_mib = next(
        int(line.split()[1]) // 1024 for line in memory_lines if line.startswith("MemAvailable:")
    )
    if available_mib < args.min_free_cpu_mib:
        print(
            json.dumps(
                {
                    "status": "blocked_low_cpu_memory",
                    "available_mib": available_mib,
                    "required_mib": args.min_free_cpu_mib,
                },
                indent=2,
            )
        )
        return 3
    import torch
    from diffusers import StableDiffusion3Pipeline
    from transformers import (
        CLIPTextModelWithProjection,
        CLIPTokenizer,
        T5EncoderModel,
        T5TokenizerFast,
    )

    torch.set_num_threads(args.cpu_threads)
    load_args = {"local_files_only": True, "torch_dtype": torch.float32, "low_cpu_mem_usage": True}
    encoder1 = CLIPTextModelWithProjection.from_pretrained(
        args.sd3_dir / "text_encoder", **load_args
    )
    encoder2 = CLIPTextModelWithProjection.from_pretrained(
        args.sd3_dir / "text_encoder_2", **load_args
    )
    encoder3 = T5EncoderModel.from_pretrained(args.sd3_dir / "text_encoder_3", **load_args)
    pipe = StableDiffusion3Pipeline(
        transformer=None,
        scheduler=None,
        vae=None,
        text_encoder=encoder1,
        text_encoder_2=encoder2,
        text_encoder_3=encoder3,
        tokenizer=CLIPTokenizer.from_pretrained(args.sd3_dir / "tokenizer", local_files_only=True),
        tokenizer_2=CLIPTokenizer.from_pretrained(
            args.sd3_dir / "tokenizer_2", local_files_only=True
        ),
        tokenizer_3=T5TokenizerFast.from_pretrained(
            args.sd3_dir / "tokenizer_3", local_files_only=True
        ),
    )
    with torch.inference_mode():
        prompt, _, pooled, _ = pipe.encode_prompt(
            prompt="",
            prompt_2="",
            prompt_3="",
            device=torch.device("cpu"),
            num_images_per_prompt=1,
            do_classifier_free_guidance=False,
            max_sequence_length=256,
        )
    if not torch.isfinite(prompt).all() or not torch.isfinite(pooled).all():
        raise RuntimeError("Prompt embeddings contain non-finite values")
    if tuple(prompt.shape) != (1, 333, 4096) or tuple(pooled.shape) != (1, 2048):
        raise RuntimeError(f"Unexpected SD3 embedding shapes: {prompt.shape}, {pooled.shape}")
    if torch.cuda.is_initialized():
        raise RuntimeError("Unexpected CUDA context initialization")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    destinations = [
        args.output_dir / "prompt_embeds.pt",
        args.output_dir / "pooled_prompt_embeds.pt",
    ]
    for dest, tensor in zip(destinations, [prompt, pooled]):
        if dest.exists():
            raise FileExistsError(f"Refusing to replace existing tensor: {dest}")
        torch.save(tensor.cpu(), dest)
    provenance = {
        "status": "locally_derived_genuine_sd3_empty_prompt",
        "author_exact_embedding_reproduction_claimed": False,
        "source_model": "stabilityai/stable-diffusion-3-medium-diffusers",
        "source_revision": "ea42f8cef0f178587cf766dc8129abd379c90671",
        "prompt": "",
        "prompt_2": "",
        "prompt_3": "",
        "max_sequence_length": 256,
        "device": "cpu",
        "dtype": "float32",
        "gpu_inference_tested": False,
        "cpu_mem_available_mib_before_load": available_mib,
        "artifacts": [
            {"path": str(dest), "sha256": file_sha(dest), "shape": list(tensor.shape)}
            for dest, tensor in zip(destinations, [prompt, pooled])
        ],
    }
    (args.output_dir / "empty-prompt-provenance.json").write_text(
        json.dumps(provenance, indent=2) + "\n"
    )
    manifest_path = RUNTIME / "model-manifest.json"
    if manifest_path.is_file() and args.output_dir.resolve() == (RUNTIME / "models").resolve():
        manifest = json.loads(manifest_path.read_text())
        manifest["sources"]["prompt_embeddings"] = provenance
        manifest["files"] = [
            f for f in manifest["files"] if f["component"] != "prompt_embeddings"
        ] + [
            {
                "component": "prompt_embeddings",
                "path": str(dest),
                "relative_path": dest.name,
                "size_bytes": dest.stat().st_size,
                "sha256": file_sha(dest),
                "upstream_lfs_verified": False,
                "locally_derived": True,
            }
            for dest in destinations
        ]
        manifest["missing_assets"] = [
            a for a in manifest.get("missing_assets", []) if not Path(a["expected_path"]).is_file()
        ]
        manifest["model_assets_complete"] = not manifest["missing_assets"]
        manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(provenance, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
