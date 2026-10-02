set shell := ["sh", "-eu", "-c"]
set export
runtime := env("LIVEMOMENTS_RUNTIME_DIR", "/workspace/livemoments-runtime")
export LIVEMOMENTS_RUNTIME_DIR := runtime
export UV_PROJECT_ENVIRONMENT := runtime + "/venv"
export UV_CACHE_DIR := runtime + "/uv-cache"
export UV_PYTHON_INSTALL_DIR := runtime + "/python"
export HF_HOME := env("HF_HOME", runtime + "/hf-cache")

sync:
    uv sync --frozen

login:
    uv run --frozen hf auth login

raft-code:
    uv run --frozen --offline python scripts/prepare_raft.py

preflight *args:
    CUDA_VISIBLE_DEVICES='' HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 uv run --frozen --offline python scripts/preflight.py {{args}}

infer *args:
    uv run --frozen --offline python scripts/run_inference.py {{args}}

download *args:
    CUDA_VISIBLE_DEVICES='' uv run --frozen python scripts/download_models.py {{args}}

create-empty-prompt *args:
    CUDA_VISIBLE_DEVICES='' HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 uv run --frozen --offline python scripts/create_empty_prompt_embeddings.py {{args}}

help:
    uv run --frozen --offline python scripts/run_inference.py --help

check:
    CUDA_VISIBLE_DEVICES='' uv run --frozen --offline python -m unittest discover -s tests -v

synthetic-inputs *args:
    CUDA_VISIBLE_DEVICES='' uv run --frozen --offline python scripts/synthetic_smoke_inputs.py {{args}}
