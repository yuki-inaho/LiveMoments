# Portable uv setup and offline inference

This optional setup targets Linux, Python 3.11, and NVIDIA GPUs supported by the pinned CUDA 13.0 PyTorch wheel. The wheel contains `sm_120` for Blackwell; the installed driver must support CUDA 13.0. The author's model architecture, six sampling steps, seed 42, and float16 inference remain unchanged. Dependencies are frozen in `uv.lock`.

## Storage and installation

Set a writable runtime directory. The default keeps the environment and large models under `/workspace`:

```sh
export LIVEMOMENTS_RUNTIME_DIR=/workspace/livemoments-runtime
just sync
just raft-code
just help
```

`just` puts the uv environment in `$LIVEMOMENTS_RUNTIME_DIR/venv`, the uv cache in `uv-cache`, and models in `models`. `HF_HOME` can be set explicitly; otherwise it uses `hf-cache` under the runtime. Nothing requires a particular home directory or checkout location.

`just raft-code` clones the official RAFT repository at `2888e15a51fa41140771d3f498ed8023cff098d1` and disables its uint8 normalization, as recommended in the LiveMoments README for inputs that are already in `[-1,1]`. It refuses an unexpected source revision or normalization implementation instead of overwriting it.

## Model access and prompt embeddings

Accept the SD3 model terms with your own Hugging Face account and authenticate locally before downloading. Never put a token in a YAML file or commit credentials.

```sh
just login
just download --include-text-encoders
```

The download command pins the author's LiveMoments checkpoint, official SD3 scheduler/VAE, optional CLIP-L/CLIP-G/T5 encoders, and official RAFT-Sintel weights. It records sizes, SHA-256 values and available upstream LFS hashes in the runtime manifest. An upstream hash is not claimed for the RAFT archive when none is published. HTTP401/403 is recorded as an authorization blockage.

The download command returns exit2 while required prompt embeddings are missing. This is expected on a new installation: the author's exact embedding tensors are not publicly distributed. Generate a documented empty prompt from the authorized official SD3 text encoders on CPU, then verify readiness:

```sh
just create-empty-prompt
just preflight --json
```

CPU generation needs at least 32GiB available host memory by default and uses float32, an empty string, and T5 length256. The resulting shapes are `[1,333,4096]` and `[1,2048]`. This is a genuine locally computed SD3 embedding, not a zero placeholder, and does not claim exact reproduction of the author's unpublished tensors. Existing tensors are never overwritten. To derive another copy, pass a new `--output-dir`.

A selected download can still report missing assets belonging to the complete inference setup; inspect `missing_assets` rather than interpreting a downloaded component as a ready pipeline.

## Inputs and configuration

Copy `config/inference_example.yml` to a local, ignored configuration and set your input directories:

```sh
cp config/inference_example.yml config/local_inference.yml
just preflight --json --config config/local_inference.yml --check-data
```

Configuration paths support `${LIVEMOMENTS_RUNTIME_DIR}` and `${LIVEMOMENTS_REPO_DIR}`. Other environment variables must resolve explicitly. Relative paths are resolved against the YAML file's directory. A missing or unresolved path is an error.

The three directories must contain identical, nonempty filename sets:

- `lr_path`: degraded target frames to restore.
- `ref_path`: corresponding high-quality reference photographs.
- `ref_lr_path`: degraded video frames corresponding to those reference photographs, renamed to match each target filename.

These inputs are not interchangeable. In particular, `ref_lr` is the reference photograph's video counterpart, not automatically the current target. Prepare image sizes explicitly: the loader scales LR by `ref_res/lr_res`, and HQ dimensions must match that scaled target. The wrapper validates dimensions and does not silently resize mismatched inputs. Set `lr_res`/`ref_res` to match your preparation.

## GPU inference

```sh
just infer --config config/local_inference.yml \
  --output-dir "$LIVEMOMENTS_RUNTIME_DIR/outputs/run-001" \
  --output-name restored
```

Use a new or empty output directory and a single directory name for `--output-name`. The wrapper checks local assets and data on CPU, then takes a nonblocking filesystem lock and checks actual free GPU memory before launching inference. It refuses insufficient memory, another cooperating inference, or existing results. It never stops unrelated jobs or switches inference to the CPU.

The default admission threshold is 28000MiB free on GPU index0; use `--gpu-index` to select another NVIDIA device. It is conservative and is not a measured peak or a guarantee for any input size. Calibrate with a small pilot before increasing resolution or sequence length. The lock coordinates processes using the same runtime parent; it cannot reserve memory against unrelated applications.

Inference forces local/offline model loading. On this pinned Linux environment, it preloads the wheel's cuDNN engine libraries after admission to avoid system/wheel engine mixtures. Help, preflight and failed admission do not preload CUDA libraries. CPU preflight validates state keys/shapes with meta-device models, SD3 scheduler/VAE, finite prompt tensors, compiled wheel support, RAFT input normalization and optional input triplets; it does not claim GPU quality validation.

## Checks and measurements

```sh
just check
just preflight --json --config config/local_inference.yml --check-data
```

The CPU tests cover runtime overrides, relative path resolution, unresolved variables, failure/busy-GPU admission, output preservation and idempotent RAFT preparation. They do not require CUDA or download models.

When benchmarking, record the input/output dimensions, reference preparation, checkpoint revisions, prompt provenance, precision, steps, seed, attention backend, GPU model, driver, wheel CUDA/cuDNN versions, timestamps and frame count. Separate startup/model load from sampling time. A sampled `nvidia-smi` process peak and `torch.cuda.max_memory_allocated()` measure different memory scopes; report the method and sampling interval instead of comparing them as the same metric. Independent frame restoration still needs temporal review for motion, occlusion and flicker.

## Reproducible synthetic smoke

Prepare a 768×768 gradient/checker pattern and a Gaussian-blurred counterpart without using any photographs:

```sh
just synthetic-inputs --out "$LIVEMOMENTS_RUNTIME_DIR/data/synthetic-001"
just infer --config "$LIVEMOMENTS_RUNTIME_DIR/data/synthetic-001/config.yml" \
  --output-dir "$LIVEMOMENTS_RUNTIME_DIR/outputs/synthetic-001" \
  --output-name restored
```

This input exercises model loading and inference, not restoration quality on real video. A local test on an RTX5090 passed: one 768×768 output, six steps, FP16, seed42, 21.017 seconds including startup/preflight/output, and a sampled own-process peak of13404MiB at a 200ms sampling interval. [Synthetic validation metadata](synthetic-validation.json) records versions, timestamps and measurement scope without user paths, GPU UUIDs, credentials, or private media. The GPU driver snapshot was taken immediately after the run; no claim is made that this is a continuous GPU profiler or a general performance guarantee.
