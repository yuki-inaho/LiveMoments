"""Prefer cuDNN engine libraries from this environment over system cuDNN."""

import ctypes
import importlib.util
from pathlib import Path

CUDNN_LIBS = (
    "libcudnn_graph.so.9",
    "libcudnn_engines_precompiled.so.9",
    "libcudnn_engines_runtime_compiled.so.9",
    "libcudnn_engines_tensor_ir.so.9",
    "libcudnn_heuristic.so.9",
    "libcudnn_ops.so.9",
    "libcudnn_cnn.so.9",
    "libcudnn_adv.so.9",
    "libcudnn_ext.so.9",
    "libcudnn.so.9",
)


def wheel_cudnn_dir():
    spec = importlib.util.find_spec("nvidia.cudnn")
    if spec is None or not spec.submodule_search_locations:
        raise RuntimeError("nvidia-cudnn wheel is missing")
    return (Path(next(iter(spec.submodule_search_locations))) / "lib").resolve()


def preload_wheel_cudnn():
    paths = [wheel_cudnn_dir() / name for name in CUDNN_LIBS]
    for path in paths:
        ctypes.CDLL(str(path), mode=ctypes.RTLD_GLOBAL)
    return [str(path) for path in paths]
