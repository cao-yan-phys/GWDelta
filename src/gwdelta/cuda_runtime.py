from __future__ import annotations

import os
import sys
from pathlib import Path

_DLL_HANDLES: list[object] = []
_REGISTERED: set[str] = set()


def backend_wants_cuda(force_backend: str | None) -> bool:
    if force_backend is None:
        return False
    return "cuda" in force_backend.lower()


def _candidate_dll_dirs() -> list[Path]:
    roots: list[Path] = []
    for env_name in ("CUDA_PATH", "CUDA_PATH_V12_3", "CUDAToolkit_ROOT"):
        value = os.environ.get(env_name)
        if value:
            roots.append(Path(value) / "bin")

    roots.append(Path(r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.3\bin"))

    prefix = Path(sys.prefix)
    roots.extend([prefix, prefix / "Library" / "bin", prefix / "Scripts"])
    return roots


def ensure_cuda_dll_directories() -> None:
    if os.name != "nt":
        return

    path_parts = os.environ.get("PATH", "").split(os.pathsep)
    path_parts_norm = {str(Path(part).resolve()).lower() for part in path_parts if part}
    for directory in _candidate_dll_dirs():
        try:
            resolved = str(directory.resolve())
        except OSError:
            continue
        if directory.exists() and resolved.lower() not in path_parts_norm:
            os.environ["PATH"] = resolved + os.pathsep + os.environ.get("PATH", "")
            path_parts_norm.add(resolved.lower())
        if resolved in _REGISTERED or not directory.exists():
            continue
        _DLL_HANDLES.append(os.add_dll_directory(resolved))
        _REGISTERED.add(resolved)
