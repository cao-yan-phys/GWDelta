"""Runtime helpers for local CUDA backend extension modules on Windows."""

from __future__ import annotations

import os
import sys
from pathlib import Path

_DLL_HANDLES: list[object] = []
_REGISTERED: set[str] = set()


def _discover_cuda_toolkits() -> list[Path]:
    """Return likely CUDA toolkit roots, preferring explicit environment hints."""

    roots: list[Path] = []
    for env_name in ("CUDA_PATH", "CUDA_PATH_V12_3", "CUDAToolkit_ROOT"):
        value = os.environ.get(env_name)
        if value:
            roots.append(Path(value).expanduser())

    base = Path(r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA")
    if base.exists():
        versions = [path for path in base.iterdir() if path.is_dir() and path.name.lower().startswith("v")]
        versions.sort(key=lambda path: path.name.lower(), reverse=True)
        roots.extend(versions)

    unique: list[Path] = []
    seen: set[str] = set()
    for root in roots:
        try:
            resolved = str(root.resolve())
        except OSError:
            continue
        key = resolved.lower()
        if key in seen:
            continue
        seen.add(key)
        unique.append(root)
    return unique


def _ensure_cuda_env() -> Path | None:
    """Populate CUDA-related environment variables when they are missing."""

    if os.name != "nt":
        return None

    discovered = _discover_cuda_toolkits()
    for root in discovered:
        if not root.exists():
            continue
        try:
            resolved = str(root.resolve())
        except OSError:
            continue
        os.environ.setdefault("CUDA_PATH", resolved)
        os.environ.setdefault("CUDA_HOME", resolved)
        os.environ.setdefault("CUPY_CUDA_PATH", resolved)
        os.environ.setdefault("CUDAToolkit_ROOT", resolved)
        if "12.3" in root.name:
            os.environ.setdefault("CUDA_PATH_V12_3", resolved)
        return root
    return None


def backend_wants_cuda(force_backend: str | None) -> bool:
    """Return whether a FastLISAResponse/lisatools backend name is CUDA-like."""

    if force_backend is None:
        return False
    return "cuda" in force_backend.lower()


def _candidate_dll_dirs() -> list[Path]:
    roots: list[Path] = []
    for env_name in ("CUDA_PATH", "CUDA_PATH_V12_3", "CUDAToolkit_ROOT"):
        value = os.environ.get(env_name)
        if value:
            roots.append(Path(value) / "bin")

    for root in _discover_cuda_toolkits():
        roots.extend([root / "bin", root / "libnvvm" / "bin", root / "lib" / "x64", root / "lib"])

    prefix = Path(sys.prefix)
    roots.extend([prefix, prefix / "Library" / "bin", prefix / "Scripts"])
    return roots


def ensure_cuda_dll_directories() -> None:
    """Register CUDA and conda DLL directories for Python extension imports.

    Python 3.8+ can fail to load CUDA-backed ``.pyd`` files on Windows even when
    the directories are present in ``PATH``.  ``os.add_dll_directory`` makes the
    CUDA runtime and conda runtime DLLs visible to extension-module imports.
    """

    if os.name != "nt":
        return

    resolved_root = _ensure_cuda_env()
    if resolved_root is not None:
        preferred = [
            resolved_root / "bin",
            resolved_root / "libnvvm" / "bin",
            resolved_root / "lib" / "x64",
            resolved_root / "lib",
        ]
        preferred_resolved: list[str] = []
        preferred_norm: set[str] = set()
        for candidate in preferred:
            if not candidate.exists() or not candidate.is_dir():
                continue
            resolved = str(candidate.resolve())
            preferred_resolved.append(resolved)
            preferred_norm.add(resolved.lower())
        existing_parts: list[str] = []
        for part in os.environ.get("PATH", "").split(os.pathsep):
            if not part:
                continue
            try:
                key = str(Path(part).resolve()).lower()
            except OSError:
                key = part.lower()
            if key not in preferred_norm:
                existing_parts.append(part)
        os.environ["PATH"] = os.pathsep.join(preferred_resolved + existing_parts)

    path_parts = os.environ.get("PATH", "").split(os.pathsep)
    path_parts_norm = {str(Path(part).resolve()).lower() for part in path_parts if part}

    for directory in _candidate_dll_dirs():
        try:
            resolved = str(directory.resolve())
        except OSError:
            continue
        if directory.exists() and resolved.lower() not in path_parts_norm:
            current_path = os.environ.get("PATH", "")
            os.environ["PATH"] = current_path + os.pathsep + resolved if current_path else resolved
            path_parts_norm.add(resolved.lower())
        if resolved in _REGISTERED or not directory.exists():
            continue
        _DLL_HANDLES.append(os.add_dll_directory(resolved))
        _REGISTERED.add(resolved)
