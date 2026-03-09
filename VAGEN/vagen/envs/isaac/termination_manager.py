from __future__ import annotations

import importlib.util
import sys
from pathlib import Path


def _load_impl_module():
    module_name = "_image_bricks_assembling_termination_manager"
    cached = sys.modules.get(module_name)
    if cached is not None:
        return cached

    repo_root = Path(__file__).resolve().parents[4]
    module_path = (
        repo_root
        / "IsaacLab"
        / "source"
        / "isaaclab_tasks"
        / "isaaclab_tasks"
        / "manager_based"
        / "manipulation"
        / "assembling"
        / "termination_manager.py"
    )
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Failed to load assembling termination module from {module_path}")

    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


_impl = _load_impl_module()

IsaacTerminationConfig = _impl.IsaacTerminationConfig
IsaacTerminationManager = _impl.IsaacTerminationManager
TerminationStatus = _impl.TerminationStatus

__all__ = [
    "IsaacTerminationConfig",
    "IsaacTerminationManager",
    "TerminationStatus",
]
