"""Make message_queue importable in sidecar-ingest unit tests.

Same strategy as ``tests/unit/mq_consumer/conftest.py``: register the package from the source directory, and do not change sys.path.
"""

import importlib.util
import sys
from pathlib import Path


def _ensure_message_queue_importable() -> None:
    if "message_queue" in sys.modules:
        return
    try:
        if importlib.util.find_spec("message_queue") is not None:
            return
    except (ImportError, ValueError):
        pass

    for parent in Path(__file__).resolve().parents:
        pkg_dir = parent / "libs" / "message_queue"
        init_file = pkg_dir / "__init__.py"
        if not init_file.is_file():
            continue
        spec = importlib.util.spec_from_file_location(
            "message_queue",
            init_file,
            submodule_search_locations=[str(pkg_dir)],
        )
        if spec is None or spec.loader is None:
            return
        module = importlib.util.module_from_spec(spec)
        sys.modules["message_queue"] = module
        spec.loader.exec_module(module)
        return


_ensure_message_queue_importable()
