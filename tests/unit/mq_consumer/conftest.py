"""Make message_queue importable in unit tests — without polluting sys.path, so data_models is not shadowed.

libs/message_queue uses a flat layout (pyproject sits in the same directory as __init__.py, and the build config
is ``packages=["."]``). Its editable install flattens the modules inside the package and does not expose a ``message_queue``
top-level package. Putting the parent directory ``libs`` on ``sys.path`` would also make PathFinder resolve ``libs/data_models``
as a namespace package (.venv's ``data_models`` is provided by a meta-path finder and is not on sys.path,
so no "real package" can override that namespace directory), which breaks ``from data_models import ...``.

Here importlib registers ``message_queue`` into ``sys.modules`` directly from the source directory (setting
``submodule_search_locations`` so submodules resolve), and ``sys.path`` is not touched, so the two sides do not interfere.
In the production Docker image, ``bazel_docker_server.sh`` joins ``libs`` onto PYTHONPATH, and data_models'
explicit ``py`` directory, as a real package, takes priority over the namespace directory, which solves the same problem.
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
        pass  # Half-installed state: keep registering from the source directory

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
