"""Service-level filesystem paths (conf dir / per-env app config).

Lightweight module (depends only on pathlib): so modules such as attribution_dispatch can import it at the top
without pulling evaluation's heavy dependency chain (agentgraph / fastapi / langfuse).
"""

from pathlib import Path

#: ``app/paths.py`` → parents[1] == service root (``services/agent-eval``).
_SERVICE_ROOT = Path(__file__).resolve().parents[1]


def default_conf_dir() -> Path:
    """Conf directory at the service root (``services/agent-eval/conf``)."""
    return _SERVICE_ROOT / "conf"


def app_config_path(env: str = "local") -> Path:
    """Local runtime config. Environment-specific copies are not kept."""
    del env
    runtime = default_conf_dir() / "runtime.local.yaml"
    if runtime.is_file():
        return runtime
    return default_conf_dir() / "app.local.yaml"


__all__ = ["app_config_path", "default_conf_dir"]
