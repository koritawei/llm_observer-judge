"""Send process logs to stdout in one place (deployment side → application.log).

vip-py-client attaches a ``RotatingFileHandler`` on root → VIP ``client.log``,
so a business logger with no handler of its own (including ``get_logger(__name__)`` → ``app.*``)
falls into the vip file and its warning never shows up in ``application.log``.

This module's policy: **remove FileHandlers on root and keep only stdout**. VIP/RemoteConfig and the business
share one exit; ``application.log`` already rotates. ``vipserver_client`` noise is still raised to WARNING by
``main.py``.

Level: ``LOG_LEVEL`` → ``AGENT_EVAL_LOG_LEVEL`` → INFO.
Call again after SDK init (VIP may reattach a FileHandler). See docs §5.
"""

import json
import logging
import os
import sys

from benchmark.libs.agenthost.observability.logger import get_logger

logger = get_logger("agent_eval.logging")

# Prefixes left from the era when a whitelist bound namespaces; after taking over root, drop their dedicated handlers and restore propagate.
_LEGACY_APP_NAMESPACES: tuple[str, ...] = (
    "agent_eval",
    "agentgraph",
    "agent",
    "message_queue",
    "app",
)
_APP_STDIO_MARKER = "_agent_eval_app_stdio"
_DEFAULT_LEVEL_NAME = "INFO"

# Still readable by tests / diagnostics; the meaning is now "whether root has been unified onto stdout".
_APP_LOG_NAMESPACES = _LEGACY_APP_NAMESPACES


def resolve_app_log_level() -> int:
    """LOG_LEVEL → AGENT_EVAL_LOG_LEVEL → INFO."""
    raw = (
        os.environ.get("LOG_LEVEL")
        or os.environ.get("AGENT_EVAL_LOG_LEVEL")
        or _DEFAULT_LEVEL_NAME
    )
    return getattr(logging, str(raw).strip().upper(), logging.INFO)


def _is_our_stdio_handler(handler: logging.Handler) -> bool:
    return bool(getattr(handler, _APP_STDIO_MARKER, False))


class _AppTextFormatter(logging.Formatter):
    """Append get_logger structured_fields as key=value."""

    def format(self, record: logging.LogRecord) -> str:
        base = super().format(record)
        fields = getattr(record, "structured_fields", None)
        if not fields:
            return base
        suffix = " ".join(
            f"{key}={json.dumps(value, ensure_ascii=False)}"
            for key, value in sorted(fields.items())
        )
        return f"{base} {suffix}"


def _make_app_stdio_handler() -> logging.Handler:
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(
        _AppTextFormatter("%(asctime)s %(levelname)s %(name)s %(message)s")
    )
    setattr(handler, _APP_STDIO_MARKER, True)
    return handler


def _root_has_file_handler() -> bool:
    root = logging.getLogger()
    return any(isinstance(h, logging.FileHandler) for h in root.handlers)


def _root_has_our_stdio() -> bool:
    root = logging.getLogger()
    return any(_is_our_stdio_handler(h) for h in root.handlers)


def app_namespaces_lack_stdio() -> bool:
    """True when root still needs reclaim (no stdout handler, or VIP FileHandler present)."""
    return (not _root_has_our_stdio()) or _root_has_file_handler()


def _strip_root_file_handlers() -> int:
    """Remove VIP/other FileHandlers from root. Returns count removed."""
    root = logging.getLogger()
    removed = 0
    for handler in list(root.handlers):
        if isinstance(handler, logging.FileHandler):
            root.removeHandler(handler)
            handler.close()
            removed += 1
    return removed


def _teardown_legacy_namespace_handlers() -> None:
    """Drop whitelist-era per-namespace stdout handlers; let records propagate to root.

    Only ``removeHandler``, **do not** ``close()``: these handlers' stream is
    ``sys.stdout``, and ``StreamHandler.close()`` would also ``stdout.close()``,
    so a new handler later attached to root could not write.
    """
    for name in _LEGACY_APP_NAMESPACES:
        named = logging.getLogger(name)
        for handler in list(named.handlers):
            if _is_our_stdio_handler(handler):
                named.removeHandler(handler)
        named.propagate = True
        # Do not pin a child logger to a concrete level, so a later root-level refresh is not blocked.
        if named.level != logging.NOTSET:
            named.setLevel(logging.NOTSET)


def ensure_application_logging(*, reason: str = "") -> bool:
    """Unify all logs onto stdout; strip VIP root FileHandlers.

    Returns True if this call changed handler topology (new stdout or stripped files).
    """
    # oss2 INFO errors include raw XML details, DEBUG includes signed headers/URLs.
    # Preserve only WARNING+ SDK diagnostics independently of the app's INFO/DEBUG level.
    logging.getLogger("oss2").setLevel(logging.WARNING)
    for name, sdk_logger in list(logging.Logger.manager.loggerDict.items()):
        if name.startswith("oss2.") and isinstance(sdk_logger, logging.Logger):
            sdk_logger.setLevel(logging.WARNING)
    level = resolve_app_log_level()
    needed = app_namespaces_lack_stdio()
    stripped = _strip_root_file_handlers()
    _teardown_legacy_namespace_handlers()

    root = logging.getLogger()
    root.setLevel(level)
    has_ours = False
    for handler in root.handlers:
        if _is_our_stdio_handler(handler):
            handler.setLevel(level)
            has_ours = True
    if not has_ours:
        handler = _make_app_stdio_handler()
        handler.setLevel(level)
        root.addHandler(handler)

    changed = needed or stripped > 0
    if changed:
        logger.warning(
            "unified process logs to stdout (VIP FileHandler stripped) "
            "reason=%s level=%s stripped_file_handlers=%s",
            reason or "-",
            logging.getLevelName(level),
            stripped,
        )
    return changed
