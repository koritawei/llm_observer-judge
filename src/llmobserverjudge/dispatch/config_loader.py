"""Copy-on-Write config loader with mtime polling hot-reload."""

from __future__ import annotations

import asyncio
import logging
from pathlib import Path
from typing import Any

import yaml

from llmobserverjudge.dispatch.config_models import DispatchFileConfig, DispatchSection

logger = logging.getLogger(__name__)


class DispatchConfigLoader:
    """Holds immutable DispatchSection snapshots; failed reload keeps previous."""

    def __init__(self, path: str | Path | None = None, *, initial: DispatchSection | None = None) -> None:
        self.path = Path(path) if path else None
        self._config: DispatchSection = initial or default_dispatch_section()
        self._version: int = 1 if initial is not None else 0
        self._mtime: float | None = None
        self._stop = asyncio.Event()

    @property
    def version(self) -> int:
        return self._version

    @property
    def config(self) -> DispatchSection:
        """Callers may retain the returned reference across await (CoW)."""
        return self._config

    def load_sync(self) -> bool:
        if self.path is None:
            return False
        try:
            text = self.path.read_text(encoding="utf-8")
            raw: Any = yaml.safe_load(text)
            parsed = DispatchFileConfig.model_validate(raw)
            from llmobserverjudge.dispatch.conditions import validate_condition_syntax

            for rule in parsed.dispatch.routing_rules:
                validate_condition_syntax(rule.condition)
            self._config = parsed.dispatch
            self._version += 1
            self._mtime = self.path.stat().st_mtime
            logger.info(
                "dispatch_config_loaded path=%s version=%d rules=%d",
                self.path,
                self._version,
                len(self._config.routing_rules),
            )
            return True
        except Exception:
            logger.exception(
                "dispatch_config_reload_failed path=%s keeping version=%d",
                self.path,
                self._version,
            )
            return False

    async def load(self) -> bool:
        return await asyncio.to_thread(self.load_sync)

    async def watch_loop(self, *, interval_s: float = 1.0) -> None:
        if self.path is None:
            return
        while not self._stop.is_set():
            try:
                mtime = self.path.stat().st_mtime
                if self._mtime is None or mtime > self._mtime:
                    await self.load()
            except FileNotFoundError:
                logger.warning("dispatch_config_missing path=%s", self.path)
            try:
                await asyncio.wait_for(self._stop.wait(), timeout=interval_s)
            except TimeoutError:
                pass

    def stop(self) -> None:
        self._stop.set()


def default_dispatch_section() -> DispatchSection:
    return DispatchSection.model_validate(
        {
            "default_model": "default-judge",
            "publish_priority_shards": False,
            "routing_rules": [
                {
                    "condition": "tags contains 'safety'",
                    "judge_agents": ["judge_safety", "judge_code_eval"],
                    "priority": "P0",
                },
                {
                    "condition": "always",
                    "judge_agents": ["judge_quality", "judge_fast_vllm"],
                    "priority": "P2",
                },
            ],
            "eval_routing_rules": [
                {
                    "eval_agent_id": "judge_safety",
                    "bad_case_threshold": 0.7,
                    "attribution_agent": "attributor_default",
                    "attribution_priority": "P0",
                    "annotation_trigger": "always",
                },
                {
                    "eval_agent_id": "judge_code_eval",
                    "bad_case_threshold": 0.6,
                    "attribution_agent": "attributor_default",
                    "attribution_priority": "P0",
                    "annotation_trigger": "never",
                },
                {
                    "eval_agent_id": "judge_quality",
                    "bad_case_threshold": 0.6,
                    "attribution_agent": "attributor_default",
                    "attribution_priority": "P1",
                    "annotation_trigger": "low_confidence",
                },
            ],
        }
    )


def load_dispatch_config(path: str | Path) -> DispatchConfigLoader:
    loader = DispatchConfigLoader(path)
    ok = loader.load_sync()
    if not ok:
        raise RuntimeError(f"failed to load dispatch config: {path}")
    return loader
