"""MQ readiness consecutive-failure tracker (SDD-IG-11)."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class HealthTracker:
    threshold: int = 3
    _counts: dict[str, int] = field(default_factory=dict)

    def incr(self, name: str) -> int:
        self._counts[name] = self._counts.get(name, 0) + 1
        return self._counts[name]

    def reset(self, name: str) -> None:
        self._counts[name] = 0

    def count(self, name: str) -> int:
        return self._counts.get(name, 0)

    def is_unhealthy(self, name: str) -> bool:
        return self.count(name) >= self.threshold
