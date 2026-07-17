"""Multi-tenant / multi-project constants and helpers."""

from __future__ import annotations

import re
from dataclasses import dataclass
from uuid import uuid4

DEFAULT_PROJECT_ID = "default"
DEFAULT_PROJECT_NAME = "default"

_SLUG_RE = re.compile(r"[^a-zA-Z0-9_-]+")


def normalize_project_name(name: str) -> str:
    text = (name or "").strip()
    return text or DEFAULT_PROJECT_NAME


def slugify_project_name(name: str) -> str:
    base = _SLUG_RE.sub("-", normalize_project_name(name)).strip("-").lower()
    return base or "project"


def new_project_id(name: str | None = None) -> str:
    """Stable project id. Display name uniqueness is enforced per tenant separately."""
    if name and normalize_project_name(name) == DEFAULT_PROJECT_NAME:
        return DEFAULT_PROJECT_ID
    slug = slugify_project_name(name or "project")[:48]
    return f"{slug}-{uuid4().hex[:8]}"


@dataclass(frozen=True)
class Project:
    project_id: str
    tenant_id: str
    name: str

    def to_dict(self) -> dict[str, str]:
        return {
            "project_id": self.project_id,
            "tenant_id": self.tenant_id,
            "name": self.name,
        }
