"""The single in-process snapshot slot for runtime config — a domain leaf module.

Holds only the three snapshot slots and read/write primitives, and **imports no config model**.
Readers depend on this module directly; :mod:`benchmark.domain.runtime_config_store` refreshes
from the database and forwards writes, so the store and the leaf never each keep their own copy
of state, and function bodies do not need inline imports to break a cycle.

Reader contract: slots are ``None`` until ``runtime_config_store`` finishes seeding. Callers
must fall back to their module's default empty config (this leaf cannot construct it, or it
would depend on the config models in reverse).
"""

from dataclasses import dataclass


@dataclass
class _SnapshotSlots:
    """In-process snapshot slots. Fields are replaceable so we avoid ``global``."""

    eval_file: object = None
    odps_config: object = None
    external_daily_config: object = None


_slots = _SnapshotSlots()


def current_eval_config_file() -> object:
    """Current eval-config snapshot (``None`` before seeding)."""

    return _slots.eval_file


def current_odps_export_config() -> object:
    """Current ODPS export-config snapshot (``None`` before seeding)."""

    return _slots.odps_config


def current_external_daily_config() -> object:
    """Current batch scheduled-task snapshot (``None`` before seeding)."""

    return _slots.external_daily_config


def replace_eval_config_file(value: object) -> None:
    """Replace the eval-config snapshot. Shared by seed, refresh, and tests."""

    _slots.eval_file = value


def replace_odps_export_config(value: object) -> None:
    """Replace the ODPS config snapshot. Shared by seed, refresh, and tests."""

    _slots.odps_config = value


def replace_external_daily_config(value: object) -> None:
    """Replace the scheduled-task snapshot. Shared by seed, refresh, and tests."""

    _slots.external_daily_config = value
