"""agent-eval data models — unified ORM models, observers, and database configuration.

Built on libs/data_models (Tortoise ORM), this package provides:
- ORM models (BaseModel subclasses, composed with ObserverMixin)
- lifecycle observers (ModelObserver[T], auto-discovered and registered)
- database configuration (DatabaseConfig)
- business enums (StrEnum, shared by table fields + service schemas + domain)

Data access and business logic live in :mod:`benchmark.services`.

Usage::

    from benchmark.records import TraceRecord, create_database_config
    from benchmark.records.enums import TriageLevel, TraceStatus
"""

from benchmark.records.database import create_database_config
from benchmark.records.models import TraceRecord

__all__ = [
    # Models
    "TraceRecord",
    # Database
    "create_database_config",
]
