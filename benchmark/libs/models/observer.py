"""Per-model lifecycle observer bound by ``register_database``."""

from typing import Generic, TypeVar

ModelT = TypeVar("ModelT")


class ModelObserver(Generic[ModelT]):
    async def on_save(self, instance: ModelT) -> None:
        return None

    async def on_delete(self, instance: ModelT) -> None:
        return None
