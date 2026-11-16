import importlib
from dataclasses import dataclass
from typing import Any, Callable, get_args

from tortoise import Tortoise


@dataclass
class DatabaseConfig:
    environment: str
    models: list[str]
    observers: Callable[[], Any] | list[Any]
    db_url: str


async def register_database(_app, config: DatabaseConfig) -> None:
    url = config.db_url
    if url.startswith("postgresql://"):
        url = "psycopg://" + url[len("postgresql://") :]
    try:
        await Tortoise.close_connections()
    except Exception:
        pass
    await Tortoise.init(
        db_url=url,
        modules={"models": list(config.models)},
        use_tz=True,
        timezone="UTC",
    )
    _bind_observers(config)
    if config.environment == "test":
        await Tortoise.generate_schemas()


def _bind_observers(config: DatabaseConfig) -> None:
    raw = config.observers() if callable(config.observers) else list(config.observers or [])
    for name in raw:
        module = importlib.import_module(name) if isinstance(name, str) else name
        for obj in vars(module).values():
            if not isinstance(obj, type) or obj.__name__ == "ModelObserver":
                continue
            model = None
            for base in getattr(obj, "__orig_bases__", ()):
                args = get_args(base)
                if args and isinstance(args[0], type):
                    model = args[0]
                    break
            if model is None or not hasattr(model, "set_observer"):
                continue
            model.clear_observer()
            model.set_observer(obj())


async def close_database() -> None:
    await Tortoise.close_connections()
