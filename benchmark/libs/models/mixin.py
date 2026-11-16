"""Optional per-model observer slot.

Only models that mix this in expose ``get_observer``. Item tables stay plain
models so bulk writes do not grow a lifecycle hook.
"""


class ObserverMixin:
    @classmethod
    def set_observer(cls, observer: object) -> None:
        if cls.__dict__.get("_bound_observer") is not None:
            raise RuntimeError("observer already set")
        cls._bound_observer = observer

    @classmethod
    def clear_observer(cls) -> None:
        cls._bound_observer = None

    @classmethod
    def get_observer(cls):
        return cls.__dict__.get("_bound_observer")
