def agent_component(*_args, **_kwargs):
    def decorator(cls):
        cls.component_metadata = {"kind": "agent", "args": _args, "kwargs": _kwargs}
        cls.__agent_component__ = {"name": _kwargs.get("name"), **_kwargs}
        return cls

    return decorator
