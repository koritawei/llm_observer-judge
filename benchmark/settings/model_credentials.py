"""Named-model credential resolution for service-local model adapters."""

import os


def resolve_model_api_key(
    *,
    api_key_env: str | None,
    default_env: str,
    model_name: str,
) -> str:
    """Resolve a model API key without exposing secret values outside config code."""

    env_name = api_key_env or default_env
    api_key = os.environ.get(env_name, "")
    if not api_key:
        raise ValueError(f"Model {model_name!r} requires environment variable {env_name!r}")
    return api_key


__all__ = ["resolve_model_api_key"]
