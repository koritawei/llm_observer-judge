"""api_router must mount console / runtime-config, or production assembly returns 404."""

from benchmark.routes.router import api_router


def test_api_router_includes_console_and_runtime_config() -> None:
    paths = {getattr(route, "path", "") for route in api_router.routes}
    assert "/v1/runtime-config/console/benchmark-agents" in paths
    assert "/v1/runtime-config/evaluators" in paths
    assert "/v1/runtime-config/odps-jobs" in paths
