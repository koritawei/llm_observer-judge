"""Eager-import registry for benchmark coordinator services.

Importing this module registers all ``@benchmark_component`` classes with
agentgraph so ``AppRuntime`` can discover them by name.

**Static registration entry for the unified agent view** (``GET /v1/benchmarks`` walks this tuple):
a new Benchmark component must also be appended to ``BENCHMARK_SERVICES_CLASSES``, otherwise
that agent does not appear in the catalog. The full code anchors for the registration path
and the public response are in ``docs/adding-a-benchmark.md`` §0.5.
"""

from benchmark.libs.agenthost.observability.logger import get_logger

AmapAgentBenchmarkServices = None
from benchmark.runs.example.services import (
    ExampleBenchmarkServices,
)
IevalExternalBenchmarkServices = None
SearchEvalBenchmarkServices = None
load_search_v2_judge = None

logger = get_logger(__name__)

BENCHMARK_SERVICES_CLASSES = (ExampleBenchmarkServices,)

__all__ = [
    "AmapAgentBenchmarkServices",
    "BENCHMARK_SERVICES_CLASSES",
    "ExampleBenchmarkServices",
    "IevalExternalBenchmarkServices",
    "SearchEvalBenchmarkServices",
]

# Non-example benchmark services are not registered.
