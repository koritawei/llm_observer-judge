"""``@agent_component`` registration entry for cluster_report.

Import this module to trigger the registration side effect of :class:`ClusterReportAgent`
so ``ProviderRegistry.discover()`` can scan it — the same mechanism as
``business.evaluators.registry`` / ``business.agents.registry``.

**Without importing this module, AgentManager never discovers ``cluster_report_agent``**,
and `runtime.agent_manager.get("cluster_report_agent")` fails.
"""

from benchmark.reporting.cluster.agent import ClusterReportAgent  # noqa: F401 - registers @agent_component

__all__ = ["ClusterReportAgent"]
