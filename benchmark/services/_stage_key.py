"""Stage-prefix key helper — joins (stage, logic_name) into an agent_manager registration name.

Convention: the judge / attribution stages are distinguished as ``<stage>_<logic_name>``.
"""

STAGE_JUDGE: str = "judge"
STAGE_ATTRIBUTION: str = "attribution"
STAGE_BENCHMARK: str = "benchmark"


def stage_agent_key(stage: str, logic_name: str) -> str:
    """Build an agent_manager key of the form ``<stage>_<logic_name>``."""
    return f"{stage}_{logic_name}"


__all__ = ["STAGE_JUDGE", "STAGE_ATTRIBUTION", "STAGE_BENCHMARK", "stage_agent_key"]
