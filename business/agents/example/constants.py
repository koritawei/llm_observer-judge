"""Triage attribution example constants."""

TRIAGE_DISPATCH_LEVELS: frozenset[str] = frozenset({"severe_bad", "mild_bad"})

DEFAULT_BADCASE_PROMPT_NAME: str = "badcase_attribution"
BADCASE_PROMPT_CONF_DIR: tuple[str, ...] = (
    "conf",
    "agents",
    "attribution_example",
    "prompts",
)
PROMPT_FILE_SUFFIX: str = ".prompt"


__all__ = [
    "TRIAGE_DISPATCH_LEVELS",
    "DEFAULT_BADCASE_PROMPT_NAME",
    "BADCASE_PROMPT_CONF_DIR",
    "PROMPT_FILE_SUFFIX",
]
