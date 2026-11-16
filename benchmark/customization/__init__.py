"""Business prompt/skill customization: Git content-repo read/write and bundle materialization.

The source of truth is the Git repo ``agent_eval_prompt_skill`` (read by commit sha,
never by branch); this package handles upload validation (manifest), atomic commit (git_store),
materialization by ref (bundle_store), and upload orchestration (service).
"""

from benchmark.customization.errors import CustomizationError

__all__ = ["CustomizationError"]
