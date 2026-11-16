"""Shared scripts bootstrap — sys.path injection + .env load + ops-script environment defaults.

Ops scripts ``import _bootstrap`` at the top (when run as ``python scripts/xxx.py``, the scripts/ directory is
``sys.path[0]``, so the module is found directly). After that,
``from app... import ...`` can be written at the top with no E402 exemption.

Side-effect order (idempotent; repeated import does not stack):

1. Insert the service root (parent of scripts/) into ``sys.path``
2. Load ``<root>/.env`` (``override=False``; do not override process env already exported)
3. ``AGENT_EVAL_MQ_ENABLED=false`` — ops scripts do not start the MetaQ consumer

Quiet scripts that also need the running healer / daily scheduler disabled should import
:mod:`_bootstrap_quiet` (adds two disable switches on top of the above).
"""

import os
import sys
from pathlib import Path

from dotenv import load_dotenv

#: services/agent-eval root (parent of scripts/).
_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

_ENV_FILE = _ROOT / ".env"
if _ENV_FILE.is_file():
    load_dotenv(_ENV_FILE, override=False)
else:
    print(f"WARN: {_ENV_FILE} not found — rely on process env", file=sys.stderr)

# Shared defaults for ops scripts: do not start the MetaQ consumer (disable healer / daily scheduler per script as needed).
os.environ.setdefault("AGENT_EVAL_MQ_ENABLED", "false")
