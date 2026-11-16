"""quiet bootstrap — disables the running healer / daily scheduler on top of :mod:`_bootstrap`.

Imported by quiet ops scripts that need a full lifespan but must not start background self-heal / scheduled triggers
(dispatch_aone_for_run / regenerate_report): lifespan creates BullMQ
workers, and healer / daily_scheduler race the script's own data operations
(for example the healer marks a run the script is handling as stale and failed), so they are explicitly turned off.
ODPS export scheduling is also disabled: the script is a short-lived process; after startup compensation grabs the execution lock it is
killed on process exit (export abandoned mid-way, done marker missing, lock left behind blocking the resident node's scheduled
retry — root cause of the 20260923 production incident); to export from a script use export_odps_to_oss.py.
"""

import os

import _bootstrap  # run the shared bootstrap first (path/.env/disable MQ)

del _bootstrap  # imported only for side effects: already executed, release the module name

os.environ.setdefault("EVAL_QUEUE_HEALER_DISABLED", "true")
os.environ.setdefault("EXTERNAL_DAILY_BENCHMARK_DISABLED", "true")
os.environ.setdefault("TRACE_DAILY_REPORT_DISABLED", "true")
os.environ.setdefault("ODPS_EXPORT_DISABLED", "true")
# benchmark streaming (Kafka realtime-stream shape) is also disabled: the script starts no background consumers.
os.environ.setdefault("BENCHMARK_STREAMING_ENABLED", "false")
# sidecar MetaQ / CMMS native libs raise boost::lock_error Abort when a short-lived script exits
# (rerun 20261103 production incident). Must override the true already injected by setenv.sh.
os.environ["SIDECAR_TRACE_ENABLED"] = "false"
