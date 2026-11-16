"""Benchmark service.

``kernel`` owns scheduling, leases, and completion.
``defaults`` are the plugins shipped with the kernel.
``libs`` replaces internal platform libraries with open-source implementations.
"""

from benchmark.kernel.runtime import Kernel, open_kernel

__all__ = ["Kernel", "open_kernel"]
