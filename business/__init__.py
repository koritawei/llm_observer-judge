"""User-authored plugins.

Profiles and default judges live in ``benchmark``. This package is only for
plugins a deployment adds itself. They return decisions; the kernel commits them.
"""

from business.attribution import RecordingAttribution

__all__ = ["RecordingAttribution"]
