"""Secondary publishing of benchmark reports (long-term archive channels such as Aone Pages).

Decoupled from the OSS primary path (:mod:`benchmark.reporting.framework.report_storage`):
this package only republishes report HTML to a long-lived URL, and failure never blocks a run.
"""

from benchmark.reporting.publish.pages_publisher import (
    ReportSiteReportPublisher,
    GitPagesReportPublisher,
    build_pages_publisher,
)

__all__ = [
    "ReportSiteReportPublisher",
    "GitPagesReportPublisher",
    "build_pages_publisher",
]
