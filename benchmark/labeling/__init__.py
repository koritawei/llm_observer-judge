"""Label Studio 集成服务子包。

共享常量见 ``benchmark.labeling.constants``（独立模块避免循环依赖）。
"""

from benchmark.labeling.case_review_hook import LsCaseReviewAfterRunHook
from benchmark.labeling.client import LabelStudioClient
from benchmark.labeling.dispatcher import LabelStudioDispatcher
from benchmark.labeling.project_resolver import ProjectResolver
from benchmark.labeling.webhook_service import LabelStudioWebhookService

__all__ = [
    "LabelStudioClient",
    "LabelStudioDispatcher",
    "LabelStudioWebhookService",
    "LsCaseReviewAfterRunHook",
    "ProjectResolver",
]
