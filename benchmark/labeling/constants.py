"""Label Studio 集成共享常量。

独立模块承载常量：client / dispatcher / project_resolver 等子模块从这里导入，
避免子模块反向 import 包 ``__init__`` 形成循环依赖（``__init__`` re-export 子模块类）。
"""

#: 评测 badcase 在 LS 中的 stage 标记
STAGE_BADCASE_REVIEW = "badcase_review"

#: webhook 订阅的 LS 事件集合
WEBHOOK_ACTIONS = frozenset({"ANNOTATION_CREATED", "ANNOTATION_UPDATED"})
#: webhook 回调路径（与 API 路由注册一致）
WEBHOOK_PATH = "/api/v1/label-studio/webhook"

#: LS 分页拉取页大小
LIST_PAGE_SIZE = 1000
#: 推送 LS 时 input/output 等 JSON 字段的预览截断长度
JSON_PREVIEW_MAX_CHARS = 2000
