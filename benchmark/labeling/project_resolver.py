"""LS project 自动发现/创建 + 内存缓存。"""

import asyncio
import hashlib
import re
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

from benchmark.libs.agenthost.observability.logger import get_logger

from benchmark.labeling.client import LabelStudioClient, _RegisterWebhookSpec
from benchmark.labeling.constants import WEBHOOK_ACTIONS, WEBHOOK_PATH

# Label Studio Project.title max_length=50（API validation）。
_MAX_LS_TITLE_LEN = 50
_TITLE_HASH_LEN = 8


def read_label_config(config_path: str) -> str:
    """读取 label config XML 文件内容。"""
    return Path(config_path).read_text(encoding="utf-8")


logger = get_logger("agent_eval.label_studio.project_resolver")


@dataclass(frozen=True)
class ProjectResolverConfig:
    """``ProjectResolver`` 构造配置（FUNC-PARAMS-001 参数收敛）。"""

    callback_url: str
    webhook_secret: str | None = None
    default_label_config: str = "conf/label_studio/badcase_review.xml"
    label_config_map: dict[str, str] | None = None


@dataclass(frozen=True)
class ProjectIdentity:
    """project 归属身份（resolve / _build_title / _ensure_project 共用）。"""

    agent_name: str
    project_id: str
    dataset_name: str | None = None
    run_name: str | None = None


class ProjectResolver:
    """dict 缓存 + LS 懒发现/自动创建。

    每个 agent 一个 project，title = "{YYYYMMDD}-{project_id}-{agent_name}"。
    有 dataset_name + run_name 时改为 "{YYYYMMDD}-{dataset_name}-{run_name}"。
    超长时截断并追加短 hash，保证 ≤50 且不同输入不碰撞。
    日期前缀实现按天隔离，project_id 实现按项目隔离。
    创建 project 时按 label_config_map 路由选择 XML 模板。
    """

    _DATE_PREFIX_RE = re.compile(r"^\d{8}-")

    def __init__(
        self,
        *,
        client: LabelStudioClient,
        config: "ProjectResolverConfig",
    ) -> None:
        self._client = client
        self._callback_url = config.callback_url.rstrip("/")
        self._webhook_secret = config.webhook_secret
        self._default_label_config = config.default_label_config
        self._label_config_map: dict[str, str] = dict(config.label_config_map or {})
        self._cache: dict[str, int] = {}
        self._known_ids: set[int] = set()
        self._lock = asyncio.Lock()

    async def resolve(
        self,
        identity: "ProjectIdentity",
    ) -> int:
        project_name = self._build_title(identity)
        cached = self._cache.get(project_name)
        if cached is not None:
            return cached
        async with self._lock:
            cached = self._cache.get(project_name)
            if cached is not None:
                return cached
            project_id_resolved = await asyncio.to_thread(
                self._ensure_project,
                identity,
            )
            self._cache[project_name] = project_id_resolved
            self._known_ids.add(project_id_resolved)
            return project_id_resolved

    def is_known_project(self, project_id: int) -> bool:
        return project_id in self._known_ids

    async def warm(self) -> int:
        projects = await asyncio.to_thread(self._client.list_projects)
        count = 0
        for p in projects:
            title = p.get("title", "")
            if not self._DATE_PREFIX_RE.match(title):
                continue
            pid = p["id"]
            self._cache[title] = pid
            self._known_ids.add(pid)
            count += 1
        logger.info("ls_project_cache warmed: %d entries", count)
        return count

    def _build_title(
        self,
        identity: "ProjectIdentity",
    ) -> str:
        agent_name = identity.agent_name
        project_id = identity.project_id
        dataset_name = identity.dataset_name
        run_name = identity.run_name
        date_prefix = datetime.now().strftime("%Y%m%d")
        if dataset_name and run_name:
            title = f"{date_prefix}-{dataset_name}-{run_name}"
        else:
            title = f"{date_prefix}-{project_id}-{agent_name}"
        return self._fit_title(title)

    @staticmethod
    def _fit_title(title: str) -> str:
        """将 title 压到 LS 上限；超长时保留前缀 + 短 hash 防碰撞。"""
        if len(title) <= _MAX_LS_TITLE_LEN:
            return title
        digest = hashlib.sha1(title.encode("utf-8")).hexdigest()[:_TITLE_HASH_LEN]
        # "{prefix}-{digest}" 总长 = keep + 1 + hash_len == _MAX_LS_TITLE_LEN
        keep = _MAX_LS_TITLE_LEN - _TITLE_HASH_LEN - 1
        return f"{title[:keep]}-{digest}"

    def _resolve_label_config(self, agent_name: str) -> str:
        config_path = self._label_config_map.get(
            agent_name,
            self._default_label_config,
        )
        return read_label_config(config_path)

    def _ensure_project(
        self,
        identity: "ProjectIdentity",
    ) -> int:
        title = self._build_title(identity)
        for p in self._client.list_projects():
            if p.get("title") == title:
                self._ensure_webhook(p["id"])
                return p["id"]
        label_config = self._resolve_label_config(identity.agent_name)
        project = self._client.create_project(title, label_config)
        self._register_webhook(project["id"])
        return project["id"]

    def _ensure_webhook(self, project_id: int) -> None:
        """已有项目补齐 webhook — 幂等，URL + actions 匹配则跳过。"""
        webhook_url = f"{self._callback_url}{WEBHOOK_PATH}"
        try:
            for w in self._client.list_webhooks(project_id):
                if w.get("url") == webhook_url and set(w.get("actions", [])) == WEBHOOK_ACTIONS:
                    return
        except Exception:
            logger.warning(
                "list_webhooks failed project_id=%s, will attempt register",
                project_id,
                exc_info=True,
            )
        self._register_webhook(project_id)

    def _register_webhook(self, project_id: int) -> None:
        webhook_url = f"{self._callback_url}{WEBHOOK_PATH}"
        headers = (
            {"Authorization": f"Bearer {self._webhook_secret}"} if self._webhook_secret else None
        )
        self._client.register_webhook(
            _RegisterWebhookSpec(
                project_id=project_id,
                url=webhook_url,
                actions=list(WEBHOOK_ACTIONS),
                send_payload=True,
                headers=headers,
            ),
        )
