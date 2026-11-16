"""Label Studio REST API client.

Wraps the Label Studio HTTP API for project management, task import,
webhook registration, and annotation retrieval.  No Label Studio SDK
dependency — plain ``httpx`` calls.
"""

from dataclasses import dataclass
from typing import Any

import httpx  # noqa: TID251 — 本模块即 Label Studio HTTP client 封装层自身（libs/api_client 无 LS 封装可复用）
from benchmark.libs.agenthost.observability.logger import get_logger

from benchmark.labeling.constants import LIST_PAGE_SIZE

logger = get_logger("agent_eval.label_studio_client")

# HTTP 状态码常量（对齐 trace_resolver.langfuse_client 的 _HTTP_OK 模式）
_HTTP_OK = 200
_HTTP_FOUND = 302
_HTTP_BAD_REQUEST = 400


class LabelStudioImportTimeoutError(Exception):
    """Raised when polling for import task IDs exceeds max retries."""


class LabelStudioImportFailedError(Exception):
    """Raised when the import job reports ``status=failed``."""


class LabelStudioAuthError(Exception):
    """Raised when compat-mode login fails (wrong credentials or LS unreachable)."""


@dataclass(frozen=True)
class LabelStudioAuth:
    """LS 认证配置：``token`` 模式用 ``api_key``；``compat`` 模式用 ``email``+``password``。"""

    api_key: str | None = None
    mode: str = "token"
    email: str | None = None
    password: str | None = None


@dataclass(frozen=True)
class _RequestOptions:
    """``_request`` 可选载荷（json body / query params）。"""

    json: Any = None
    params: dict[str, Any] | None = None


@dataclass(frozen=True)
class _RegisterWebhookSpec:
    """``register_webhook`` 入参（FUNC-PARAMS-001）。"""

    project_id: str | int
    url: str
    actions: list[str]
    headers: dict[str, str] | None = None
    send_payload: bool = True


class LabelStudioClient:
    """Label Studio REST API client.

    Supports two authentication modes:

    * ``token`` (default) — ``Authorization: Token <api_key>`` header.
    * ``compat`` — session-cookie login for LS >= 1.24 Community editions
      that disabled legacy token auth.  Requires ``email`` and ``password``.
    """

    def __init__(
        self,
        base_url: str,
        auth: LabelStudioAuth,
        timeout: float = 10.0,
    ) -> None:
        self._base_url = base_url.rstrip("/")
        self._api_key = auth.api_key
        self._timeout = timeout
        self._auth_mode = auth.mode
        self._session_cookies: httpx.Cookies | None = None
        self._csrf_token: str | None = None
        self._email = auth.email
        self._password = auth.password
        self._http = httpx.Client(timeout=self._timeout)

        if auth.mode == "compat":
            if not auth.email or not auth.password:
                raise ValueError("compat auth_mode requires email and password")
            self._login(auth.email, auth.password)

    def _login(self, email: str, password: str) -> None:
        with httpx.Client(timeout=self._timeout, follow_redirects=False) as http:
            get_resp = http.get(f"{self._base_url}/user/login")
            csrf = get_resp.cookies.get("csrftoken", "")
            login_resp = http.post(
                f"{self._base_url}/user/login",
                data={
                    "csrfmiddlewaretoken": csrf,
                    "email": email,
                    "password": password,
                },
                headers={
                    "Referer": f"{self._base_url}/user/login",
                    "Content-Type": "application/x-www-form-urlencoded",
                },
                cookies=get_resp.cookies,
            )
            # Django 登录成功 → 302 重定向；失败 → 200 重渲染表单（无 sessionid）。
            if login_resp.status_code == _HTTP_FOUND:
                pass  # 明确成功
            elif login_resp.status_code == _HTTP_OK and "sessionid" in login_resp.cookies:
                pass  # 某些 LS 配置 200 + session cookie 也算成功
            else:
                location = login_resp.headers.get("location", "")
                raise LabelStudioAuthError(
                    f"compat login failed: HTTP {login_resp.status_code}, "
                    f"sessionid={'sessionid' in login_resp.cookies}"
                    + (f", location={location}" if location else "")
                )
            merged = httpx.Cookies()
            merged.update(get_resp.cookies)
            merged.update(login_resp.cookies)
            self._session_cookies = merged
            self._csrf_token = merged.get("csrftoken", csrf)
        logger.info("compat login ok base_url=%s", self._base_url)

    def close(self) -> None:
        self._http.close()

    def _headers(self) -> dict[str, str]:
        if self._auth_mode == "compat" and self._csrf_token:
            return {"X-CSRFToken": self._csrf_token}
        return {"Authorization": f"Token {self._api_key}"}

    def _url(self, path: str) -> str:
        return f"{self._base_url}{path}"

    def _request(
        self,
        method: str,
        path: str,
        options: "_RequestOptions | None" = None,
    ) -> httpx.Response:
        json = options.json if options else None
        params = options.params if options else None
        resp = self._http.request(
            method,
            self._url(path),
            headers=self._headers(),
            cookies=self._session_cookies,
            json=json,
            params=params,
        )
        if (
            resp.status_code in (401, 403)
            and self._auth_mode == "compat"
            and self._email
            and self._password
        ):
            logger.info("compat session expired, re-login base_url=%s", self._base_url)
            self._login(self._email, self._password)
            resp = self._http.request(
                method,
                self._url(path),
                headers=self._headers(),
                cookies=self._session_cookies,
                json=json,
                params=params,
            )
        if resp.status_code >= _HTTP_BAD_REQUEST:
            logger.warning(
                "ls_client: %s %s → %s body=%s",
                method,
                path,
                resp.status_code,
                resp.text[:500],
            )
        resp.raise_for_status()
        return resp

    def create_project(
        self,
        title: str,
        label_config: str,
        maximum_annotations: int = 1,
    ) -> dict[str, Any]:
        resp = self._request(
            "POST",
            "/api/projects/",
            options=_RequestOptions(
                json={
                    "title": title,
                    "label_config": label_config,
                    "maximum_annotations": maximum_annotations,
                },
            ),
        )
        result = resp.json()
        logger.info(
            "ls_client.create_project project_id=%s title=%s",
            result.get("id"),
            title,
        )
        return result

    def list_projects(self) -> list[dict[str, Any]]:
        """分页拉取全部项目（LS 默认每页 30 条）。"""
        all_projects: list[dict[str, Any]] = []
        page = 1
        while True:
            resp = self._request(
                "GET",
                "/api/projects/",
                options=_RequestOptions(params={"page_size": LIST_PAGE_SIZE, "page": page}),
            )
            data = resp.json()
            if isinstance(data, dict):
                batch = data.get("results", [])
            else:
                batch = data
            all_projects.extend(batch)
            if len(batch) < LIST_PAGE_SIZE:
                break
            page += 1
        return all_projects

    def import_tasks(
        self,
        project_id: str | int,
        tasks: list[dict[str, Any]],
    ) -> dict[str, Any]:
        logger.debug(
            "ls_client.import_tasks project_id=%s task_count=%s",
            project_id,
            len(tasks),
        )
        resp = self._request(
            "POST",
            f"/api/projects/{project_id}/import",
            options=_RequestOptions(
                json=tasks,
                params={"return_task_ids": "true"},
            ),
        )
        return resp.json()

    def list_webhooks(self, project_id: int) -> list[dict[str, Any]]:
        resp = self._request(
            "GET",
            "/api/webhooks/",
            options=_RequestOptions(params={"project": project_id}),
        )
        return resp.json()

    def register_webhook(
        self,
        spec: "_RegisterWebhookSpec",
    ) -> dict[str, Any]:
        project_id = spec.project_id
        url = spec.url
        actions = spec.actions
        headers = spec.headers
        send_payload = spec.send_payload
        body: dict[str, Any] = {
            "project": project_id,
            "url": url,
            "send_payload": send_payload,
            "send_for_all_actions": False,
            "actions": actions,
        }
        if headers:
            body["headers"] = headers
        resp = self._request("POST", "/api/webhooks/", options=_RequestOptions(json=body))
        result = resp.json()
        logger.info(
            "ls_client.register_webhook project_id=%s webhook_id=%s url=%s",
            project_id,
            result.get("id"),
            url,
        )
        return result
