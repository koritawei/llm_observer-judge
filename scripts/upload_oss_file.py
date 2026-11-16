#!/usr/bin/env python3
"""运维：手动上传本地文件到 OSS（object key 前缀自定义）。

复用 benchmark 链路的 RemoteConfig 存储配置（``agent-eval.benchmark.oss``：
bucket / endpoint / credential），经
:func:`benchmark.reporting.framework.report_storage.build_data_storage_client`
构造**无预置前缀**的客户端 —— object key 完全由本脚本拼接
（``{prefix}/{basename}``，或单文件时 ``--object-key`` 全量指定）。

凭证默认走 Normandy 资源绑定（``credential_mode=provider``，RemoteConfig 配置），
本地/CI 可用 ``env`` 模式；静态 AK/SK 不入仓（ST-OSS-001）。

用法::

  # 上传单个文件到自定义前缀（key = {prefix}/{basename}）
  python3 scripts/upload_oss_file.py \
      --prefix ieval-service/benchmark/benchmark_result/20260901 \
      /tmp/benchmark_final_results.csv

  # 单文件完全自定义 object key（覆盖 prefix 拼接）
  python3 scripts/upload_oss_file.py \
      --object-key misc/readme.html /tmp/readme.html

  # 多文件批量上传（共用同一 prefix）
  python3 scripts/upload_oss_file.py --prefix misc/dumps a.csv b.csv

  # 上传后额外生成 30 天预签名 GET URL（私有 bucket 可直接访问）
  python3 scripts/upload_oss_file.py --prefix misc/dumps --sign a.csv

  # dry-run（只打印上传计划，不实际上传）
  python3 scripts/upload_oss_file.py --prefix misc/dumps --dry-run a.csv

线上运行请用 scripts/upload_oss_file.sh（对齐 bazel 部署布局的 PYTHONPATH）。
"""

import argparse
import asyncio
import mimetypes
import sys
from pathlib import Path
from typing import cast

import _bootstrap  # sys.path/.env/禁 MQ，须先于 app.* 执行
from benchmark.libs.storage_ext import ReportStorageClient
from benchmark.reporting.framework.report_storage import build_data_storage_client

del _bootstrap  # 仅为副作用导入：已执行，释放模块名

#: 默认预签名 URL 有效期 — 30 天（与报告链路一致）。
_DEFAULT_SIGN_EXPIRES_SECONDS = 30 * 24 * 60 * 60


def _resolve_object_key(local_path: Path, *, prefix: str, object_key: str | None) -> str:
    """拼接目标 object key：``--object-key`` 优先，否则 ``{prefix}/{basename}``。"""
    if object_key:
        return object_key.strip().lstrip("/")
    cleaned = prefix.strip().strip("/")
    return f"{cleaned}/{local_path.name}" if cleaned else local_path.name


def _resolve_content_type(local_path: Path, explicit: str | None) -> str | None:
    """显式 ``--content-type`` 优先；否则按扩展名猜测，猜不到返回 None（OSS 默认）。"""
    if explicit:
        return explicit
    guessed, _ = mimetypes.guess_type(local_path.name)
    return guessed


async def _upload_one(
    client: ReportStorageClient,
    *,
    local_path: Path,
    object_key: str,
    content_type: str | None,
    sign: bool,
    sign_expires: int,
    dry_run: bool,
) -> bool:
    """上传单个文件；成功打印访问 URL，按需附预签名 URL。返回是否成功。"""
    size = local_path.stat().st_size
    print(f"  {local_path} -> {object_key} ({size} bytes, content_type={content_type or '(auto)'})")
    if dry_run:
        return True

    url = await client.upload(object_key, local_path.read_bytes(), content_type=content_type)
    if url is None:
        print(f"FAIL: upload returned None: {local_path} -> {object_key}", file=sys.stderr)
        return False
    print(f"OK: {url}")

    if sign:
        signed = await client.sign_url(object_key, expires_seconds=sign_expires)
        if signed is None:
            print(f"WARN: sign_url failed for {object_key}", file=sys.stderr)
        else:
            print(f"  signed_url({sign_expires}s): {signed}")
    return True


async def _main() -> int:
    parser = argparse.ArgumentParser(
        description="Upload local file(s) to OSS with a custom object key prefix. "
        "Storage config (bucket/endpoint/credential) comes from RemoteConfig "
        "agent-eval.benchmark.oss via build_data_storage_client.",
    )
    parser.add_argument("files", nargs="+", type=Path, help="Local file(s) to upload")
    parser.add_argument(
        "--prefix",
        type=str,
        default="",
        help="Object key prefix; final key = {prefix}/{basename} (default: no prefix)",
    )
    parser.add_argument(
        "--object-key",
        type=str,
        default=None,
        help="Full object key (single-file only; overrides --prefix)",
    )
    parser.add_argument(
        "--content-type",
        type=str,
        default=None,
        help="Explicit Content-Type for all files (default: guess by extension)",
    )
    parser.add_argument(
        "--sign",
        action="store_true",
        help="Also print a presigned GET URL after upload",
    )
    parser.add_argument(
        "--sign-expires",
        type=int,
        default=_DEFAULT_SIGN_EXPIRES_SECONDS,
        help=f"Presigned URL TTL seconds (default {_DEFAULT_SIGN_EXPIRES_SECONDS} = 30d)",
    )
    parser.add_argument("--dry-run", action="store_true", help="Print plan only, no upload")
    args = parser.parse_args()

    if args.object_key and len(args.files) > 1:
        parser.error("--object-key only works with a single file")

    files: list[Path] = []
    for f in args.files:
        if not f.is_file():
            print(f"ERROR: not a file: {f}", file=sys.stderr)
            return 1
        files.append(f)

    if args.dry_run:
        print("[dry-run] upload plan:")
        for f in files:
            key = _resolve_object_key(f, prefix=args.prefix, object_key=args.object_key)
            ct = _resolve_content_type(f, args.content_type)
            print(f"  {f} -> {key} (content_type={ct or '(auto)'})")
        return 0

    client = build_data_storage_client()
    if client is None:
        print(
            "ERROR: storage client unavailable — RemoteConfig agent-eval.benchmark.oss "
            "missing bucket/upload_endpoint, or libs/storage + oss2 not installed.",
            file=sys.stderr,
        )
        return 1

    failures = 0
    for f in files:
        ok = await _upload_one(
            cast(ReportStorageClient, client),
            local_path=f,
            object_key=_resolve_object_key(f, prefix=args.prefix, object_key=args.object_key),
            content_type=_resolve_content_type(f, args.content_type),
            sign=args.sign,
            sign_expires=args.sign_expires,
            dry_run=args.dry_run,
        )
        if not ok:
            failures += 1

    if failures:
        print(f"DONE with {failures}/{len(files)} failure(s)", file=sys.stderr)
        return 1
    print(f"DONE: {len(files)} file(s) uploaded")
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(_main()))
