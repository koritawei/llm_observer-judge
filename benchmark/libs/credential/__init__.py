"""Environment-backed credentials. Replaces the private credential SDK."""

from __future__ import annotations

import os
from types import SimpleNamespace


class names:
    @staticmethod
    def of_aliyun_kvstore_instance_id(instance_id: str) -> str:
        return instance_id

    @staticmethod
    def of_odps_account_notification_app(app_name: str) -> str:
        return app_name

    @staticmethod
    def of_aliyun_oss_bucket_name(bucket_name: str) -> str:
        return bucket_name


class helpers:
    @staticmethod
    def get_endpoint(urn: str) -> str:
        del urn
        return os.environ.get("BENCHMARK_REDIS_ENDPOINT", "127.0.0.1:6380")


class _Provider:
    def get_credential(self, urn: str):
        del urn
        return SimpleNamespace(
            username=os.environ.get("BENCHMARK_REDIS_USERNAME") or None,
            password=os.environ.get("BENCHMARK_REDIS_PASSWORD", ""),
            access_key_id=os.environ.get("BENCHMARK_STORAGE_ACCESS_KEY", ""),
            access_key_secret=os.environ.get("BENCHMARK_STORAGE_SECRET", ""),
            security_token=os.environ.get("BENCHMARK_STORAGE_TOKEN") or None,
        )


class factory:
    @staticmethod
    def get_default_credential_provider() -> _Provider:
        return _Provider()

    @staticmethod
    def get_acs_credential_provider(urn: str) -> _Provider:
        del urn
        return _Provider()
