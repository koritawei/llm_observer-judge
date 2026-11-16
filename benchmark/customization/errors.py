"""Explicit failure types for business customization (no silent fallback; see design §4.7)."""


class CustomizationError(Exception):
    """Expected failure on the customization path: carries an HTTP status and a stable error code."""

    def __init__(self, status_code: int, code: str, detail: str) -> None:
        self.status_code = status_code
        self.code = code
        self.detail = detail
        super().__init__(f"{code}: {detail}")


__all__ = ["CustomizationError"]
