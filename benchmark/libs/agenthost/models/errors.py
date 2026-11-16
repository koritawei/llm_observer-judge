class ProviderError(Exception):
    def __init__(self, message: str = "", *args, provider: str | None = None, status_code: int | None = None, **extra):
        super().__init__(message, *args)
        self.provider = provider
        self.status_code = status_code
        self.extra = extra


class AuthenticationError(ProviderError):
    pass


class InvalidRequestError(ProviderError):
    pass


class ModelTimeoutError(ProviderError):
    pass
