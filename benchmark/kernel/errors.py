class EvalError(Exception):
    status = 400
    code = "bad_request"

    def __init__(self, message: str, *, status: int | None = None, code: str | None = None):
        super().__init__(message)
        self.message = message
        if status is not None:
            self.status = status
        if code is not None:
            self.code = code


class ConfigError(EvalError):
    status = 500
    code = "config_invalid"


class NotFound(EvalError):
    status = 404
    code = "not_found"


class Conflict(EvalError):
    status = 409
    code = "conflict"


class Unauthorized(EvalError):
    status = 401
    code = "unauthorized"


class Unavailable(EvalError):
    status = 503
    code = "unavailable"
