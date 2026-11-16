class TraceMiddleware:
    """No-op trace middleware. Callers can wrap a handler without a hosted tracer."""

    def __init__(self, *_args, **_kwargs):
        pass

    async def __call__(self, request, call_next):
        return await call_next(request)

    @staticmethod
    def get_current_node_otel_span():
        return None
