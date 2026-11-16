"""Retry classification for callback error text — shared by the benchmark callback contract and queue submit classification.

``is_retryable_callback_error`` was originally defined in
:mod:`benchmark.runs.schema` (callback schema layer). Because
:mod:`benchmark.queue.submit_failure` (queue infrastructure, which must not depend on business packages)
needs the same classification, it was moved down to utils for both sides. Callers:
- ``JudgeCallback.is_retryable`` / ``AttributionCallback.is_retryable``
  (:mod:`benchmark.runs.schema`)
- :mod:`benchmark.queue.submit_failure` (submit / callback failure classification)
"""

#: Callback error text — explicitly non-retryable (resubmitting the same payload does not change the result)
_NON_RETRYABLE_CALLBACK_MARKERS: tuple[str, ...] = (
    "适配后全部 response 为空",
    "请检查 enrich 命中",
    "enrich 命中",
    "not configured",
    "validationerror",
    "authentication",
    "unauthorized",
    "forbidden",
    "invalid request",
    "bad request",
    "permission denied",
    # judge false success: no error but criterion_scores is empty (ieval_external-20260909:58)
    "empty_criterion_scores",
)

#: Callback error text — explicitly retryable (transient failure)
_RETRYABLE_CALLBACK_MARKERS: tuple[str, ...] = (
    "timeout",
    "timed out",
    "timeouterror",
    "connection",
    "temporarily",
    "unavailable",
    "429",
    "rate limit",
    "too many requests",
    "502",
    "503",
    "504",
    "500",
    "internal server",
    "bad gateway",
    "service unavailable",
    "gateway timeout",
    "throttl",
    "celery",
    "worker lost",
    "redis",
)


def is_retryable_callback_error(error: str | None) -> bool:
    """Decide whether an ieval judge callback ``error`` is retryable (design §8.3).

    - No error → not a failure (the caller should take the success path)
    - Hits a non-retryable marker (enrich miss / auth / params) → False → force_fail
    - Hits a retryable marker (timeout / 429 / 5xx / connection) → True → running→wait
    - Unknown text: conservative True (still bounded by attempt count and time window)
    """
    if not error or not str(error).strip():
        return False
    text = str(error)
    low = text.lower()
    for marker in _NON_RETRYABLE_CALLBACK_MARKERS:
        if marker.lower() in low or marker in text:
            return False
    for marker in _RETRYABLE_CALLBACK_MARKERS:
        if marker.lower() in low:
            return True
    return True
