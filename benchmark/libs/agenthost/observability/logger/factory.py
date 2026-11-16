import logging


class LazyLoggerProxy:
    def __init__(self, name: str):
        self._name = name

    def __getattr__(self, item: str):
        return getattr(logging.getLogger(self._name), item)
