"""Eval plugin runtime package: RemoteConfig binding resolution + local / remote adapters.

- :mod:`host` — ``resolve_plugin`` (RemoteConfig binding → registry → default) and
  ``call_plugin*`` (async call sites prefer the async variant; remote does not block the event loop)
- :mod:`class_loader` — ``module:Name`` allowlist loading (classes / functions)
- :mod:`remote_client` — unified remote HTTP contract
- :mod:`remote_parse` / :mod:`remote_adapters` — remote adapters for each extension point

This file **does not re-export**: ``host`` depends on :mod:`benchmark.domain.eval_plugin_config`, and that module
goes through ``benchmark.settings`` → ``benchmark.services`` and back into this package; a package-level re-export
makes entry points that import eval_plugin_config first (ops scripts and similar) hit a circular import. Callers
always import submodules directly,
for example ``from benchmark.plugins.host import resolve_plugin``.
"""
