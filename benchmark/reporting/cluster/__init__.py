"""Cluster-report agent package — L2 semantic clustering and artifact rendering.

Layers (aligned with "rules first; the LLM does not do statistics"):
- ``constants`` / ``schema`` — constants and data contracts
- ``stats/`` — L1 deterministic statistics (pure code arithmetic)
- ``agent`` — L2 LLM semantic clustering (consumes L1 results only; does not recount)
- ``render/`` — Markdown / JSON / xlsx artifact rendering
"""
