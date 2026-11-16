# -*- coding: utf-8 -*-
"""Self-evolution control plane. It seals candidates and never merges them."""

__all__ = ["EVOLUTION_NODES", "EvolutionService", "build_evolution_graph"]

from benchmark.evolution.graph import EVOLUTION_NODES, build_evolution_graph
from benchmark.evolution.service import EvolutionService
