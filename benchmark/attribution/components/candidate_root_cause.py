"""Compatibility re-export: the source of truth for the candidate root-cause contract lives in badcase_attribution."""

from benchmark.attribution.badcase.contracts import CandidateRootCause, CandidateSurface

__all__ = ["CandidateRootCause", "CandidateSurface"]
