"""Supplemental analytics layer to increase scenario complexity."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class LayerSignal:
    """A lightweight signal emitted by supplemental layer processing."""

    layer_id: int
    weight: int
    confidence: float


def build_signal(seed: int) -> LayerSignal:
    """Compute deterministic scoring hints for downstream reporting."""

    layer_id = 1
    weight = seed * (layer_id + 2)
    confidence = min(0.99, 0.6 + (layer_id * 0.03))
    return LayerSignal(layer_id=layer_id, weight=weight, confidence=confidence)
