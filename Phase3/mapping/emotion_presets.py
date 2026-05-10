"""
mapping/emotion_presets.py
──────────────────────────
Hand-tuned VisualState presets for each emotion class.

Design rationale per emotion
─────────────────────────────
HAPPY     - warm golden hue, fast bright particles, star shapes,
            high energy, moderate geometry
SAD       - cool blue, slow drifting drops, sparse particles,
            soft blur, minimal geometry
CALM      - violet/purple, gentle circles, medium density,
            smooth blur, flowing geometry
ANGRY     - deep red/orange, fast jagged spikes, dense particles,
            sharp edges, high geometric complexity
ENERGETIC - green/teal, very fast diamonds, very dense, maximum
            radial bars, high bloom
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

from mapping.visual_state import (
    VisualState, ColorPalette, ParticleConfig, GeometryConfig
)

EMOTION_NAMES = ["happy", "sad", "calm", "angry", "energetic"]


@dataclass(frozen=True)
class EmotionPreset:
    """Associates an emotion name with its target VisualState."""
    name: str
    state: VisualState

    def __repr__(self) -> str:
        return f"EmotionPreset({self.name!r})"


# ─────────────────────────────────────────────────────────────
#  Preset definitions
# ─────────────────────────────────────────────────────────────

def _happy() -> VisualState:
    return VisualState(
        color=ColorPalette(
            hue=42.0,
            saturation=0.92,
            brightness=0.72,
            secondary_hue=28.0,
            hue_spread=25.0,
        ),
        particles=ParticleConfig(
            count=0.72,
            speed=0.65,
            size=0.50,
            opacity=0.85,
            lifetime=0.55,
            shape="star",
            turbulence=0.30,
        ),
        geometry=GeometryConfig(
            complexity=0.55,
            ring_count=4,
            rotation_speed=0.55,
            sides=5,
            radial_bars=True,
            bar_height=0.65,
        ),
        blur_radius=0.25,
        bloom_intensity=0.55,
        transition_speed=0.65,
        background_dim=0.80,
        beat_pulse=0.70,
        emotion="happy",
        confidence=1.0,
    )


def _sad() -> VisualState:
    return VisualState(
        color=ColorPalette(
            hue=218.0,
            saturation=0.55,
            brightness=0.42,
            secondary_hue=235.0,
            hue_spread=12.0,
        ),
        particles=ParticleConfig(
            count=0.22,
            speed=0.18,
            size=0.38,
            opacity=0.55,
            lifetime=0.80,
            shape="drop",
            turbulence=0.08,
        ),
        geometry=GeometryConfig(
            complexity=0.20,
            ring_count=2,
            rotation_speed=0.10,
            sides=4,
            radial_bars=False,
            bar_height=0.20,
        ),
        blur_radius=0.80,
        bloom_intensity=0.12,
        transition_speed=0.25,
        background_dim=0.92,
        beat_pulse=0.15,
        emotion="sad",
        confidence=1.0,
    )


def _calm() -> VisualState:
    return VisualState(
        color=ColorPalette(
            hue=265.0,
            saturation=0.55,
            brightness=0.60,
            secondary_hue=285.0,
            hue_spread=18.0,
        ),
        particles=ParticleConfig(
            count=0.32,
            speed=0.28,
            size=0.42,
            opacity=0.70,
            lifetime=0.65,
            shape="circle",
            turbulence=0.12,
        ),
        geometry=GeometryConfig(
            complexity=0.38,
            ring_count=3,
            rotation_speed=0.22,
            sides=6,
            radial_bars=True,
            bar_height=0.40,
        ),
        blur_radius=0.62,
        bloom_intensity=0.28,
        transition_speed=0.40,
        background_dim=0.88,
        beat_pulse=0.30,
        emotion="calm",
        confidence=1.0,
    )


def _angry() -> VisualState:
    return VisualState(
        color=ColorPalette(
            hue=5.0,
            saturation=0.96,
            brightness=0.65,
            secondary_hue=22.0,
            hue_spread=18.0,
        ),
        particles=ParticleConfig(
            count=0.90,
            speed=0.92,
            size=0.55,
            opacity=0.90,
            lifetime=0.30,
            shape="spike",
            turbulence=0.85,
        ),
        geometry=GeometryConfig(
            complexity=0.88,
            ring_count=6,
            rotation_speed=0.80,
            sides=3,
            radial_bars=True,
            bar_height=0.90,
        ),
        blur_radius=0.08,
        bloom_intensity=0.75,
        transition_speed=0.85,
        background_dim=0.72,
        beat_pulse=0.95,
        emotion="angry",
        confidence=1.0,
    )


def _energetic() -> VisualState:
    return VisualState(
        color=ColorPalette(
            hue=152.0,
            saturation=0.82,
            brightness=0.65,
            secondary_hue=170.0,
            hue_spread=30.0,
        ),
        particles=ParticleConfig(
            count=0.82,
            speed=0.80,
            size=0.45,
            opacity=0.80,
            lifetime=0.40,
            shape="diamond",
            turbulence=0.55,
        ),
        geometry=GeometryConfig(
            complexity=0.72,
            ring_count=5,
            rotation_speed=0.68,
            sides=4,
            radial_bars=True,
            bar_height=0.80,
        ),
        blur_radius=0.18,
        bloom_intensity=0.65,
        transition_speed=0.75,
        background_dim=0.75,
        beat_pulse=0.85,
        emotion="energetic",
        confidence=1.0,
    )


# ─────────────────────────────────────────────────────────────
#  Registry
# ─────────────────────────────────────────────────────────────

EMOTION_PRESETS: Dict[str, EmotionPreset] = {
    "happy":     EmotionPreset("happy",     _happy()),
    "sad":       EmotionPreset("sad",       _sad()),
    "calm":      EmotionPreset("calm",      _calm()),
    "angry":     EmotionPreset("angry",     _angry()),
    "energetic": EmotionPreset("energetic", _energetic()),
}


def get_preset(emotion: str) -> VisualState:
    """
    Return the VisualState preset for an emotion name.
    Raises KeyError for unknown emotions.
    """
    if emotion not in EMOTION_PRESETS:
        raise KeyError(
            f"Unknown emotion {emotion!r}. "
            f"Valid: {sorted(EMOTION_PRESETS.keys())}"
        )
    return EMOTION_PRESETS[emotion].state


def all_presets() -> Dict[str, VisualState]:
    """Return dict of all emotion -> VisualState presets."""
    return {name: p.state for name, p in EMOTION_PRESETS.items()}
