"""mapping — emotion-to-visual parameter mapping package."""

from mapping.visual_state import VisualState, ColorPalette, ParticleConfig, GeometryConfig
from mapping.emotion_presets import EMOTION_PRESETS, get_preset, EmotionPreset
from mapping.interpolator import StateInterpolator, EMAInterpolator, ConfidenceBlender
from mapping.mapper import EmotionToVisualMapper

__all__ = [
    "VisualState", "ColorPalette", "ParticleConfig", "GeometryConfig",
    "EMOTION_PRESETS", "get_preset", "EmotionPreset",
    "StateInterpolator", "EMAInterpolator", "ConfidenceBlender",
    "EmotionToVisualMapper",
]
