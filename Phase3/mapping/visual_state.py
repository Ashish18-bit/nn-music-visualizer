"""
mapping/visual_state.py
───────────────────────
VisualState — the central data contract between the emotion
classifier (Phase 2) and the rendering engine (Phase 4).

Every visual parameter the renderer needs is stored here as a
typed, validated Python dataclass. The interpolator operates
entirely on VisualState objects, blending between them smoothly.

Structure
─────────
VisualState
  ├── ColorPalette       hue, saturation, brightness, secondary_hue
  ├── ParticleConfig     count, speed, size, shape, opacity, lifetime
  ├── GeometryConfig     complexity, ring_count, rotation_speed, sides
  └── scalars            blur_radius, bloom_intensity, transition_speed

All numeric fields are float in [0, 1] unless documented otherwise.
Hue fields are in degrees [0, 360].
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Literal, Dict, Any, List


# ─────────────────────────────────────────────────────────────
#  Sub-configs
# ─────────────────────────────────────────────────────────────

@dataclass
class ColorPalette:
    """
    HSL colour parameters for the primary and secondary palette.

    hue             : primary hue angle [0, 360]
    saturation      : colour richness [0, 1]
    brightness      : lightness [0, 1]
    secondary_hue   : accent/highlight hue [0, 360]
    hue_spread      : how far particles deviate from primary hue [0, 60]
    """
    hue: float = 265.0
    saturation: float = 0.55
    brightness: float = 0.60
    secondary_hue: float = 280.0
    hue_spread: float = 20.0

    def __post_init__(self):
        self.hue = float(self.hue) % 360.0
        self.saturation = float(max(0.0, min(1.0, self.saturation)))
        self.brightness = float(max(0.0, min(1.0, self.brightness)))
        self.secondary_hue = float(self.secondary_hue) % 360.0
        self.hue_spread = float(max(0.0, min(60.0, self.hue_spread)))

    @property
    def css_hsl(self) -> str:
        return (
            f"hsl({self.hue:.0f}, "
            f"{self.saturation * 100:.0f}%, "
            f"{self.brightness * 100:.0f}%)"
        )


@dataclass
class ParticleConfig:
    """
    Particle system parameters.

    count       : number of live particles [0, 1] normalised
    speed       : motion speed [0, 1] normalised
    size        : particle radius [0, 1] normalised
    opacity     : base opacity [0, 1]
    lifetime    : how long each particle lives [0, 1] normalised
    shape       : particle shape identifier
    turbulence  : randomness in motion direction [0, 1]
    """
    count: float = 0.30
    speed: float = 0.35
    size: float = 0.40
    opacity: float = 0.75
    lifetime: float = 0.50
    shape: str = "circle"
    turbulence: float = 0.20

    def __post_init__(self):
        for attr in ("count", "speed", "size", "opacity", "lifetime", "turbulence"):
            setattr(self, attr, float(max(0.0, min(1.0, getattr(self, attr)))))
        valid_shapes = ("circle", "star", "diamond", "spike", "drop")
        if self.shape not in valid_shapes:
            self.shape = "circle"


@dataclass
class GeometryConfig:
    """
    Procedural geometry parameters.

    complexity      : overall geometry density [0, 1]
    ring_count      : number of polygon rings [1, 8]
    rotation_speed  : how fast rings spin [0, 1] normalised
    sides           : polygon vertex count [3, 12]
    radial_bars     : whether to draw frequency radial bars
    bar_height      : radial bar height [0, 1] normalised
    """
    complexity: float = 0.40
    ring_count: int = 3
    rotation_speed: float = 0.25
    sides: int = 6
    radial_bars: bool = True
    bar_height: float = 0.50

    def __post_init__(self):
        self.complexity = float(max(0.0, min(1.0, self.complexity)))
        self.ring_count = int(max(1, min(8, self.ring_count)))
        self.rotation_speed = float(max(0.0, min(1.0, self.rotation_speed)))
        self.sides = int(max(3, min(12, self.sides)))
        self.bar_height = float(max(0.0, min(1.0, self.bar_height)))


# ─────────────────────────────────────────────────────────────
#  VisualState
# ─────────────────────────────────────────────────────────────

@dataclass
class VisualState:
    """
    Complete visual parameter state passed to the renderer each frame.

    Scalar fields
    ─────────────
    blur_radius         : background trail blur [0, 1]
    bloom_intensity     : glow effect strength [0, 1]
    transition_speed    : how fast this state blends in [0, 1]
    background_dim      : how dark the background is [0, 1]
    beat_pulse          : bass-hit reactive pulse intensity [0, 1]

    Composite fields
    ────────────────
    color       : ColorPalette
    particles   : ParticleConfig
    geometry    : GeometryConfig

    Metadata
    ────────
    emotion         : which emotion this state represents
    confidence      : model confidence for this state [0, 1]
    is_transitioning: True while interpolating between emotions
    """
    # ── composite ───────────────────────────────────────────
    color: ColorPalette = field(default_factory=ColorPalette)
    particles: ParticleConfig = field(default_factory=ParticleConfig)
    geometry: GeometryConfig = field(default_factory=GeometryConfig)

    # ── scalars ─────────────────────────────────────────────
    blur_radius: float = 0.60
    bloom_intensity: float = 0.30
    transition_speed: float = 0.50
    background_dim: float = 0.85
    beat_pulse: float = 0.40

    # ── metadata ────────────────────────────────────────────
    emotion: str = "calm"
    confidence: float = 1.0
    is_transitioning: bool = False

    def __post_init__(self):
        for attr in ("blur_radius", "bloom_intensity", "transition_speed",
                     "background_dim", "beat_pulse", "confidence"):
            setattr(self, attr, float(max(0.0, min(1.0, getattr(self, attr)))))

    # ── serialisation ────────────────────────────────────────

    def to_dict(self) -> Dict[str, Any]:
        """Flatten to a plain dict for JSON serialisation / IPC."""
        return {
            "color_hue":           self.color.hue,
            "color_saturation":    self.color.saturation,
            "color_brightness":    self.color.brightness,
            "color_secondary_hue": self.color.secondary_hue,
            "color_hue_spread":    self.color.hue_spread,
            "particle_count":      self.particles.count,
            "particle_speed":      self.particles.speed,
            "particle_size":       self.particles.size,
            "particle_opacity":    self.particles.opacity,
            "particle_lifetime":   self.particles.lifetime,
            "particle_shape":      self.particles.shape,
            "particle_turbulence": self.particles.turbulence,
            "geo_complexity":      self.geometry.complexity,
            "geo_ring_count":      self.geometry.ring_count,
            "geo_rotation_speed":  self.geometry.rotation_speed,
            "geo_sides":           self.geometry.sides,
            "geo_radial_bars":     self.geometry.radial_bars,
            "geo_bar_height":      self.geometry.bar_height,
            "blur_radius":         self.blur_radius,
            "bloom_intensity":     self.bloom_intensity,
            "transition_speed":    self.transition_speed,
            "background_dim":      self.background_dim,
            "beat_pulse":          self.beat_pulse,
            "emotion":             self.emotion,
            "confidence":          self.confidence,
            "is_transitioning":    self.is_transitioning,
        }

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), indent=2)

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "VisualState":
        return cls(
            color=ColorPalette(
                hue=d.get("color_hue", 265.0),
                saturation=d.get("color_saturation", 0.55),
                brightness=d.get("color_brightness", 0.60),
                secondary_hue=d.get("color_secondary_hue", 280.0),
                hue_spread=d.get("color_hue_spread", 20.0),
            ),
            particles=ParticleConfig(
                count=d.get("particle_count", 0.30),
                speed=d.get("particle_speed", 0.35),
                size=d.get("particle_size", 0.40),
                opacity=d.get("particle_opacity", 0.75),
                lifetime=d.get("particle_lifetime", 0.50),
                shape=d.get("particle_shape", "circle"),
                turbulence=d.get("particle_turbulence", 0.20),
            ),
            geometry=GeometryConfig(
                complexity=d.get("geo_complexity", 0.40),
                ring_count=d.get("geo_ring_count", 3),
                rotation_speed=d.get("geo_rotation_speed", 0.25),
                sides=d.get("geo_sides", 6),
                radial_bars=d.get("geo_radial_bars", True),
                bar_height=d.get("geo_bar_height", 0.50),
            ),
            blur_radius=d.get("blur_radius", 0.60),
            bloom_intensity=d.get("bloom_intensity", 0.30),
            transition_speed=d.get("transition_speed", 0.50),
            background_dim=d.get("background_dim", 0.85),
            beat_pulse=d.get("beat_pulse", 0.40),
            emotion=d.get("emotion", "calm"),
            confidence=d.get("confidence", 1.0),
            is_transitioning=d.get("is_transitioning", False),
        )

    # ── numeric vector ────────────────────────────────────────

    def to_vector(self) -> List[float]:
        """
        Export all continuous numeric parameters as a flat list.
        Used by the interpolator for vectorised blending.
        Length = 21. Hue fields normalised to [0, 1] by dividing by 360.
        """
        return [
            self.color.hue / 360.0,
            self.color.saturation,
            self.color.brightness,
            self.color.secondary_hue / 360.0,
            self.color.hue_spread / 60.0,
            self.particles.count,
            self.particles.speed,
            self.particles.size,
            self.particles.opacity,
            self.particles.lifetime,
            self.particles.turbulence,
            self.geometry.complexity,
            self.geometry.ring_count / 8.0,
            self.geometry.rotation_speed,
            self.geometry.sides / 12.0,
            self.geometry.bar_height,
            self.blur_radius,
            self.bloom_intensity,
            self.transition_speed,
            self.background_dim,
            self.beat_pulse,
        ]

    @classmethod
    def from_vector(
        cls,
        vec: List[float],
        emotion: str = "calm",
        confidence: float = 1.0,
        shape: str = "circle",
        is_transitioning: bool = False,
    ) -> "VisualState":
        """Reconstruct a VisualState from a to_vector() output."""
        v = vec
        return cls(
            color=ColorPalette(
                hue=v[0] * 360.0,
                saturation=v[1],
                brightness=v[2],
                secondary_hue=v[3] * 360.0,
                hue_spread=v[4] * 60.0,
            ),
            particles=ParticleConfig(
                count=v[5],
                speed=v[6],
                size=v[7],
                opacity=v[8],
                lifetime=v[9],
                turbulence=v[10],
                shape=shape,
            ),
            geometry=GeometryConfig(
                complexity=v[11],
                ring_count=max(1, round(v[12] * 8)),
                rotation_speed=v[13],
                sides=max(3, round(v[14] * 12)),
                bar_height=v[15],
            ),
            blur_radius=v[16],
            bloom_intensity=v[17],
            transition_speed=v[18],
            background_dim=v[19],
            beat_pulse=v[20],
            emotion=emotion,
            confidence=confidence,
            is_transitioning=is_transitioning,
        )

    def __repr__(self) -> str:
        return (
            f"VisualState(emotion={self.emotion!r}, "
            f"conf={self.confidence:.2f}, "
            f"hue={self.color.hue:.0f}deg, "
            f"particles={self.particles.count:.2f}, "
            f"speed={self.particles.speed:.2f}, "
            f"blur={self.blur_radius:.2f})"
        )
