"""
renderer/visual_types.py
────────────────────────
Concrete renderer-level types derived from the normalised
VisualState produced by Phase 3.

VisualState uses values in [0, 1] for portability.
RendererState maps those to pixel-space values the Pygame
engine and WebSocket server can use directly.

Also defines the Particle dataclass used by the particle system.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass, field
from typing import Tuple

# RGB colour alias
Color = Tuple[int, int, int]


# ─────────────────────────────────────────────────────────────
#  Colour helpers
# ─────────────────────────────────────────────────────────────

def hsl_to_rgb(h: float, s: float, l: float) -> Color:
    """
    Convert HSL (degrees, 0-1, 0-1) to RGB (0-255).
    Pure Python — no dependencies.
    """
    h = h % 360.0
    s = max(0.0, min(1.0, s))
    l = max(0.0, min(1.0, l))

    c = (1.0 - abs(2 * l - 1.0)) * s
    x = c * (1.0 - abs((h / 60.0) % 2.0 - 1.0))
    m = l - c / 2.0

    if   h < 60:  r, g, b = c, x, 0
    elif h < 120: r, g, b = x, c, 0
    elif h < 180: r, g, b = 0, c, x
    elif h < 240: r, g, b = 0, x, c
    elif h < 300: r, g, b = x, 0, c
    else:         r, g, b = c, 0, x

    return (
        int((r + m) * 255),
        int((g + m) * 255),
        int((b + m) * 255),
    )


def lerp_color(c1: Color, c2: Color, t: float) -> Color:
    """Linear interpolate between two RGB colours."""
    t = max(0.0, min(1.0, t))
    return (
        int(c1[0] + (c2[0] - c1[0]) * t),
        int(c1[1] + (c2[1] - c1[1]) * t),
        int(c1[2] + (c2[2] - c1[2]) * t),
    )


def with_alpha(color: Color, alpha: int) -> Tuple[int, int, int, int]:
    """Return RGBA tuple from RGB + alpha (0-255)."""
    return (color[0], color[1], color[2], max(0, min(255, alpha)))


# ─────────────────────────────────────────────────────────────
#  RendererState
# ─────────────────────────────────────────────────────────────

@dataclass
class RendererState:
    """
    Pixel-space rendering parameters derived from VisualState.

    All values are in concrete units (pixels, radians, RGB).
    Produced by RendererState.from_visual_state().
    """
    # colour
    primary_color: Color = (167, 139, 250)     # purple default
    secondary_color: Color = (124, 58, 237)
    background_color: Color = (8, 8, 15)

    # particles
    particle_count: int = 60
    particle_speed: float = 1.5              # px/frame
    particle_size: float = 6.0              # px radius
    particle_opacity: int = 180             # 0-255
    particle_shape: str = "circle"
    particle_turbulence: float = 0.2
    particle_lifetime: int = 90            # frames

    # geometry
    geo_ring_count: int = 3
    geo_rotation_speed: float = 0.01       # rad/frame
    geo_sides: int = 6
    geo_complexity: float = 0.4
    geo_radial_bars: bool = True
    geo_bar_height: float = 0.5
    geo_base_radius: int = 80

    # post-processing
    blur_radius: float = 0.6
    bloom_intensity: float = 0.3
    background_dim: float = 0.85           # 0=clear each frame, 1=never clear
    beat_pulse: float = 0.3

    # metadata
    emotion: str = "calm"
    confidence: float = 1.0
    is_transitioning: bool = False

    @classmethod
    def from_visual_state(
        cls,
        vs,                            # VisualState (avoid circular import)
        cfg: dict,
        canvas_w: int = 1280,
        canvas_h: int = 720,
    ) -> "RendererState":
        """
        Map normalised VisualState fields to pixel-space values
        using the visual_params ranges in config.yaml.
        """
        vp = cfg.get("visual_params", {})

        def scale(norm, lo, hi):
            return lo + norm * (hi - lo)

        p_lo, p_hi = vp.get("particle_count_range", [20, 300])
        s_lo, s_hi = vp.get("particle_speed_range", [0.5, 6.0])
        sz_lo, sz_hi = vp.get("particle_size_range", [3, 18])
        r_lo, r_hi = vp.get("geo_rotation_speed_range", [0.002, 0.04])

        primary = hsl_to_rgb(
            vs.color.hue,
            vs.color.saturation,
            vs.color.brightness,
        )
        secondary = hsl_to_rgb(
            vs.color.secondary_hue,
            vs.color.saturation * 0.8,
            vs.color.brightness * 0.9,
        )

        # Background: scale dim → alpha for trail effect
        # High dim = dark overlay = long trails
        bg_alpha = int(vs.blur_radius * 200 + 10)

        return cls(
            primary_color=primary,
            secondary_color=secondary,
            background_color=cfg.get("renderer", {}).get(
                "background_color", [8, 8, 15]
            ),
            particle_count=int(scale(vs.particles.count, p_lo, p_hi)),
            particle_speed=scale(vs.particles.speed, s_lo, s_hi),
            particle_size=scale(vs.particles.size, sz_lo, sz_hi),
            particle_opacity=int(vs.particles.opacity * 255),
            particle_shape=vs.particles.shape,
            particle_turbulence=vs.particles.turbulence,
            particle_lifetime=int(60 + vs.particles.lifetime * 120),
            geo_ring_count=vs.geometry.ring_count,
            geo_rotation_speed=scale(
                vs.geometry.rotation_speed, r_lo, r_hi
            ),
            geo_sides=vs.geometry.sides,
            geo_complexity=vs.geometry.complexity,
            geo_radial_bars=vs.geometry.radial_bars,
            geo_bar_height=vs.geometry.bar_height,
            geo_base_radius=int(min(canvas_w, canvas_h) * 0.10),
            blur_radius=vs.blur_radius,
            bloom_intensity=vs.bloom_intensity,
            background_dim=vs.background_dim,
            beat_pulse=vs.beat_pulse,
            emotion=vs.emotion,
            confidence=vs.confidence,
            is_transitioning=vs.is_transitioning,
        )


# ─────────────────────────────────────────────────────────────
#  Particle
# ─────────────────────────────────────────────────────────────

@dataclass
class Particle:
    """
    A single particle in the particle system.

    All positional values are in pixel coordinates.
    life counts up from 0; particle dies when life >= max_life.
    """
    x: float
    y: float
    vx: float
    vy: float
    size: float
    color: Color
    opacity: int          # 0-255
    shape: str
    life: int = 0
    max_life: int = 90
    angle: float = 0.0    # rotation angle (radians)
    spin: float = 0.0     # rotation speed (radians/frame)

    @property
    def alive(self) -> bool:
        return self.life < self.max_life

    @property
    def life_fraction(self) -> float:
        """0.0 = just born, 1.0 = about to die."""
        return self.life / max(1, self.max_life)

    @property
    def alpha(self) -> int:
        """Fade out in the last 30% of lifetime."""
        fade_start = 0.70
        if self.life_fraction >= fade_start:
            t = (self.life_fraction - fade_start) / (1.0 - fade_start)
            return int(self.opacity * (1.0 - t))
        return self.opacity

    def update(self, turbulence: float = 0.0) -> None:
        """Advance position, apply turbulence, age by one frame."""
        if turbulence > 0:
            self.vx += random.gauss(0, turbulence * 0.3)
            self.vy += random.gauss(0, turbulence * 0.3)
            # Dampen so turbulence doesn't accelerate indefinitely
            self.vx *= 0.98
            self.vy *= 0.98
        self.x += self.vx
        self.y += self.vy
        self.angle += self.spin
        self.life += 1

    @classmethod
    def spawn(
        cls,
        canvas_w: int,
        canvas_h: int,
        rs: RendererState,
        hue_jitter: float = 20.0,
    ) -> "Particle":
        """
        Spawn a new particle with randomised position and velocity
        based on the current RendererState.
        """
        # Random start position (edge or centre cluster depending on shape)
        cx, cy = canvas_w / 2, canvas_h / 2
        angle = random.uniform(0, 2 * math.pi)
        dist = random.uniform(0, min(canvas_w, canvas_h) * 0.4)
        x = cx + math.cos(angle) * dist
        y = cy + math.sin(angle) * dist

        # Velocity directed away from centre with speed variation
        speed = rs.particle_speed * random.uniform(0.5, 1.5)
        out_angle = angle + random.gauss(0, 0.4)
        vx = math.cos(out_angle) * speed
        vy = math.sin(out_angle) * speed

        # Colour jitter around primary
        h_base = (rs.primary_color[0] / 255.0) * 360  # rough hue from RGB
        jitter_h = random.uniform(-hue_jitter, hue_jitter)
        color = rs.primary_color   # use directly; jitter handled by hue spread

        size = rs.particle_size * random.uniform(0.6, 1.4)

        return cls(
            x=x, y=y, vx=vx, vy=vy,
            size=size,
            color=color,
            opacity=rs.particle_opacity,
            shape=rs.particle_shape,
            life=0,
            max_life=rs.particle_lifetime,
            angle=random.uniform(0, 2 * math.pi),
            spin=random.gauss(0, 0.05),
        )
