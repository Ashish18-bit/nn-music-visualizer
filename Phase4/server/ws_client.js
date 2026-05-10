/**
 * server/ws_client.js
 * ───────────────────
 * Browser-side WebSocket client for the NN Music Visualizer.
 *
 * Connects to the FastAPI WebSocket server (ws_server.py) and
 * renders the incoming VisualState stream using the HTML Canvas API.
 *
 * Features
 * ────────
 * - Auto-reconnect with exponential backoff
 * - Particle system (up to 300 particles)
 * - Procedural polygon rings
 * - Radial frequency bars
 * - Background trail / bloom effect
 * - Emotion label HUD
 *
 * Usage
 * ─────
 * Include in an HTML file:
 *   <canvas id="viz"></canvas>
 *   <script src="ws_client.js"></script>
 *
 * Or set WS_URL before including:
 *   <script>window.WS_URL = "ws://192.168.1.10:8765/ws";</script>
 *   <script src="ws_client.js"></script>
 */

"use strict";

// ── Config ───────────────────────────────────────────────────
const WS_URL = window.WS_URL || `ws://${location.hostname}:8765/ws`;
const MAX_PARTICLES = 300;
const TARGET_FPS    = 60;

// ── Canvas setup ─────────────────────────────────────────────
const canvas = document.getElementById("viz") || (() => {
  const c = document.createElement("canvas");
  document.body.appendChild(c);
  document.body.style.margin = "0";
  document.body.style.background = "#08080f";
  return c;
})();
const ctx = canvas.getContext("2d");

function resize() {
  canvas.width  = window.innerWidth;
  canvas.height = window.innerHeight;
}
resize();
window.addEventListener("resize", resize);

// ── State ─────────────────────────────────────────────────────
let state = {
  frame: 0,
  emotion: "calm",
  confidence: 1.0,
  is_transitioning: false,
  primary_color:   { r: 167, g: 139, b: 250 },
  secondary_color: { r: 124, g: 58,  b: 237 },
  particle_count:  60,
  particle_speed:  1.5,
  particle_size:   6,
  particle_opacity: 180,
  particle_shape:  "circle",
  particle_turbulence: 0.2,
  geo_ring_count:  3,
  geo_rotation_speed: 0.01,
  geo_sides:       6,
  geo_complexity:  0.4,
  geo_radial_bars: true,
  geo_bar_height:  0.5,
  blur_radius:     0.6,
  bloom_intensity: 0.3,
  background_dim:  0.85,
  beat_pulse:      0.3,
};

// Ring rotation angles
const ringAngles = new Array(8).fill(0);

// ── Particles ─────────────────────────────────────────────────
class Particle {
  constructor(s) { this.reset(s); }

  reset(s) {
    const cx = canvas.width / 2, cy = canvas.height / 2;
    const angle = Math.random() * Math.PI * 2;
    const dist  = Math.random() * Math.min(canvas.width, canvas.height) * 0.4;
    this.x = cx + Math.cos(angle) * dist;
    this.y = cy + Math.sin(angle) * dist;
    const speed = s.particle_speed * (0.5 + Math.random());
    const outAngle = angle + (Math.random() - 0.5) * 0.8;
    this.vx = Math.cos(outAngle) * speed;
    this.vy = Math.sin(outAngle) * speed;
    this.size   = s.particle_size * (0.6 + Math.random() * 0.8);
    this.life   = 0;
    this.maxLife = 60 + s.particle_opacity / 255 * 120 | 0;
    this.angle  = Math.random() * Math.PI * 2;
    this.spin   = (Math.random() - 0.5) * 0.1;
    this.shape  = s.particle_shape;
    this.opacity = s.particle_opacity;
    this.r = s.primary_color.r;
    this.g = s.primary_color.g;
    this.b = s.primary_color.b;
    this.alive = true;
  }

  update(s) {
    if (s.particle_turbulence > 0) {
      this.vx += (Math.random() - 0.5) * s.particle_turbulence * 0.4;
      this.vy += (Math.random() - 0.5) * s.particle_turbulence * 0.4;
      this.vx *= 0.98;
      this.vy *= 0.98;
    }
    this.x += this.vx;
    this.y += this.vy;
    this.angle += this.spin;
    this.life++;
    if (this.life >= this.maxLife) this.alive = false;
  }

  get alpha() {
    const t = this.life / this.maxLife;
    return t > 0.7 ? this.opacity * (1 - (t - 0.7) / 0.3) | 0 : this.opacity;
  }

  draw(ctx) {
    const a = this.alpha / 255;
    if (a <= 0) return;
    ctx.save();
    ctx.globalAlpha = a;
    ctx.fillStyle = `rgb(${this.r},${this.g},${this.b})`;
    ctx.strokeStyle = ctx.fillStyle;
    ctx.translate(this.x, this.y);
    ctx.rotate(this.angle);

    const r = this.size;
    ctx.beginPath();

    switch (this.shape) {
      case "star":
        for (let i = 0; i < 10; i++) {
          const a2 = (i * Math.PI / 5) - Math.PI / 2;
          const rr = i % 2 === 0 ? r : r * 0.4;
          i === 0 ? ctx.moveTo(Math.cos(a2)*rr, Math.sin(a2)*rr)
                  : ctx.lineTo(Math.cos(a2)*rr, Math.sin(a2)*rr);
        }
        break;
      case "diamond":
        ctx.moveTo(0, -r); ctx.lineTo(r*0.6, 0);
        ctx.lineTo(0, r);  ctx.lineTo(-r*0.6, 0);
        break;
      case "spike":
        ctx.moveTo(0, -r*1.8); ctx.lineTo(r*0.4, r*0.4);
        ctx.lineTo(-r*0.4, r*0.4);
        break;
      case "drop":
        ctx.arc(0, 0, r * 0.6, 0, Math.PI * 2);
        break;
      default:  // circle
        ctx.arc(0, 0, r, 0, Math.PI * 2);
    }
    ctx.closePath();
    ctx.fill();
    ctx.restore();
  }
}

const particles = [];

function updateParticles() {
  // Update
  for (const p of particles) p.update(state);
  // Remove dead / out-of-bounds
  const margin = 60;
  for (let i = particles.length - 1; i >= 0; i--) {
    const p = particles[i];
    if (!p.alive || p.x < -margin || p.x > canvas.width + margin
        || p.y < -margin || p.y > canvas.height + margin) {
      particles.splice(i, 1);
    }
  }
  // Spawn
  const target = Math.min(state.particle_count, MAX_PARTICLES);
  while (particles.length < target) particles.push(new Particle(state));
}

// ── Geometry ──────────────────────────────────────────────────
let frameCount = 0;

function drawRings() {
  const cx = canvas.width / 2, cy = canvas.height / 2;
  const n  = state.geo_ring_count;
  const base = Math.min(canvas.width, canvas.height) * 0.10;
  const spacing = Math.min(canvas.width, canvas.height) * 0.07;

  for (let i = 0; i < n; i++) {
    const dir = i % 2 === 0 ? 1 : -1;
    ringAngles[i] = (ringAngles[i] + dir * state.geo_rotation_speed) % (Math.PI*2);
    const r = base + i * spacing;
    const sides = state.geo_sides;
    const opacity = Math.max(0.05, (1 - i / n) * state.geo_complexity * 0.4);
    const col = i % 2 === 0 ? state.primary_color : state.secondary_color;

    ctx.beginPath();
    for (let j = 0; j <= sides; j++) {
      const a = ringAngles[i] + (j / sides) * Math.PI * 2;
      const x = cx + Math.cos(a) * r;
      const y = cy + Math.sin(a) * r;
      j === 0 ? ctx.moveTo(x, y) : ctx.lineTo(x, y);
    }
    ctx.closePath();
    ctx.strokeStyle = `rgba(${col.r},${col.g},${col.b},${opacity})`;
    ctx.lineWidth = Math.max(1, state.geo_complexity * 2.5);
    ctx.stroke();
  }
}

function drawRadialBars() {
  if (!state.geo_radial_bars) return;
  const cx = canvas.width / 2, cy = canvas.height / 2;
  const n  = 48;
  const inner = Math.min(canvas.width, canvas.height) * 0.10;
  const maxH  = Math.min(canvas.width, canvas.height) * 0.18 * state.geo_bar_height;
  const beat  = 1 + state.beat_pulse * 0.5;
  const col   = state.primary_color;

  for (let i = 0; i < n; i++) {
    const angle = (2 * Math.PI * i / n) - Math.PI / 2;
    const amp = Math.max(0, Math.min(1,
      (0.3 + 0.5 * Math.abs(Math.sin(frameCount * 0.05 + i * 0.3 + state.beat_pulse * 2))) * beat
    ));
    const len = inner + amp * maxH;
    const opacity = 0.25 + amp * 0.7;

    ctx.beginPath();
    ctx.moveTo(cx + Math.cos(angle) * inner, cy + Math.sin(angle) * inner);
    ctx.lineTo(cx + Math.cos(angle) * len,   cy + Math.sin(angle) * len);
    ctx.strokeStyle = `rgba(${col.r},${col.g},${col.b},${opacity})`;
    ctx.lineWidth = Math.max(1, state.geo_complexity * 2);
    ctx.stroke();
  }
}

// ── Background ────────────────────────────────────────────────
function drawBackground() {
  const alpha = Math.max(0.03, (1 - state.blur_radius) * 0.94 + 0.04);
  ctx.fillStyle = `rgba(8,8,15,${alpha})`;
  ctx.fillRect(0, 0, canvas.width, canvas.height);

  // Bloom
  if (state.bloom_intensity > 0.05) {
    const cx = canvas.width / 2, cy = canvas.height / 2;
    const maxR = Math.min(canvas.width, canvas.height) * 0.35;
    const col = state.primary_color;
    for (let i = 0; i < 5; i++) {
      const t = i / 5;
      const r = maxR * (1 - t * 0.6);
      const a = state.bloom_intensity * (1 - t) * 0.14 * (1 + state.beat_pulse * 0.5);
      const grad = ctx.createRadialGradient(cx, cy, 0, cx, cy, r);
      grad.addColorStop(0,   `rgba(${col.r},${col.g},${col.b},${a})`);
      grad.addColorStop(1,   `rgba(${col.r},${col.g},${col.b},0)`);
      ctx.fillStyle = grad;
      ctx.fillRect(cx - r, cy - r, r*2, r*2);
    }
  }
}

// ── HUD ───────────────────────────────────────────────────────
function drawHUD() {
  const col = state.primary_color;
  ctx.font = "bold 18px monospace";
  ctx.fillStyle = `rgb(${col.r},${col.g},${col.b})`;
  ctx.fillText(state.emotion.toUpperCase(), 14, 30);

  // Confidence bar
  ctx.fillStyle = "rgba(40,40,60,0.8)";
  ctx.fillRect(14, 38, 120, 6);
  ctx.fillStyle = `rgb(${col.r},${col.g},${col.b})`;
  ctx.fillRect(14, 38, 120 * state.confidence, 6);

  ctx.font = "12px monospace";
  ctx.fillStyle = "rgba(180,180,200,0.7)";
  ctx.fillText(`conf ${(state.confidence * 100).toFixed(0)}%`, 140, 45);

  if (state.is_transitioning) {
    ctx.fillStyle = "rgba(160,160,220,0.6)";
    ctx.fillText("⟳ blending", 14, 62);
  }
}

// ── Main loop ─────────────────────────────────────────────────
function render() {
  frameCount++;
  drawBackground();
  drawRings();
  drawRadialBars();
  updateParticles();
  for (const p of particles) p.draw(ctx);
  drawHUD();
  requestAnimationFrame(render);
}
requestAnimationFrame(render);

// ── WebSocket connection ───────────────────────────────────────
let ws, reconnectDelay = 1000;

function connect() {
  ws = new WebSocket(WS_URL);

  ws.onopen = () => {
    console.log("[NN Viz] Connected to", WS_URL);
    reconnectDelay = 1000;
  };

  ws.onmessage = (e) => {
    try {
      const data = JSON.parse(e.data);
      Object.assign(state, data);
    } catch (err) {
      console.warn("[NN Viz] JSON parse error:", err);
    }
  };

  ws.onerror = (e) => console.warn("[NN Viz] WS error:", e);

  ws.onclose = () => {
    console.log(`[NN Viz] Disconnected. Reconnecting in ${reconnectDelay}ms…`);
    setTimeout(connect, reconnectDelay);
    reconnectDelay = Math.min(reconnectDelay * 2, 16000);
  };
}
connect();
