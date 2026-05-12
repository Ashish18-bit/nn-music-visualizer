"use strict";

const WS_URL = window.WS_URL || `ws://${location.hostname}:8765/ws`;
const MAX_PARTICLES = 300;

// ── Canvas setup ─────────────────────────────────────────────
const canvas = document.getElementById("viz") || (() => {
  const c = document.createElement("canvas");
  document.body.appendChild(c);
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

// ── LOCAL audio analysis (runs entirely in the browser) ───────
let localAnalyser = null;
let localFreqData = null;
let localAmplitude = 0;
let localFreqBins = new Array(48).fill(0);

// Called by index.html when audio context is ready
window.setLocalAnalyser = function(analyserNode) {
  localAnalyser = analyserNode;
  localFreqData = new Uint8Array(analyserNode.frequencyBinCount);
};

function updateLocalAudio() {
  if (!localAnalyser || !localFreqData) return;

  // Get raw frequency data
  localAnalyser.getByteFrequencyData(localFreqData);

  // Compute amplitude (0-1)
  let sum = 0;
  for (let i = 0; i < localFreqData.length; i++) sum += localFreqData[i];
  localAmplitude = sum / localFreqData.length / 255;

  // Downsample to 48 bins for radial bars
  const binSize = Math.floor(localFreqData.length / 48);
  for (let i = 0; i < 48; i++) {
    let avg = 0;
    for (let j = 0; j < binSize; j++) {
      avg += localFreqData[i * binSize + j];
    }
    localFreqBins[i] = avg / binSize / 255;
  }

  // Directly drive visual parameters from audio
  // Beat pulse — driven by bass frequencies (first 8 bins)
  let bass = 0;
  for (let i = 0; i < 8; i++) bass += localFreqBins[i];
  bass /= 8;
  state.beat_pulse = Math.min(1.0, bass * 1.8);

  // Particle speed — driven by mid frequencies
  let mid = 0;
  for (let i = 8; i < 24; i++) mid += localFreqBins[i];
  mid /= 16;
  const baseSpeed = state.particle_speed;
  state._live_speed = baseSpeed * (0.5 + mid * 2.0);

  // Bloom intensity — driven by overall amplitude
  state._live_bloom = Math.min(1.0, localAmplitude * 2.5);

  // Geometry bar height — direct frequency data
  state._live_bars = localFreqBins.slice();
}

// ── Ring rotation angles ──────────────────────────────────────
const ringAngles = new Array(8).fill(0);
let frameCount = 0;

// ── Particles ─────────────────────────────────────────────────
class Particle {
  constructor() { this.reset(); }

  reset() {
    const cx = canvas.width / 2, cy = canvas.height / 2;
    const angle = Math.random() * Math.PI * 2;
    const dist  = Math.random() * Math.min(canvas.width, canvas.height) * 0.4;
    this.x = cx + Math.cos(angle) * dist;
    this.y = cy + Math.sin(angle) * dist;
    const speed = (state._live_speed || state.particle_speed) * (0.5 + Math.random());
    const outAngle = angle + (Math.random() - 0.5) * 0.8;
    this.vx = Math.cos(outAngle) * speed;
    this.vy = Math.sin(outAngle) * speed;
    this.size   = state.particle_size * (0.6 + Math.random() * 0.8);
    this.life   = 0;
    this.maxLife = 60 + Math.random() * 90;
    this.angle  = Math.random() * Math.PI * 2;
    this.spin   = (Math.random() - 0.5) * 0.1;
    this.shape  = state.particle_shape;
    this.r = state.primary_color.r;
    this.g = state.primary_color.g;
    this.b = state.primary_color.b;
    this.baseOpacity = state.particle_opacity;
  }

  update() {
    // Speed reacts to live audio
    const speedMult = state._live_speed
      ? state._live_speed / Math.max(0.1, state.particle_speed)
      : 1.0;

    if (state.particle_turbulence > 0) {
      this.vx += (Math.random() - 0.5) * state.particle_turbulence * 0.4;
      this.vy += (Math.random() - 0.5) * state.particle_turbulence * 0.4;
      this.vx *= 0.98;
      this.vy *= 0.98;
    }
    this.x += this.vx * speedMult;
    this.y += this.vy * speedMult;
    this.angle += this.spin;
    this.life++;
  }

  get alive() { return this.life < this.maxLife; }

  get alpha() {
    const t = this.life / this.maxLife;
    // Pulse with beat
    const pulsed = this.baseOpacity * (1 + state.beat_pulse * 0.5);
    return t > 0.7
      ? pulsed * (1 - (t - 0.7) / 0.3)
      : pulsed;
  }

  draw() {
    const a = Math.min(255, this.alpha) / 255;
    if (a <= 0) return;
    ctx.save();
    ctx.globalAlpha = a;
    ctx.fillStyle = `rgb(${this.r},${this.g},${this.b})`;
    ctx.translate(this.x, this.y);
    ctx.rotate(this.angle);

    // Scale size with beat pulse
    const r = this.size * (1 + state.beat_pulse * 0.4);

    ctx.beginPath();
    switch (this.shape) {
      case "star":
        for (let i = 0; i < 10; i++) {
          const a2 = (i * Math.PI / 5) - Math.PI / 2;
          const rr = i % 2 === 0 ? r : r * 0.4;
          i === 0
            ? ctx.moveTo(Math.cos(a2)*rr, Math.sin(a2)*rr)
            : ctx.lineTo(Math.cos(a2)*rr, Math.sin(a2)*rr);
        }
        break;
      case "diamond":
        ctx.moveTo(0,-r); ctx.lineTo(r*0.6,0);
        ctx.lineTo(0,r);  ctx.lineTo(-r*0.6,0);
        break;
      case "spike":
        ctx.moveTo(0,-r*1.8);
        ctx.lineTo(r*0.4,r*0.4);
        ctx.lineTo(-r*0.4,r*0.4);
        break;
      case "drop":
        ctx.arc(0, 0, r * 0.6, 0, Math.PI * 2);
        break;
      default:
        ctx.arc(0, 0, r, 0, Math.PI * 2);
    }
    ctx.closePath();
    ctx.fill();
    ctx.restore();
  }
}

const particles = [];

function updateParticles() {
  for (const p of particles) p.update();

  const margin = 60;
  for (let i = particles.length - 1; i >= 0; i--) {
    const p = particles[i];
    if (!p.alive
        || p.x < -margin || p.x > canvas.width + margin
        || p.y < -margin || p.y > canvas.height + margin) {
      particles.splice(i, 1);
    }
  }

  const target = Math.min(state.particle_count, MAX_PARTICLES);
  while (particles.length < target) particles.push(new Particle());
}

// ── Background ────────────────────────────────────────────────
function drawBackground() {
  // Trail alpha: high blur = long trails
  const trailAlpha = Math.max(0.03, (1 - state.blur_radius) * 0.92 + 0.04);
  ctx.fillStyle = `rgba(8,8,15,${trailAlpha})`;
  ctx.fillRect(0, 0, canvas.width, canvas.height);

  // Bloom — reacts to live audio amplitude
  const bloomIntensity = state._live_bloom !== undefined
    ? state._live_bloom
    : state.bloom_intensity;

  if (bloomIntensity > 0.05) {
    const cx = canvas.width / 2, cy = canvas.height / 2;
    const maxR = Math.min(canvas.width, canvas.height) * 0.35;
    const col = state.primary_color;

    for (let i = 0; i < 5; i++) {
      const t = i / 5;
      const r = maxR * (1 - t * 0.6);
      const a = bloomIntensity * (1 - t) * 0.18 * (1 + state.beat_pulse);
      const grad = ctx.createRadialGradient(cx,cy,0, cx,cy,r);
      grad.addColorStop(0, `rgba(${col.r},${col.g},${col.b},${a})`);
      grad.addColorStop(1, `rgba(${col.r},${col.g},${col.b},0)`);
      ctx.fillStyle = grad;
      ctx.fillRect(cx-r, cy-r, r*2, r*2);
    }
  }
}

// ── Geometry ──────────────────────────────────────────────────
function drawRings() {
  const cx = canvas.width / 2, cy = canvas.height / 2;
  const n  = state.geo_ring_count;
  const base = Math.min(canvas.width, canvas.height) * 0.10;
  const spacing = Math.min(canvas.width, canvas.height) * 0.07;

  // Rings rotate faster with more bass
  const speedBoost = 1 + state.beat_pulse * 2;

  for (let i = 0; i < n; i++) {
    const dir = i % 2 === 0 ? 1 : -1;
    ringAngles[i] = (ringAngles[i] + dir * state.geo_rotation_speed * speedBoost) % (Math.PI*2);

    // Ring pulses outward on beat
    const beatRadius = state.beat_pulse * 15;
    const r = base + i * spacing + beatRadius;
    const sides = state.geo_sides;
    const opacity = Math.max(0.04, (1 - i/n) * state.geo_complexity * 0.5);
    const col = i % 2 === 0 ? state.primary_color : state.secondary_color;

    ctx.beginPath();
    for (let j = 0; j <= sides; j++) {
      const a = ringAngles[i] + (j/sides) * Math.PI * 2;
      const x = cx + Math.cos(a) * r;
      const y = cy + Math.sin(a) * r;
      j === 0 ? ctx.moveTo(x,y) : ctx.lineTo(x,y);
    }
    ctx.closePath();
    ctx.strokeStyle = `rgba(${col.r},${col.g},${col.b},${opacity})`;
    ctx.lineWidth = Math.max(1, state.geo_complexity * 3);
    ctx.stroke();
  }
}

function drawRadialBars() {
  if (!state.geo_radial_bars) return;
  const cx = canvas.width / 2, cy = canvas.height / 2;
  const n  = 48;
  const inner = Math.min(canvas.width, canvas.height) * 0.10;
  const maxH  = Math.min(canvas.width, canvas.height) * 0.22 * state.geo_bar_height;
  const col   = state.primary_color;

  for (let i = 0; i < n; i++) {
    const angle = (2 * Math.PI * i / n) - Math.PI / 2;

    // Use REAL frequency data if available, else fallback demo
    let amp;
    if (state._live_bars && localAmplitude > 0.01) {
      amp = state._live_bars[i] || 0;
    } else {
      // Demo sine wave fallback
      amp = 0.2 + 0.4 * Math.abs(
        Math.sin(frameCount * 0.04 + i * 0.3 + state.beat_pulse * 2)
      );
    }

    // Beat boost
    amp = Math.min(1.0, amp * (1 + state.beat_pulse * 0.8));
    const len = inner + amp * maxH;
    const opacity = 0.3 + amp * 0.7;

    ctx.beginPath();
    ctx.moveTo(cx + Math.cos(angle)*inner, cy + Math.sin(angle)*inner);
    ctx.lineTo(cx + Math.cos(angle)*len,   cy + Math.sin(angle)*len);
    ctx.strokeStyle = `rgba(${col.r},${col.g},${col.b},${opacity})`;
    ctx.lineWidth = Math.max(1, state.geo_complexity * 2.5);
    ctx.stroke();
  }
}

// ── Main render loop ──────────────────────────────────────────
function render() {
  frameCount++;

  // Pull live audio data every frame
  updateLocalAudio();

  drawBackground();
  drawRings();
  drawRadialBars();
  updateParticles();
  for (const p of particles) p.draw();

  requestAnimationFrame(render);
}
requestAnimationFrame(render);

// ── WebSocket ─────────────────────────────────────────────────
let ws, reconnectDelay = 1000;

function connect() {
  ws = new WebSocket(WS_URL);

  ws.onopen = () => {
    console.log("[NN Viz] Connected");
    reconnectDelay = 1000;
  };

  ws.onmessage = (e) => {
    try {
      const data = JSON.parse(e.data);
      // Merge server state but don't override live audio params
      Object.assign(state, data);
    } catch(err) {
      console.warn("[NN Viz] parse error:", err);
    }
  };

  ws.onerror = (e) => console.warn("[NN Viz] error:", e);

  ws.onclose = () => {
    setTimeout(connect, reconnectDelay);
    reconnectDelay = Math.min(reconnectDelay * 2, 16000);
  };
}
connect();
// ── Audio streaming to server ─────────────────────────────────
let audioWs = null;
const AUDIO_WS_URL = WS_URL.replace("/ws", "/audio");

function connectAudioWs() {
  audioWs = new WebSocket(AUDIO_WS_URL);
  audioWs.binaryType = "arraybuffer";

  audioWs.onopen = () => {
    console.log("[NN Viz] Audio WS connected");
  };

  audioWs.onclose = () => {
    setTimeout(connectAudioWs, 2000);
  };
}
connectAudioWs();

// Called by index.html when audio context is ready
window.setLocalAnalyser = function(analyserNode, sampleRate) {
  localAnalyser = analyserNode;
  localFreqData = new Uint8Array(analyserNode.frequencyBinCount);

  // Tell server the sample rate
  if (audioWs && audioWs.readyState === WebSocket.OPEN) {
    audioWs.send(JSON.stringify({ sampleRate: sampleRate || 44100 }));
  }

  // Stream audio chunks to server every 250ms
  const scriptProcessor = analyserNode.context.createScriptProcessor(4096, 1, 1);
  scriptProcessor.onaudioprocess = (e) => {
    if (audioWs && audioWs.readyState === WebSocket.OPEN) {
      const inputData = e.inputBuffer.getChannelData(0);
      const pcm = new Float32Array(inputData);
      audioWs.send(pcm.buffer);
    }
  };
  analyserNode.connect(scriptProcessor);
  scriptProcessor.connect(analyserNode.context.destination);
};