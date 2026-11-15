"""
Reverse Sign Renderer & Interactive 3D Avatar HTML Generator
==============================================================
Generates a zero-dependency, GPU-accelerated HTML5 Canvas 3D Hand Skeletal Visualizer.
Features:
1. 3D Hand Perspective Projection with interactive 360-degree Orbit & Pitch controls.
2. Web Speech API (Microphone) real-time Speech-to-Sign transcription.
3. Fluid 60 FPS Biomechanical Hand Interpolation between sign poses.
4. Interactive Sign Timeline with jump-to-sign scrubbing.
5. Speed controller (0.5x, 1x, 1.5x, 2x), pause/play, loop, and step controls.
6. Embedded in Streamlit or exported as standalone HTML artifact.
"""

import json
import sys
import os
from typing import Dict, Any, Optional

# Ensure project root is accessible
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from utils.text_to_asl_synthesizer import TextToASLSynthesizer


def generate_avatar_html(
    default_text: str = "HELLO TEAM NEED HELP TODAY",
    synthesizer: Optional[TextToASLSynthesizer] = None,
    library_json: Optional[Dict[str, Any]] = None
) -> str:
    """
    Renders a self-contained, responsive, modern dark-themed HTML/JS 3D visualizer
    with Web Speech API microphone capture and interactive 3D hand animation.
    """
    if synthesizer is None:
        synthesizer = TextToASLSynthesizer()

    payload = synthesizer.generate_animation_payload(default_text)
    full_library = library_json or synthesizer.library

    payload_json = json.dumps(payload)
    library_json_str = json.dumps(full_library)

    html_code = f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>ASL 3D Reverse Speech-to-Sign Avatar</title>
  <style>
    * {{
      box-sizing: border-box;
      margin: 0;
      padding: 0;
      font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Helvetica Neue", Arial, sans-serif;
    }}
    body {{
      background: #0B1120;
      color: #F8FAFC;
      display: flex;
      flex-direction: column;
      align-items: center;
      padding: 12px;
      overflow-x: hidden;
    }}
    .avatar-card {{
      width: 100%;
      max-width: 980px;
      background: #0F172A;
      border: 1px solid #1E293B;
      border-radius: 16px;
      box-shadow: 0 10px 25px -5px rgba(0, 0, 0, 0.5), 0 8px 10px -6px rgba(0, 0, 0, 0.4);
      padding: 18px;
    }}
    .header-bar {{
      display: flex;
      justify-content: space-between;
      align-items: center;
      margin-bottom: 14px;
      border-bottom: 1px solid #1E293B;
      padding-bottom: 10px;
      flex-wrap: wrap;
      gap: 10px;
    }}
    .header-title {{
      font-size: 1.18rem;
      font-weight: 800;
      color: #38BDF8;
      display: flex;
      align-items: center;
      gap: 8px;
    }}
    .header-badge {{
      background: #1E293B;
      color: #94A3B8;
      font-size: 0.75rem;
      padding: 3px 8px;
      border-radius: 6px;
      border: 1px solid #334155;
    }}

    /* Input & Microphone Bar */
    .input-section {{
      display: flex;
      gap: 8px;
      margin-bottom: 14px;
      flex-wrap: wrap;
    }}
    .speech-input {{
      flex: 1;
      min-width: 260px;
      background: #1E293B;
      border: 1px solid #334155;
      border-radius: 8px;
      padding: 10px 14px;
      font-size: 0.95rem;
      color: #F8FAFC;
      outline: none;
      transition: border-color 0.2s;
    }}
    .speech-input:focus {{
      border-color: #38BDF8;
      box-shadow: 0 0 0 2px rgba(56, 189, 248, 0.2);
    }}
    .btn {{
      background: #0284C7;
      color: #FFFFFF;
      border: none;
      border-radius: 8px;
      padding: 10px 16px;
      font-weight: 700;
      font-size: 0.88rem;
      cursor: pointer;
      display: inline-flex;
      align-items: center;
      gap: 6px;
      transition: all 0.15s ease;
    }}
    .btn:hover {{
      background: #0369A1;
      transform: translateY(-1px);
    }}
    .btn-mic {{
      background: #DC2626;
    }}
    .btn-mic:hover {{
      background: #B91C1C;
    }}
    .btn-mic.listening {{
      background: #EF4444;
      animation: pulse 1.2s infinite;
    }}
    @keyframes pulse {{
      0% {{ box-shadow: 0 0 0 0 rgba(239, 68, 68, 0.7); }}
      70% {{ box-shadow: 0 0 0 10px rgba(239, 68, 68, 0); }}
      100% {{ box-shadow: 0 0 0 0 rgba(239, 68, 68, 0); }}
    }}
    .btn-secondary {{
      background: #334155;
      color: #E2E8F0;
    }}
    .btn-secondary:hover {{
      background: #475569;
    }}

    /* Preset Suggestions */
    .presets-row {{
      display: flex;
      gap: 6px;
      margin-bottom: 12px;
      flex-wrap: wrap;
      align-items: center;
    }}
    .preset-label {{
      font-size: 0.78rem;
      color: #64748B;
      font-weight: 600;
    }}
    .preset-btn {{
      background: #1E293B;
      border: 1px solid #334155;
      color: #94A3B8;
      border-radius: 6px;
      padding: 3px 8px;
      font-size: 0.78rem;
      cursor: pointer;
      transition: all 0.15s;
    }}
    .preset-btn:hover {{
      background: #334155;
      color: #38BDF8;
      border-color: #38BDF8;
    }}

    /* Visualizer & 3D Canvas Container */
    .stage-container {{
      position: relative;
      width: 100%;
      height: 420px;
      background: radial-gradient(circle at 50% 40%, #172554 0%, #0F172A 70%, #020617 100%);
      border: 1px solid #1E293B;
      border-radius: 12px;
      overflow: hidden;
      display: flex;
      justify-content: center;
      align-items: center;
      cursor: grab;
      user-select: none;
    }}
    .stage-container:active {{
      cursor: grabbing;
    }}
    #avatarCanvas {{
      display: block;
      width: 100%;
      height: 100%;
    }}

    /* Overlay HUD elements */
    .hud-sign-card {{
      position: absolute;
      top: 14px;
      left: 14px;
      background: rgba(15, 23, 42, 0.85);
      backdrop-filter: blur(8px);
      border: 1px solid rgba(56, 189, 248, 0.4);
      border-radius: 10px;
      padding: 10px 14px;
      max-width: 240px;
      pointer-events: none;
    }}
    .hud-sign-title {{
      font-size: 1.35rem;
      font-weight: 900;
      color: #38BDF8;
      line-height: 1.1;
      display: flex;
      align-items: center;
      gap: 6px;
    }}
    .hud-sign-sub {{
      font-size: 0.74rem;
      color: #94A3B8;
      margin-top: 4px;
      line-height: 1.25;
    }}

    .hud-orbit-hint {{
      position: absolute;
      top: 14px;
      right: 14px;
      font-size: 0.72rem;
      color: #64748B;
      background: rgba(15, 23, 42, 0.6);
      padding: 4px 8px;
      border-radius: 6px;
      pointer-events: none;
    }}

    /* Playback Timeline & Controls */
    .controls-row {{
      display: flex;
      justify-content: space-between;
      align-items: center;
      margin-top: 12px;
      flex-wrap: wrap;
      gap: 10px;
    }}
    .play-btns-group {{
      display: flex;
      gap: 6px;
      align-items: center;
    }}
    .speed-group {{
      display: flex;
      align-items: center;
      gap: 6px;
      font-size: 0.82rem;
      color: #94A3B8;
    }}
    .speed-select {{
      background: #1E293B;
      color: #38BDF8;
      border: 1px solid #334155;
      border-radius: 6px;
      padding: 4px 8px;
      font-weight: 700;
      outline: none;
      cursor: pointer;
    }}

    /* Sign Timeline Badges */
    .timeline-container {{
      margin-top: 12px;
      background: #020617;
      border: 1px solid #1E293B;
      border-radius: 10px;
      padding: 10px;
      overflow-x: auto;
      white-space: nowrap;
      display: flex;
      gap: 6px;
      scrollbar-width: thin;
      scrollbar-color: #334155 #020617;
    }}
    .timeline-token {{
      display: inline-flex;
      flex-direction: column;
      align-items: center;
      background: #1E293B;
      border: 1px solid #334155;
      border-radius: 8px;
      padding: 6px 10px;
      cursor: pointer;
      transition: all 0.15s ease;
      min-width: 44px;
    }}
    .timeline-token:hover {{
      border-color: #38BDF8;
      transform: translateY(-2px);
    }}
    .timeline-token.active {{
      background: #0369A1;
      border-color: #38BDF8;
      box-shadow: 0 0 12px rgba(56, 189, 248, 0.5);
      transform: scale(1.06);
    }}
    .token-text {{
      font-size: 1.05rem;
      font-weight: 800;
      color: #F8FAFC;
    }}
    .token-type {{
      font-size: 0.65rem;
      color: #94A3B8;
      text-transform: uppercase;
      margin-top: 2px;
    }}

    /* Legend */
    .legend-row {{
      display: flex;
      gap: 12px;
      margin-top: 10px;
      justify-content: center;
      font-size: 0.72rem;
      color: #64748B;
      flex-wrap: wrap;
    }}
    .legend-item {{
      display: flex;
      align-items: center;
      gap: 4px;
    }}
    .legend-dot {{
      width: 8px;
      height: 8px;
      border-radius: 50%;
    }}
  </style>
</head>
<body>
  <div class="avatar-card">
    <div class="header-bar">
      <div class="header-title">
        <span>🗣️➡️🤟 Reverse Communication Bridge</span>
        <span class="header-badge">3D Sign Visualizer</span>
      </div>
      <div class="header-badge" id="statusBadge">Engine Ready (60 FPS)</div>
    </div>

    <!-- Speech & Text Input -->
    <div class="input-section">
      <input type="text" id="sentenceInput" class="speech-input" placeholder="Type or speak a sentence (e.g. 'Hello! How are you?')" value="{default_text}">
      <button class="btn btn-mic" id="btnMic" title="Click to speak using Web Speech API">🎙️ Speak Now</button>
      <button class="btn" id="btnTranslate">✨ Translate & Perform</button>
    </div>

    <!-- Quick Presets -->
    <div class="presets-row">
      <span class="preset-label">Quick Phrases:</span>
      <button class="preset-btn" onclick="applyPreset('HELLO HOW ARE YOU')">👋 Hello! How are you?</button>
      <button class="preset-btn" onclick="applyPreset('THANK YOU VERY MUCH')">🙏 Thank you</button>
      <button class="preset-btn" onclick="applyPreset('NEED HELP COMPUTER TODAY')">💻 Need help with computer</button>
      <button class="preset-btn" onclick="applyPreset('NICE TO MEET YOU')">🤝 Nice to meet you</button>
      <button class="preset-btn" onclick="applyPreset('GOOD MORNING TEAM')">☀️ Good morning team</button>
    </div>

    <!-- 3D Stage -->
    <div class="stage-container" id="stageContainer">
      <canvas id="avatarCanvas"></canvas>
      
      <div class="hud-sign-card" id="hudCard">
        <div class="hud-sign-title" id="hudSign">REST</div>
        <div class="hud-sign-sub" id="hudDesc">Relaxed natural neutral hand posture.</div>
      </div>

      <div class="hud-orbit-hint">🔄 Drag to Orbit 3D • Scroll to Zoom</div>
    </div>

    <!-- Controls Row -->
    <div class="controls-row">
      <div class="play-btns-group">
        <button class="btn btn-secondary" id="btnPrev" title="Previous Sign">⏮️ Prev</button>
        <button class="btn" id="btnPlayPause">⏸️ Pause</button>
        <button class="btn btn-secondary" id="btnNext" title="Next Sign">⏭️ Next</button>
        <button class="btn btn-secondary" id="btnReset" title="Replay from Beginning">🔄 Replay</button>
      </div>

      <div class="speed-group">
        <span>Playback Speed:</span>
        <select class="speed-select" id="speedSelect">
          <option value="0.5">0.5x (Slow/Learn)</option>
          <option value="0.75">0.75x</option>
          <option value="1.0" selected>1.0x (Normal)</option>
          <option value="1.5">1.5x (Fast)</option>
          <option value="2.0">2.0x (Rapid)</option>
        </select>
      </div>
    </div>

    <!-- Timeline Scrubber -->
    <div class="timeline-container" id="timelineContainer">
      <!-- Dynamic tokens populated by JS -->
    </div>

    <!-- 3D Anatomical Color Legend -->
    <div class="legend-row">
      <div class="legend-item"><div class="legend-dot" style="background:#F59E0B;"></div> Thumb</div>
      <div class="legend-item"><div class="legend-dot" style="background:#06B6D4;"></div> Index</div>
      <div class="legend-item"><div class="legend-dot" style="background:#10B981;"></div> Middle</div>
      <div class="legend-item"><div class="legend-dot" style="background:#8B5CF6;"></div> Ring</div>
      <div class="legend-item"><div class="legend-dot" style="background:#F43F5E;"></div> Pinky</div>
      <div class="legend-item"><div class="legend-dot" style="background:#38BDF8;"></div> Wrist/Palm</div>
    </div>
  </div>

  <script>
    // Embedded Animation & Canonical Knowledge Base
    let currentPayload = {payload_json};
    const canonicalLibrary = {library_json_str};

    const canvas = document.getElementById('avatarCanvas');
    const ctx = canvas.getContext('2d');
    const stage = document.getElementById('stageContainer');

    // Canvas Sizing
    function resizeCanvas() {{
      const rect = stage.getBoundingClientRect();
      const dpr = window.devicePixelRatio || 1;
      canvas.width = rect.width * dpr;
      canvas.height = rect.height * dpr;
      ctx.scale(dpr, dpr);
    }}
    window.addEventListener('resize', resizeCanvas);
    resizeCanvas();

    // 3D Camera / Orbit State
    let yaw = -0.22;       // Horizontal orbit (radians)
    let pitch = 0.12;      // Vertical tilt (radians)
    let zoomScale = 1.0;
    let isDragging = false;
    let lastMouseX = 0, lastMouseY = 0;

    stage.addEventListener('mousedown', (e) => {{
      isDragging = true;
      lastMouseX = e.clientX;
      lastMouseY = e.clientY;
    }});
    window.addEventListener('mouseup', () => isDragging = false);
    window.addEventListener('mousemove', (e) => {{
      if (!isDragging) return;
      const dx = e.clientX - lastMouseX;
      const dy = e.clientY - lastMouseY;
      yaw += dx * 0.008;
      pitch += dy * 0.008;
      pitch = Math.max(-1.1, Math.min(1.1, pitch));
      lastMouseX = e.clientX;
      lastMouseY = e.clientY;
    }});

    // Touch support for mobile/tablets
    stage.addEventListener('touchstart', (e) => {{
      if (e.touches.length === 1) {{
        isDragging = true;
        lastMouseX = e.touches[0].clientX;
        lastMouseY = e.touches[0].clientY;
      }}
    }});
    window.addEventListener('touchend', () => isDragging = false);
    window.addEventListener('touchmove', (e) => {{
      if (!isDragging || e.touches.length !== 1) return;
      const dx = e.touches[0].clientX - lastMouseX;
      const dy = e.touches[0].clientY - lastMouseY;
      yaw += dx * 0.008;
      pitch += dy * 0.008;
      pitch = Math.max(-1.1, Math.min(1.1, pitch));
      lastMouseX = e.touches[0].clientX;
      lastMouseY = e.touches[0].clientY;
    }});

    // Mouse Wheel Zoom
    stage.addEventListener('wheel', (e) => {{
      e.preventDefault();
      zoomScale -= e.deltaY * 0.0012;
      zoomScale = Math.max(0.5, Math.min(2.5, zoomScale));
    }}, {{ passive: false }});

    // Playback Engine State
    let isPlaying = true;
    let playbackSpeed = 1.0;
    let currentTimeMs = 0;
    let lastFrameTime = performance.now();
    let currentKeyframeIndex = 0;

    // Joint Colors
    const JOINT_COLORS = {{
      0: '#38BDF8', 1: '#38BDF8', 5: '#38BDF8', 9: '#38BDF8', 13: '#38BDF8', 17: '#38BDF8', // Palm
      2: '#F59E0B', 3: '#F59E0B', 4: '#F59E0B',                                              // Thumb
      6: '#06B6D4', 7: '#06B6D4', 8: '#06B6D4',                                              // Index
      10: '#10B981', 11: '#10B981', 12: '#10B981',                                          // Middle
      14: '#8B5CF6', 15: '#8B5CF6', 16: '#8B5CF6',                                          // Ring
      18: '#F43F5E', 19: '#F43F5E', 20: '#F43F5E'                                           // Pinky
    }};

    // 3D Perspective Projection Function
    function project3D(pt, width, height) {{
      const x = pt[0], y = pt[1], z = pt[2];

      // 1. Rotate Y (Yaw)
      const cosY = Math.cos(yaw), sinY = Math.sin(yaw);
      const x1 = x * cosY + z * sinY;
      const z1 = -x * sinY + z * cosY;

      // 2. Rotate X (Pitch)
      const cosX = Math.cos(pitch), sinX = Math.sin(pitch);
      const y1 = y * cosX - z1 * sinX;
      const z2 = y * sinX + z1 * cosX;

      // 3. Perspective Camera
      const cameraDist = 2.8;
      const factor = cameraDist / (cameraDist - z2);
      const baseScale = Math.min(width, height) * 0.62 * zoomScale;

      const screenX = (width / 2) + x1 * baseScale * factor;
      // Invert Y so -Y (fingers) point upwards
      const screenY = (height / 2 + 35) + y1 * baseScale * factor;

      return {{
        x: screenX,
        y: screenY,
        z: z2,
        depthFactor: factor
      }};
    }}

    // Biomechanical Cosine / Smoothstep Easing
    function smoothstep(t) {{
      return t * t * (3 - 2 * t);
    }}

    // Compute Interpolated Pose at Current Time
    function computeCurrentPose(timeMs) {{
      const kfs = currentPayload.keyframes;
      if (!kfs || kfs.length === 0) return null;

      // Find active keyframe
      let activeIdx = 0;
      for (let i = 0; i < kfs.length; i++) {{
        if (timeMs >= kfs[i].start_time_ms) {{
          activeIdx = i;
        }} else {{
          break;
        }}
      }}
      currentKeyframeIndex = activeIdx;

      const currKf = kfs[activeIdx];
      const nextKf = kfs[Math.min(activeIdx + 1, kfs.length - 1)];

      const elapsed = timeMs - currKf.start_time_ms;
      const holdTime = currKf.hold_time_ms;
      const transTime = currKf.transition_time_ms || 250;

      // If holding current sign
      if (elapsed <= holdTime || activeIdx === kfs.length - 1) {{
        return {{
          landmarks: currKf.landmarks,
          keyframe: currKf,
          index: activeIdx
        }};
      }}

      // If transitioning to next sign
      const transProgress = Math.min(1.0, (elapsed - holdTime) / transTime);
      const eased = smoothstep(transProgress);

      const interpolated = [];
      const l1 = currKf.landmarks;
      const l2 = nextKf.landmarks;

      for (let j = 0; j < 21; j++) {{
        const p1 = l1[j] || [0, 0, 0];
        const p2 = l2[j] || [0, 0, 0];
        interpolated.push([
          p1[0] + (p2[0] - p1[0]) * eased,
          p1[1] + (p2[1] - p1[1]) * eased,
          p1[2] + (p2[2] - p1[2]) * eased
        ]);
      }}

      return {{
        landmarks: interpolated,
        keyframe: currKf,
        nextKeyframe: nextKf,
        index: activeIdx
      }};
    }}

    // Render 3D Hand Skeleton Loop
    function render(timestamp) {{
      const dt = (timestamp - lastFrameTime);
      lastFrameTime = timestamp;

      if (isPlaying && currentPayload.total_duration_ms > 0) {{
        currentTimeMs += dt * playbackSpeed;
        if (currentTimeMs >= currentPayload.total_duration_ms) {{
          currentTimeMs = 0; // Loop back
        }}
      }}

      const rect = stage.getBoundingClientRect();
      const w = rect.width;
      const h = rect.height;

      ctx.clearRect(0, 0, w, h);

      const poseData = computeCurrentPose(currentTimeMs);

      if (poseData && poseData.landmarks) {{
        const pts3D = poseData.landmarks;
        const projected = pts3D.map(pt => project3D(pt, w, h));

        // Update HUD
        const kf = poseData.keyframe;
        document.getElementById('hudSign').textContent = kf.sign;
        document.getElementById('hudDesc').textContent = kf.desc || 'Active sign articulation.';

        // Update Timeline Active Badge
        updateTimelineActive(poseData.index);

        // 1. Draw Bones (Connections)
        const conns = currentPayload.connections || [];
        ctx.lineCap = 'round';

        // Sort connections by average depth for correct painter's order
        const sortedConns = [...conns].map(c => {{
          const zAvg = (projected[c[0]].z + projected[c[1]].z) / 2;
          return {{ c, zAvg }};
        }}).sort((a, b) => b.zAvg - a.zAvg);

        for (const item of sortedConns) {{
          const [i1, i2] = item.c;
          const p1 = projected[i1];
          const p2 = projected[i2];

          // Depth-attenuated line width
          const lineWidth = Math.max(2.5, 4.5 * ((p1.depthFactor + p2.depthFactor) / 2));
          
          // Glow effect
          ctx.beginPath();
          ctx.moveTo(p1.x, p1.y);
          ctx.lineTo(p2.x, p2.y);
          ctx.strokeStyle = 'rgba(56, 189, 248, 0.2)';
          ctx.lineWidth = lineWidth * 2.2;
          ctx.stroke();

          // Core bone
          ctx.beginPath();
          ctx.moveTo(p1.x, p1.y);
          ctx.lineTo(p2.x, p2.y);
          ctx.strokeStyle = JOINT_COLORS[i2] || '#38BDF8';
          ctx.lineWidth = lineWidth;
          ctx.stroke();
        }}

        // 2. Draw Joints (Nodes)
        // Sort joints by depth
        const sortedNodes = projected.map((p, idx) => ({{ ...p, idx }})).sort((a, b) => b.z - a.z);

        for (const node of sortedNodes) {{
          const baseRadius = (node.idx === 0 || node.idx === 4 || node.idx === 8 || node.idx === 12 || node.idx === 16 || node.idx === 20) ? 7.5 : 5.5;
          const r = Math.max(3.0, baseRadius * node.depthFactor);

          const color = JOINT_COLORS[node.idx] || '#38BDF8';

          // Outer atmospheric glow
          const grad = ctx.createRadialGradient(node.x, node.y, r * 0.2, node.x, node.y, r * 2.2);
          grad.addColorStop(0, color);
          grad.addColorStop(1, 'rgba(0, 0, 0, 0)');

          ctx.beginPath();
          ctx.arc(node.x, node.y, r * 2.2, 0, Math.PI * 2);
          ctx.fillStyle = grad;
          ctx.fill();

          // Joint core
          ctx.beginPath();
          ctx.arc(node.x, node.y, r, 0, Math.PI * 2);
          ctx.fillStyle = '#FFFFFF';
          ctx.fill();

          ctx.beginPath();
          ctx.arc(node.x, node.y, r * 0.8, 0, Math.PI * 2);
          ctx.fillStyle = color;
          ctx.fill();
        }}
      }}

      requestAnimationFrame(render);
    }}
    requestAnimationFrame(render);

    // Build Timeline Pills
    function buildTimeline() {{
      const cont = document.getElementById('timelineContainer');
      cont.innerHTML = '';
      const kfs = currentPayload.keyframes;
      if (!kfs) return;

      kfs.forEach((kf, idx) => {{
        const tok = document.createElement('div');
        tok.className = 'timeline-token' + (idx === 0 ? ' active' : '');
        tok.id = 'tok-' + idx;
        tok.innerHTML = `
          <div class="token-text">${{kf.sign === 'SPACE' ? '␣' : (kf.sign === 'REST' ? '⏸' : kf.sign)}}</div>
          <div class="token-type">${{kf.type || 'sign'}}</div>
        `;
        tok.addEventListener('click', () => {{
          jumpToKeyframe(idx);
        }});
        cont.appendChild(tok);
      }});
    }}

    function updateTimelineActive(activeIndex) {{
      const pills = document.querySelectorAll('.timeline-token');
      pills.forEach((p, idx) => {{
        if (idx === activeIndex) {{
          if (!p.classList.contains('active')) {{
            p.classList.add('active');
            p.scrollIntoView({{ behavior: 'smooth', block: 'nearest', inline: 'center' }});
          }}
        }} else {{
          p.classList.remove('active');
        }}
      }});
    }}

    function jumpToKeyframe(index) {{
      const kfs = currentPayload.keyframes;
      if (index >= 0 && index < kfs.length) {{
        currentTimeMs = kfs[index].start_time_ms;
        updateTimelineActive(index);
      }}
    }}

    buildTimeline();

    // Playback Controls
    const btnPlayPause = document.getElementById('btnPlayPause');
    btnPlayPause.addEventListener('click', () => {{
      isPlaying = !isPlaying;
      btnPlayPause.textContent = isPlaying ? '⏸️ Pause' : '▶️ Play';
    }});

    document.getElementById('btnReset').addEventListener('click', () => {{
      currentTimeMs = 0;
      isPlaying = true;
      btnPlayPause.textContent = '⏸️ Pause';
    }});

    document.getElementById('btnNext').addEventListener('click', () => {{
      const kfs = currentPayload.keyframes;
      const nextIdx = Math.min(currentKeyframeIndex + 1, kfs.length - 1);
      jumpToKeyframe(nextIdx);
    }});

    document.getElementById('btnPrev').addEventListener('click', () => {{
      const prevIdx = Math.max(0, currentKeyframeIndex - 1);
      jumpToKeyframe(prevIdx);
    }});

    document.getElementById('speedSelect').addEventListener('change', (e) => {{
      playbackSpeed = parseFloat(e.target.value);
    }});

    // Synthesize sentence on button click
    function synthesizeText(text) {{
      if (!text || !text.trim()) return;

      const statusBadge = document.getElementById('statusBadge');
      statusBadge.textContent = 'Synthesizing ASL...';

      // Dynamic client-side synthesis from full canonical library
      const clean = text.toUpperCase().replace(/[^A-Z0-9\s]/g, ' ').replace(/\s+/g, ' ').trim();
      const words = clean.split(' ');
      
      const keyframes = [];
      let timeMs = 0;
      const holdTime = 480;
      const transTime = 240;

      const restPose = canonicalLibrary['REST'] || Object.values(canonicalLibrary)[0];

      // Initial rest
      keyframes.push({{
        step_index: 0,
        sign: 'REST',
        display: 'Rest',
        type: 'posture',
        landmarks: restPose,
        start_time_ms: timeMs,
        hold_time_ms: 200,
        transition_time_ms: transTime,
        desc: 'Initial relaxed posture.'
      }});
      timeMs += 200 + transTime;

      words.forEach((w, wIdx) => {{
        // Whole word check
        if (canonicalLibrary[w] && w.length > 1) {{
          keyframes.push({{
            step_index: keyframes.length,
            sign: w,
            display: w,
            type: 'word',
            landmarks: canonicalLibrary[w],
            start_time_ms: timeMs,
            hold_time_ms: holdTime * 1.2,
            transition_time_ms: transTime,
            desc: `Whole-word sign for '${{w}}'.`
          }});
          timeMs += (holdTime * 1.2) + transTime;
        }} else {{
          // Fingerspell characters
          for (let ch of w) {{
            const pose = canonicalLibrary[ch] || restPose;
            keyframes.push({{
              step_index: keyframes.length,
              sign: ch,
              display: ch,
              type: 'letter',
              landmarks: pose,
              start_time_ms: timeMs,
              hold_time_ms: holdTime,
              transition_time_ms: transTime,
              desc: `Fingerspelled letter '${{ch}}'.`
            }});
            timeMs += holdTime + transTime;
          }}
        }}

        // Space between words
        if (wIdx < words.length - 1) {{
          keyframes.push({{
            step_index: keyframes.length,
            sign: 'SPACE',
            display: '␣',
            type: 'control',
            landmarks: canonicalLibrary['SPACE'] || restPose,
            start_time_ms: timeMs,
            hold_time_ms: 320,
            transition_time_ms: transTime,
            desc: 'Word separation gesture.'
          }});
          timeMs += 320 + transTime;
        }}
      }});

      // Concluding rest
      keyframes.push({{
        step_index: keyframes.length,
        sign: 'REST',
        display: 'Rest',
        type: 'posture',
        landmarks: restPose,
        start_time_ms: timeMs,
        hold_time_ms: 300,
        transition_time_ms: transTime,
        desc: 'Final relaxed posture.'
      }});
      timeMs += 300;

      currentPayload = {{
        input_text: text,
        keyframes: keyframes,
        total_duration_ms: timeMs,
        connections: currentPayload.connections
      }};

      currentTimeMs = 0;
      isPlaying = true;
      btnPlayPause.textContent = '⏸️ Pause';
      buildTimeline();
      statusBadge.textContent = `Performing: "${{clean.slice(0, 22)}}..."`;
    }}

    document.getElementById('btnTranslate').addEventListener('click', () => {{
      const txt = document.getElementById('sentenceInput').value;
      synthesizeText(txt);
    }});

    document.getElementById('sentenceInput').addEventListener('keydown', (e) => {{
      if (e.key === 'Enter') {{
        synthesizeText(e.target.value);
      }}
    }});

    window.applyPreset = function(text) {{
      document.getElementById('sentenceInput').value = text;
      synthesizeText(text);
    }};

    // Web Speech API Microphone Integration
    const btnMic = document.getElementById('btnMic');
    let recognition = null;
    let isListening = false;

    const SpeechRec = window.SpeechRecognition || window.webkitSpeechRecognition;

    if (SpeechRec) {{
      recognition = new SpeechRec();
      recognition.continuous = false;
      recognition.interimResults = true;
      recognition.lang = 'en-US';

      recognition.onstart = () => {{
        isListening = true;
        btnMic.classList.add('listening');
        btnMic.textContent = '🛑 Listening...';
        document.getElementById('statusBadge').textContent = 'Listening to microphone...';
      }};

      recognition.onresult = (e) => {{
        let transcript = '';
        for (let i = e.resultIndex; i < e.results.length; i++) {{
          transcript += e.results[i][0].transcript;
        }}
        document.getElementById('sentenceInput').value = transcript;
        if (e.results[0].isFinal) {{
          synthesizeText(transcript);
        }}
      }};

      recognition.onerror = (e) => {{
        console.warn('Speech Recognition error:', e.error);
        stopListening();
        document.getElementById('statusBadge').textContent = 'Mic: ' + e.error;
      }};

      recognition.onend = () => {{
        stopListening();
      }};

      function stopListening() {{
        isListening = false;
        btnMic.classList.remove('listening');
        btnMic.textContent = '🎙️ Speak Now';
      }}

      btnMic.addEventListener('click', () => {{
        if (isListening) {{
          recognition.stop();
        }} else {{
          try {{
            recognition.start();
          }} catch (err) {{
            console.error(err);
          }}
        }}
      }});
    }} else {{
      btnMic.title = 'Web Speech API is supported on Google Chrome, Edge, and modern browsers.';
      btnMic.addEventListener('click', () => {{
        alert('Web Speech API is not supported in this embedded browser view. Please type your phrase in the text box above!');
      }});
    }}
  </script>
</body>
</html>
"""
    return html_code


if __name__ == "__main__":
    code = generate_avatar_html()
    print(f"Generated Avatar HTML ({len(code)} characters).")
