/**
 * skeleton-renderer.js
 * ══════════════════════════════════════════════════════════════════════════════
 * Aashay's Sign Lang — Ghost Skeleton overlay renderer
 *
 * Draws the MediaPipe Holistic landmark skeleton on a transparent canvas
 * layered over the live video feed.  The "ghost" aesthetic uses low-opacity
 * cyan/white lines and dots so the signer can see their detected pose without
 * the overlay obscuring the video.
 *
 * Usage
 * ─────
 *  import { SkeletonRenderer } from './skeleton-renderer.js';
 *
 *  const renderer = new SkeletonRenderer(overlayCanvas);
 *  // Pass as onSkeleton callback to HolisticBridge:
 *  bridge = new HolisticBridge({ onSkeleton: (r) => renderer.draw(r), ... });
 */

// ── Connection topology (MediaPipe Holistic indices) ──────────────────────────

const POSE_CONNECTIONS = [
  // Torso
  [11,12],[11,23],[12,24],[23,24],
  // Left arm
  [11,13],[13,15],[15,17],[15,19],[15,21],[17,19],
  // Right arm
  [12,14],[14,16],[16,18],[16,20],[16,22],[18,20],
  // Left leg
  [23,25],[25,27],[27,29],[27,31],[29,31],
  // Right leg
  [24,26],[26,28],[28,30],[28,32],[30,32],
  // Face outline (simplified)
  [0,1],[1,2],[2,3],[3,7],[0,4],[4,5],[5,6],[6,8],[9,10],
];

const HAND_CONNECTIONS = [
  // Thumb
  [0,1],[1,2],[2,3],[3,4],
  // Index
  [0,5],[5,6],[6,7],[7,8],
  // Middle
  [0,9],[9,10],[10,11],[11,12],
  // Ring
  [0,13],[13,14],[14,15],[15,16],
  // Pinky
  [0,17],[17,18],[18,19],[19,20],
  // Palm
  [5,9],[9,13],[13,17],
];

// ── Style tokens ──────────────────────────────────────────────────────────────

const STYLE = {
  pose: {
    line:   'rgba(0, 230, 255, 0.55)',
    dot:    'rgba(0, 230, 255, 0.85)',
    dotR:   3,
    lineW:  1.5,
  },
  hand: {
    line:   'rgba(180, 255, 180, 0.65)',
    dot:    'rgba(180, 255, 180, 0.90)',
    dotR:   2.5,
    lineW:  1.2,
  },
  face: {
    dot:    'rgba(255, 255, 255, 0.18)',
    dotR:   1,
  },
  grid: {
    line:   'rgba(255, 255, 255, 0.06)',
    lineW:  0.5,
    cols:   6,
    rows:   4,
  },
};


// ── SkeletonRenderer ──────────────────────────────────────────────────────────

export class SkeletonRenderer {
  /**
   * @param {HTMLCanvasElement} canvas  Transparent overlay canvas
   * @param {object} [opts]
   * @param {boolean} [opts.showGrid]   Draw alignment grid (default true)
   * @param {boolean} [opts.showFace]   Draw face mesh dots (default true)
   */
  constructor(canvas, { showGrid = true, showFace = true } = {}) {
    this._canvas   = canvas;
    this._ctx      = canvas.getContext('2d');
    this._showGrid = showGrid;
    this._showFace = showFace;
  }

  // ── Public ──────────────────────────────────────────────────────────────────

  /** Draw one frame of skeleton data. Call from HolisticBridge.onSkeleton. */
  draw(results) {
    const { width: W, height: H } = this._canvas;
    const ctx = this._ctx;

    ctx.clearRect(0, 0, W, H);

    if (this._showGrid) this._drawGrid(W, H);
    if (results.faceLandmarks && this._showFace) {
      this._drawFaceDots(results.faceLandmarks, W, H);
    }
    if (results.poseLandmarks) {
      this._drawConnections(results.poseLandmarks, POSE_CONNECTIONS, W, H, STYLE.pose);
      this._drawDots(results.poseLandmarks, W, H, STYLE.pose);
    }
    if (results.leftHandLandmarks) {
      this._drawConnections(results.leftHandLandmarks, HAND_CONNECTIONS, W, H, STYLE.hand);
      this._drawDots(results.leftHandLandmarks, W, H, STYLE.hand);
    }
    if (results.rightHandLandmarks) {
      this._drawConnections(results.rightHandLandmarks, HAND_CONNECTIONS, W, H, STYLE.hand);
      this._drawDots(results.rightHandLandmarks, W, H, STYLE.hand);
    }
  }

  /** Clear the canvas (call on stop). */
  clear() {
    this._ctx.clearRect(0, 0, this._canvas.width, this._canvas.height);
  }

  // ── Private ─────────────────────────────────────────────────────────────────

  _drawGrid(W, H) {
    const ctx = this._ctx;
    const { cols, rows, line, lineW } = STYLE.grid;
    ctx.strokeStyle = line;
    ctx.lineWidth   = lineW;
    ctx.beginPath();
    for (let c = 1; c < cols; c++) {
      const x = (W / cols) * c;
      ctx.moveTo(x, 0);
      ctx.lineTo(x, H);
    }
    for (let r = 1; r < rows; r++) {
      const y = (H / rows) * r;
      ctx.moveTo(0, y);
      ctx.lineTo(W, y);
    }
    ctx.stroke();
  }

  _drawConnections(landmarks, connections, W, H, style) {
    const ctx = this._ctx;
    ctx.strokeStyle = style.line;
    ctx.lineWidth   = style.lineW;
    ctx.beginPath();
    for (const [a, b] of connections) {
      const lmA = landmarks[a];
      const lmB = landmarks[b];
      if (!lmA || !lmB) continue;
      ctx.moveTo(lmA.x * W, lmA.y * H);
      ctx.lineTo(lmB.x * W, lmB.y * H);
    }
    ctx.stroke();
  }

  _drawDots(landmarks, W, H, style) {
    const ctx = this._ctx;
    ctx.fillStyle = style.dot;
    for (const lm of landmarks) {
      if (!lm) continue;
      ctx.beginPath();
      ctx.arc(lm.x * W, lm.y * H, style.dotR, 0, Math.PI * 2);
      ctx.fill();
    }
  }

  _drawFaceDots(landmarks, W, H) {
    const ctx = this._ctx;
    ctx.fillStyle = STYLE.face.dot;
    // Draw every 4th face landmark to keep it lightweight
    for (let i = 0; i < landmarks.length; i += 4) {
      const lm = landmarks[i];
      if (!lm) continue;
      ctx.beginPath();
      ctx.arc(lm.x * W, lm.y * H, STYLE.face.dotR, 0, Math.PI * 2);
      ctx.fill();
    }
  }
}
