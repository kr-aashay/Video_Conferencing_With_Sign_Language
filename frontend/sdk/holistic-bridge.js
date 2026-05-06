/**
 * holistic-bridge.js
 * ══════════════════════════════════════════════════════════════════════════════
 * Aashay's Sign Lang — MediaPipe Holistic ↔ FastAPI WebSocket bridge
 *
 * Responsibilities
 * ────────────────
 *  1. Open the user's camera via getUserMedia
 *  2. Run MediaPipe Holistic on every video frame (offscreen canvas)
 *  3. Flatten landmarks → 1,629-dim float32 vector
 *  4. Send the vector to the FastAPI /ws/stream endpoint
 *  5. Return the raw HTMLVideoElement for rendering
 *  6. Auto-reconnect WebSocket on drop (exponential back-off)
 *
 * Landmark layout (matches cslr_model/dataset.py FEAT_DIM = 1,629)
 * ─────────────────────────────────────────────────────────────────
 *  pose_xyz   33 × 3 =   99
 *  lhand      21 × 3 =   63
 *  rhand      21 × 3 =   63
 *  face      468 × 3 = 1,404
 *  ─────────────────────────
 *  total              1,629
 *
 * Usage
 * ─────
 *  import { HolisticBridge } from './holistic-bridge.js';
 *
 *  const bridge = new HolisticBridge({
 *    wsUrl:        'ws://localhost:8000/ws/stream/room1/alice/bob',
 *    onCaption:    ({ glosses, caption, latency_ms }) => { ... },
 *    onSkeleton:   (results) => { ... },   // for ghost-skeleton renderer
 *    onStatus:     (status) => { ... },    // 'connecting'|'open'|'closed'
 *  });
 *
 *  const videoEl = await bridge.start();   // returns <video> element
 *  // pass videoEl to Zego sendCustomVideoCaptureRawData
 *
 *  bridge.stop();
 */

// ── Constants ─────────────────────────────────────────────────────────────────
const FEAT_DIM        = 1629;
const POSE_LM_COUNT   = 33;
const FACE_LM_COUNT   = 468;
const HAND_LM_COUNT   = 21;

// Pose landmark indices for mid-hip normalisation (mirrors Python pipeline)
const LEFT_HIP_IDX    = 23;
const RIGHT_HIP_IDX   = 24;

// WebSocket reconnect strategy
const WS_INITIAL_DELAY_MS  = 500;
const WS_MAX_DELAY_MS      = 16_000;
const WS_BACKOFF_FACTOR    = 2;


// ── Landmark flattening ───────────────────────────────────────────────────────

/**
 * Flatten a MediaPipe NormalizedLandmarkList into a Float32Array of shape (N×3).
 * Returns zeros if the landmark list is null/undefined (hand not detected).
 */
function flattenLandmarks(landmarkList, count) {
  const out = new Float32Array(count * 3);
  if (!landmarkList) return out;
  const lms = landmarkList.landmark ?? landmarkList;
  for (let i = 0; i < Math.min(lms.length, count); i++) {
    out[i * 3]     = lms[i].x ?? 0;
    out[i * 3 + 1] = lms[i].y ?? 0;
    out[i * 3 + 2] = lms[i].z ?? 0;
  }
  return out;
}

/**
 * Build the 1,629-dim feature vector from a MediaPipe Holistic result.
 * Applies mid-hip pose-centre normalisation (translation invariance).
 *
 * Layout: [pose_xyz(99) | lhand(63) | rhand(63) | face(1404)]
 */
function buildFeatureVector(results) {
  const pose  = flattenLandmarks(results.poseLandmarks,      POSE_LM_COUNT);
  const lhand = flattenLandmarks(results.leftHandLandmarks,  HAND_LM_COUNT);
  const rhand = flattenLandmarks(results.rightHandLandmarks, HAND_LM_COUNT);
  const face  = flattenLandmarks(results.faceLandmarks,      FACE_LM_COUNT);

  // Mid-hip centre (landmarks 23 and 24 in pose)
  const cx = (pose[LEFT_HIP_IDX  * 3]     + pose[RIGHT_HIP_IDX  * 3])     / 2;
  const cy = (pose[LEFT_HIP_IDX  * 3 + 1] + pose[RIGHT_HIP_IDX  * 3 + 1]) / 2;
  const cz = (pose[LEFT_HIP_IDX  * 3 + 2] + pose[RIGHT_HIP_IDX  * 3 + 2]) / 2;

  // Subtract centre from every group
  function centreGroup(arr) {
    const out = new Float32Array(arr.length);
    for (let i = 0; i < arr.length; i += 3) {
      out[i]     = arr[i]     - cx;
      out[i + 1] = arr[i + 1] - cy;
      out[i + 2] = arr[i + 2] - cz;
    }
    return out;
  }

  const poseN  = centreGroup(pose);
  const lhandN = centreGroup(lhand);
  const rhandN = centreGroup(rhand);
  const faceN  = centreGroup(face);

  // Concatenate in the same order as Python pipeline
  const feat = new Float32Array(FEAT_DIM);
  let offset = 0;
  for (const arr of [poseN, lhandN, rhandN, faceN]) {
    feat.set(arr, offset);
    offset += arr.length;
  }
  return feat;
}


// ── HolisticBridge ────────────────────────────────────────────────────────────

export class HolisticBridge {
  /**
   * @param {object} opts
   * @param {string}   opts.wsUrl       Full WebSocket URL including path params
   * @param {function} opts.onCaption   Called with { glosses, caption, latency_ms }
   * @param {function} [opts.onSkeleton]  Called with raw MediaPipe results each frame
   * @param {function} [opts.onStatus]    Called with 'connecting'|'open'|'closed'|'error'
   * @param {object}   [opts.cameraConstraints]  getUserMedia video constraints
   * @param {number}   [opts.targetFps]  Target processing FPS (default 30)
   */
  constructor({
    wsUrl,
    onCaption,
    onSkeleton   = null,
    onStatus     = null,
    cameraConstraints = { width: { ideal: 1280 }, height: { ideal: 720 }, facingMode: 'user' },
    targetFps    = 30,
  }) {
    this._wsUrl             = wsUrl;
    this._onCaption         = onCaption;
    this._onSkeleton        = onSkeleton;
    this._onStatus          = onStatus;
    this._cameraConstraints = cameraConstraints;
    this._frameInterval     = 1000 / targetFps;

    this._ws              = null;
    this._wsDelay         = WS_INITIAL_DELAY_MS;
    this._wsReconnectTimer = null;
    this._running         = false;
    this._lastFrameTime   = 0;
    this._frameCount      = 0;
    this._droppedFrames   = 0;

    this._videoEl         = null;
    this._stream          = null;
    this._holistic        = null;
    this._offscreenCanvas = null;
    this._offscreenCtx    = null;
    this._rafHandle       = null;
  }

  // ── Public API ──────────────────────────────────────────────────────────────

  /**
   * Start camera, MediaPipe, and WebSocket.
   * @returns {Promise<HTMLVideoElement>} The live video element.
   */
  async start() {
    this._running = true;

    // Camera
    this._stream  = await navigator.mediaDevices.getUserMedia({
      video: this._cameraConstraints,
      audio: false,
    });
    this._videoEl = document.createElement('video');
    this._videoEl.srcObject = this._stream;
    this._videoEl.playsInline = true;
    this._videoEl.muted = true;
    await this._videoEl.play();

    // Offscreen canvas for MediaPipe (never shown in DOM)
    this._offscreenCanvas = document.createElement('canvas');
    this._offscreenCanvas.width  = this._videoEl.videoWidth  || 1280;
    this._offscreenCanvas.height = this._videoEl.videoHeight || 720;
    this._offscreenCtx = this._offscreenCanvas.getContext('2d', { willReadFrequently: false });

    // MediaPipe Holistic
    await this._initHolistic();

    // WebSocket
    this._connectWs();

    // Frame loop
    this._scheduleFrame();

    return this._videoEl;
  }

  /** Stop everything cleanly. */
  stop() {
    this._running = false;

    if (this._rafHandle) {
      cancelAnimationFrame(this._rafHandle);
      this._rafHandle = null;
    }
    if (this._wsReconnectTimer) {
      clearTimeout(this._wsReconnectTimer);
    }
    if (this._ws) {
      this._ws.onclose = null;   // prevent reconnect loop
      this._ws.close();
      this._ws = null;
    }
    if (this._holistic) {
      this._holistic.close();
      this._holistic = null;
    }
    if (this._stream) {
      this._stream.getTracks().forEach(t => t.stop());
      this._stream = null;
    }

    this._setStatus('closed');
  }

  /** Send a reset signal to the server (clears sliding window). */
  reset() {
    this._wsSend({ type: 'reset' });
  }

  /** Telemetry snapshot. */
  stats() {
    return {
      frameCount:    this._frameCount,
      droppedFrames: this._droppedFrames,
      wsState:       this._ws?.readyState ?? WebSocket.CLOSED,
    };
  }

  // ── MediaPipe ───────────────────────────────────────────────────────────────

  async _initHolistic() {
    // MediaPipe Holistic is loaded via CDN <script> tag in the HTML.
    // This guard ensures it's available before we proceed.
    if (typeof Holistic === 'undefined') {
      throw new Error(
        'MediaPipe Holistic not loaded. Add the CDN script tag before this module.'
      );
    }

    this._holistic = new Holistic({
      locateFile: (file) =>
        `https://cdn.jsdelivr.net/npm/@mediapipe/holistic@0.5.1675471629/${file}`,
    });

    this._holistic.setOptions({
      modelComplexity:        1,
      smoothLandmarks:        true,
      enableSegmentation:     false,
      refineFaceLandmarks:    true,
      minDetectionConfidence: 0.5,
      minTrackingConfidence:  0.5,
    });

    this._holistic.onResults((results) => this._onHolisticResults(results));

    // Warm-up: send one blank frame so the WASM model initialises
    await this._holistic.initialize();
  }

  // ── Frame loop ───────────────────────────────────────────────────────────────

  _scheduleFrame() {
    if (!this._running) return;
    this._rafHandle = requestAnimationFrame((ts) => this._onAnimationFrame(ts));
  }

  _onAnimationFrame(timestamp) {
    if (!this._running) return;

    const elapsed = timestamp - this._lastFrameTime;

    if (elapsed >= this._frameInterval) {
      this._lastFrameTime = timestamp - (elapsed % this._frameInterval);
      this._processFrame();
    }

    this._scheduleFrame();
  }

  async _processFrame() {
    if (!this._videoEl || this._videoEl.readyState < 2) return;
    if (!this._holistic) return;

    // Draw current video frame to offscreen canvas
    const { videoWidth: w, videoHeight: h } = this._videoEl;
    if (w !== this._offscreenCanvas.width || h !== this._offscreenCanvas.height) {
      this._offscreenCanvas.width  = w;
      this._offscreenCanvas.height = h;
    }
    this._offscreenCtx.drawImage(this._videoEl, 0, 0, w, h);

    // Send to MediaPipe (async — results arrive via onResults callback)
    try {
      await this._holistic.send({ image: this._offscreenCanvas });
    } catch (err) {
      console.warn('[HolisticBridge] MediaPipe send error:', err);
      this._droppedFrames++;
    }
  }

  _onHolisticResults(results) {
    this._frameCount++;

    // Notify skeleton renderer
    if (this._onSkeleton) {
      this._onSkeleton(results);
    }

    // Build feature vector and send to server
    const feat = buildFeatureVector(results);
    this._wsSend({ type: 'frame', data: Array.from(feat) });
  }

  // ── WebSocket ────────────────────────────────────────────────────────────────

  _connectWs() {
    if (!this._running) return;
    this._setStatus('connecting');

    const ws = new WebSocket(this._wsUrl);
    this._ws = ws;

    ws.onopen = () => {
      console.log('[HolisticBridge] WebSocket connected');
      this._wsDelay = WS_INITIAL_DELAY_MS;   // reset back-off
      this._setStatus('open');
    };

    ws.onmessage = (event) => {
      try {
        const msg = JSON.parse(event.data);
        if (msg.type === 'caption') {
          this._onCaption({
            glosses:    msg.glosses,
            caption:    msg.caption,
            latency_ms: msg.latency_ms,
          });
        }
        // 'ack' and 'error' messages are silently consumed here
      } catch (e) {
        console.warn('[HolisticBridge] Bad WS message:', event.data);
      }
    };

    ws.onerror = (err) => {
      console.error('[HolisticBridge] WebSocket error:', err);
      this._setStatus('error');
    };

    ws.onclose = (event) => {
      console.warn(`[HolisticBridge] WebSocket closed (code=${event.code}). Reconnecting in ${this._wsDelay}ms…`);
      this._setStatus('closed');
      if (this._running) {
        this._wsReconnectTimer = setTimeout(() => {
          this._wsDelay = Math.min(this._wsDelay * WS_BACKOFF_FACTOR, WS_MAX_DELAY_MS);
          this._connectWs();
        }, this._wsDelay);
      }
    };
  }

  _wsSend(payload) {
    if (this._ws?.readyState === WebSocket.OPEN) {
      this._ws.send(JSON.stringify(payload));
    }
    // Silently drop if not connected — frames will resume after reconnect
  }

  _setStatus(status) {
    if (this._onStatus) this._onStatus(status);
  }
}
