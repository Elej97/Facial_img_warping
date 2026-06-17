import Slider from '@react-native-community/slider';
import { useEffect, useRef, useState } from 'react';
import { Platform, Pressable, ScrollView, StyleSheet, Text, View } from 'react-native';

import { frequencyProFromBase64 } from '@/services/facial-api';
import { Ionicons } from '@expo/vector-icons';

type EffectId = 'smile' | 'slim' | 'brow' | 'lip';
type ProLiveOperation = 'smile_enhancement' | 'brow_lift' | 'lip_plump' | 'slim_face' | 'aging' | 'deaging';
type MakeupTarget = 'lip' | 'cheek' | 'bronzer' | 'lash' | 'brow' | 'eye' | 'teeth';
type LandmarkPoint = { x: number; y: number; z?: number };
type MakeupProfile = { active: boolean; color: string; intensity: number };

type Anchor = { idx: number; dx: number; dy: number };

// Landmark indices follow MediaPipe FaceMesh 468-point spec.
// Deltas are in normalized image coords [0,1] at full intensity.
const EFFECTS: Record<EffectId, Anchor[]> = {
  smile: [
    { idx: 61, dx: -0.020, dy: -0.046 },
    { idx: 291, dx: 0.020, dy: -0.046 },
  ],
  slim: [
    { idx: 234, dx: 0.024, dy: 0 },
    { idx: 454, dx: -0.024, dy: 0 },
    { idx: 132, dx: 0.018, dy: 0.005 },
    { idx: 361, dx: -0.018, dy: 0.005 },
    { idx: 172, dx: 0.014, dy: 0 },
    { idx: 397, dx: -0.014, dy: 0 },
    { idx: 58, dx: 0.020, dy: 0 },
    { idx: 288, dx: -0.020, dy: 0 },
  ],
  brow: [
    { idx: 70, dx: 0, dy: -0.016 },
    { idx: 63, dx: 0, dy: -0.012 },
    { idx: 105, dx: 0, dy: -0.012 },
    { idx: 107, dx: 0, dy: -0.010 },
    { idx: 300, dx: 0, dy: -0.016 },
    { idx: 293, dx: 0, dy: -0.012 },
    { idx: 334, dx: 0, dy: -0.012 },
    { idx: 336, dx: 0, dy: -0.010 },
  ],
  lip: [
    { idx: 13, dx: 0, dy: -0.014 },
    { idx: 14, dx: 0, dy: 0.014 },
    { idx: 0, dx: 0, dy: -0.010 },
    { idx: 17, dx: 0, dy: 0.010 },
    { idx: 12, dx: 0, dy: -0.008 },
    { idx: 15, dx: 0, dy: 0.008 },
  ],
};

const EFFECT_SPREAD: Record<EffectId, number> = {
  smile: 0.16,
  slim: 1,
  brow: 0.18,
  lip: 0.45,
};

const EFFECT_META: Record<EffectId, { label: string; icon: keyof typeof Ionicons.glyphMap }> = {
  smile: { label: 'Gülümseme', icon: 'happy-outline' },
  slim: { label: 'Yüz İncelt', icon: 'remove-outline' },
  brow: { label: 'Kaş Kaldır', icon: 'arrow-up-outline' },
  lip: { label: 'Dudak Dolgun', icon: 'water-outline' },
};

const PRO_OPERATIONS: ProLiveOperation[] = [
  'smile_enhancement',
  'brow_lift',
  'lip_plump',
  'slim_face',
  'aging',
  'deaging',
];

const PRO_LABEL: Record<ProLiveOperation, string> = {
  smile_enhancement: 'Smile',
  brow_lift: 'Brow Lift',
  lip_plump: 'Lip Plump',
  slim_face: 'Slim Face',
  aging: 'Aging',
  deaging: 'De-Aging',
};

const PRO_ICON: Record<ProLiveOperation, keyof typeof Ionicons.glyphMap> = {
  smile_enhancement: 'happy-outline',
  brow_lift: 'arrow-up-outline',
  lip_plump: 'water-outline',
  slim_face: 'remove-outline',
  aging: 'time-outline',
  deaging: 'sparkles-outline',
};

function blobToBase64(blob: Blob) {
  return new Promise<string>((resolve, reject) => {
    const reader = new FileReader();
    reader.onloadend = () => {
      const result = reader.result;
      if (typeof result === 'string') {
        resolve(result.split(',')[1] ?? '');
      } else {
        reject(new Error('Failed to read preview blob'));
      }
    };
    reader.onerror = () => reject(reader.error ?? new Error('Failed to read preview blob'));
    reader.readAsDataURL(blob);
  });
}

const PRO_EFFECT_MAP: Partial<Record<ProLiveOperation, EffectId>> = {
  smile_enhancement: 'smile',
  brow_lift: 'brow',
  lip_plump: 'lip',
  slim_face: 'slim',
};

const LIVE_DETECT_INTERVAL_MS = 72;
const LIVE_AGING_INTERVAL_MS = 3600;
const LIVE_DEAGING_INTERVAL_MS = 4600;
const LIVE_AGING_TIMEOUT_MS = 2800;
const LIVE_LANDMARK_ALPHA = 0.72;
const IRIS_LEFT = [468, 469, 470, 471, 472];
const IRIS_RIGHT = [473, 474, 475, 476, 477];

const LAB_DEFAULT_INTENSITY = 50;
const LAB_INTENSITY_STEP = 5;
const LAB_SMOOTH = 2.8;

const MAKEUP_PRESETS: { key: MakeupTarget; label: string; icon: keyof typeof Ionicons.glyphMap; defaultColor: string }[] = [
  { key: 'lip', label: 'Ruj', icon: 'water-outline', defaultColor: '#D45A73' },
  { key: 'cheek', label: 'Allık', icon: 'ellipse-outline', defaultColor: '#F29AAF' },
  { key: 'bronzer', label: 'Bronzer', icon: 'sunny-outline', defaultColor: '#B97A4C' },
  { key: 'lash', label: 'Eyeliner', icon: 'eye-outline', defaultColor: '#1D1D1F' },
  { key: 'brow', label: 'Kaş', icon: 'remove-outline', defaultColor: '#5E4735' },
  { key: 'eye', label: 'Göz Rengi', icon: 'eye-outline', defaultColor: '#8B4513' },
  { key: 'teeth', label: 'Diş Beyazlatma', icon: 'sparkles-outline', defaultColor: '#FFFFFF' },
];

const MAKEUP_SWATCHES: Record<MakeupTarget, string[]> = {
  lip: ['#D45A73', '#A83253', '#F18FA7', '#BE4369', '#FF7AA2'],
  cheek: ['#F29AAF', '#F2B2A6', '#E88E7A', '#DB6F93', '#F7C1C8'],
  bronzer: ['#B97A4C', '#9F6642', '#D09A6B', '#7F5337', '#C88557'],
  lash: ['#1D1D1F', '#2E2E33', '#505057'],
  brow: ['#5E4735', '#463427', '#7A5B43', '#2D221A'],
  eye: ['#8B4513', '#1C3A70', '#2F5233', '#704214', '#1A1A2E'],
  teeth: ['#FFFFFF', '#F5F5F5', '#FFFACD', '#F0E68C', '#FAFAF0'],
};

const DEFAULT_MAKEUP_PROFILE: Record<MakeupTarget, MakeupProfile> = {
  lip: { active: false, color: '#D45A73', intensity: 0.48 },
  cheek: { active: false, color: '#F29AAF', intensity: 0.48 },
  bronzer: { active: false, color: '#B97A4C', intensity: 0.48 },
  lash: { active: false, color: '#1D1D1F', intensity: 0.48 },
  brow: { active: false, color: '#5E4735', intensity: 0.48 },
  eye: { active: false, color: '#8B4513', intensity: 0.48 },
  teeth: { active: false, color: '#FFFFFF', intensity: 0.48 },
};

const MAKEUP_PATHS: Record<MakeupTarget, number[][]> = {
  lip: [
    [61, 185, 40, 39, 37, 0, 267, 269, 270, 409, 291, 375, 321, 405, 314, 17, 84, 181, 91, 146],
    [78, 191, 80, 81, 82, 13, 312, 311, 310, 415, 308, 324, 318, 402, 317, 14, 87, 178, 88, 95],
  ],
  cheek: [
    [50, 101, 118, 117, 123, 147, 187, 205, 203, 206, 216, 212],
    [280, 330, 347, 346, 352, 376, 411, 425, 423, 426, 436, 432],
  ],
  bronzer: [
    [234, 93, 132, 58, 172, 136, 150, 149, 176],
    [454, 323, 361, 288, 397, 365, 379, 378, 400],
  ],
  lash: [
    [33, 246, 161, 160, 159, 158, 157, 173, 133],
    [263, 466, 388, 387, 386, 385, 384, 398, 362],
  ],
  brow: [
    [70, 63, 105, 66, 107],
    [300, 293, 334, 296, 336],
  ],
  eye: [
    [33, 246, 161, 160, 159, 158, 157, 173, 133],
    [263, 466, 388, 387, 386, 385, 384, 398, 362],
  ],
  teeth: [
    [78, 191, 80, 81, 82, 13, 312, 311, 310, 415, 308, 324, 318, 402, 317, 14, 87, 178, 88, 95],
  ],
};

const GRID_N = 18;
const SIGMA = 0.10;

function getAffine(
  sx1: number, sy1: number, sx2: number, sy2: number, sx3: number, sy3: number,
  dx1: number, dy1: number, dx2: number, dy2: number, dx3: number, dy3: number,
): [number, number, number, number, number, number] | null {
  const det = sx1 * (sy2 - sy3) + sx2 * (sy3 - sy1) + sx3 * (sy1 - sy2);
  if (Math.abs(det) < 1e-9) return null;

  const a = (dx1 * (sy2 - sy3) + dx2 * (sy3 - sy1) + dx3 * (sy1 - sy2)) / det;
  const b = (dy1 * (sy2 - sy3) + dy2 * (sy3 - sy1) + dy3 * (sy1 - sy2)) / det;
  const c = (dx1 * (sx3 - sx2) + dx2 * (sx1 - sx3) + dx3 * (sx2 - sx1)) / det;
  const d = (dy1 * (sx3 - sx2) + dy2 * (sx1 - sx3) + dy3 * (sx2 - sx1)) / det;
  const e = (dx1 * (sx2 * sy3 - sx3 * sy2) + dx2 * (sx3 * sy1 - sx1 * sy3) + dx3 * (sx1 * sy2 - sx2 * sy1)) / det;
  const f = (dy1 * (sx2 * sy3 - sx3 * sy2) + dy2 * (sx3 * sy1 - sx1 * sy3) + dy3 * (sx1 * sy2 - sx2 * sy1)) / det;

  return [a, b, c, d, e, f];
}

function clamp(value: number, min: number, max: number) {
  return Math.max(min, Math.min(max, value));
}

function hexToRgba(hex: string, alpha: number) {
  const normalized = hex.replace('#', '').trim();
  const full = normalized.length === 3
    ? normalized.split('').map((part) => part + part).join('')
    : normalized.padEnd(6, '0');
  const red = parseInt(full.slice(0, 2), 16);
  const green = parseInt(full.slice(2, 4), 16);
  const blue = parseInt(full.slice(4, 6), 16);
  return `rgba(${red}, ${green}, ${blue}, ${clamp(alpha, 0, 1)})`;
}

function isVeryDarkHex(hex: string) {
  const normalized = hex.replace('#', '').trim();
  const full = normalized.length === 3
    ? normalized.split('').map((part) => part + part).join('')
    : normalized.padEnd(6, '0');
  const red = parseInt(full.slice(0, 2), 16);
  const green = parseInt(full.slice(2, 4), 16);
  const blue = parseInt(full.slice(4, 6), 16);
  const luminance = (0.2126 * red) + (0.7152 * green) + (0.0722 * blue);
  return luminance <= 40;
}

function averagePoints(points: LandmarkPoint[]) {
  if (points.length === 0) {
    return { x: 0, y: 0 };
  }

  return points.reduce(
    (acc, point) => ({ x: acc.x + point.x, y: acc.y + point.y }),
    { x: 0, y: 0 }
  );
}

function getFaceBounds(lm: LandmarkPoint[]) {
  let minX = 1;
  let minY = 1;
  let maxX = 0;
  let maxY = 0;

  for (const point of lm) {
    minX = Math.min(minX, point.x);
    minY = Math.min(minY, point.y);
    maxX = Math.max(maxX, point.x);
    maxY = Math.max(maxY, point.y);
  }

  return { minX, minY, maxX, maxY, width: maxX - minX, height: maxY - minY };
}

function smoothLandmarks(previous: LandmarkPoint[] | null, next: LandmarkPoint[], alpha = LIVE_LANDMARK_ALPHA) {
  if (!previous || previous.length !== next.length) {
    return next.map((point) => ({ ...point }));
  }

  return next.map((point, index) => {
    const prev = previous[index];
    if (!prev) {
      return { ...point };
    }

    return {
      x: prev.x * alpha + point.x * (1 - alpha),
      y: prev.y * alpha + point.y * (1 - alpha),
      z: typeof prev.z === 'number' || typeof point.z === 'number'
        ? (prev.z ?? 0) * alpha + (point.z ?? 0) * (1 - alpha)
        : undefined,
    };
  });
}

function getIrisGeometry(lm: LandmarkPoint[], indices: number[], fallbackIndices: number[]) {
  const irisPoints = indices.map((idx) => lm[idx]).filter(Boolean) as LandmarkPoint[];
  if (irisPoints.length >= 3) {
    const center = averagePoints(irisPoints);
    const radius = irisPoints.reduce((maxRadius, point) => {
      const dx = point.x - center.x;
      const dy = point.y - center.y;
      return Math.max(maxRadius, Math.hypot(dx, dy));
    }, 0);

    return { center, radius: Math.max(radius * 0.82, 0.006) };
  }

  const fallbackPoints = fallbackIndices.map((idx) => lm[idx]).filter(Boolean) as LandmarkPoint[];
  if (fallbackPoints.length === 0) {
    return null;
  }

  const bounds = fallbackPoints.reduce(
    (acc, point) => ({
      minX: Math.min(acc.minX, point.x),
      minY: Math.min(acc.minY, point.y),
      maxX: Math.max(acc.maxX, point.x),
      maxY: Math.max(acc.maxY, point.y),
    }),
    { minX: 1, minY: 1, maxX: 0, maxY: 0 }
  );

  const center = {
    x: (bounds.minX + bounds.maxX) / 2,
    y: (bounds.minY + bounds.maxY) / 2,
  };
  const radius = Math.max(bounds.maxX - bounds.minX, bounds.maxY - bounds.minY) * 0.22;

  return { center, radius: Math.max(radius, 0.008) };
}

function getEyeBounds(lm: LandmarkPoint[], path: number[]) {
  const points = path.map((idx) => lm[idx]).filter(Boolean) as LandmarkPoint[];
  if (points.length < 3) {
    return null;
  }

  const bounds = points.reduce(
    (acc, point) => ({
      minX: Math.min(acc.minX, point.x),
      minY: Math.min(acc.minY, point.y),
      maxX: Math.max(acc.maxX, point.x),
      maxY: Math.max(acc.maxY, point.y),
    }),
    { minX: 1, minY: 1, maxX: 0, maxY: 0 }
  );

  return {
    center: {
      x: (bounds.minX + bounds.maxX) / 2,
      y: (bounds.minY + bounds.maxY) / 2,
    },
    width: bounds.maxX - bounds.minX,
    height: bounds.maxY - bounds.minY,
  };
}

function drawPath(ctx: CanvasRenderingContext2D, lm: LandmarkPoint[], path: number[], W: number, H: number) {
  const points = path.map((idx) => lm[idx]).filter(Boolean) as LandmarkPoint[];
  if (points.length < 2) {
    return;
  }

  ctx.beginPath();
  points.forEach((point, index) => {
    const x = point.x * W;
    const y = point.y * H;
    if (index === 0) {
      ctx.moveTo(x, y);
    } else {
      const prev = points[index - 1];
      const midX = (prev.x * W + x) / 2;
      const midY = (prev.y * H + y) / 2;
      ctx.quadraticCurveTo(prev.x * W, prev.y * H, midX, midY);
    }
  });
}

function fillPath(ctx: CanvasRenderingContext2D, lm: LandmarkPoint[], path: number[], W: number, H: number) {
  drawPath(ctx, lm, path, W, H);
  ctx.closePath();
  ctx.fill();
}

function strokePath(ctx: CanvasRenderingContext2D, lm: LandmarkPoint[], path: number[], W: number, H: number) {
  drawPath(ctx, lm, path, W, H);
  ctx.stroke();
}

function drawBrowMakeup(ctx: CanvasRenderingContext2D, lm: LandmarkPoint[], W: number, H: number, color: string, intensity: number) {
  const face = getFaceBounds(lm);
  const lift = face.height * (0.013 + intensity * 0.008);
  const thickness = face.height * (0.014 + intensity * 0.010);

  ctx.save();
  ctx.globalCompositeOperation = 'multiply';
  ctx.fillStyle = hexToRgba(color, 0.45 + intensity * 0.22);
  ctx.strokeStyle = hexToRgba(color, 0.80);
  ctx.lineCap = 'round';
  ctx.lineJoin = 'round';

  for (const path of MAKEUP_PATHS.brow) {
    const basePoints = path.map((idx) => lm[idx]).filter(Boolean) as LandmarkPoint[];
    if (basePoints.length < 3) {
      continue;
    }

    ctx.beginPath();
    basePoints.forEach((point, index) => {
      const x = point.x * W;
      const y = point.y * H;
      if (index === 0) {
        ctx.moveTo(x, y);
      } else {
        const prev = basePoints[index - 1];
        ctx.quadraticCurveTo(prev.x * W, prev.y * H, (prev.x * W + x) / 2, (prev.y * H + y) / 2);
      }
    });

    for (let i = basePoints.length - 1; i >= 0; i--) {
      const point = basePoints[i];
      const liftFactor = lift + Math.sin((i / Math.max(1, basePoints.length - 1)) * Math.PI) * thickness * 0.45;
      const x = point.x * W;
      const y = point.y * H - liftFactor;
      if (i === basePoints.length - 1) {
        ctx.lineTo(x, y);
      } else {
        const prev = basePoints[i + 1];
        const prevLift = lift + Math.sin(((i + 1) / Math.max(1, basePoints.length - 1)) * Math.PI) * thickness * 0.45;
        ctx.quadraticCurveTo((prev.x * W + x) / 2, ((prev.y * H - prevLift) + y) / 2, x, y);
      }
    }

    ctx.closePath();
    ctx.fill();

    ctx.globalAlpha = 0.30 + intensity * 0.28;
    ctx.lineWidth = 1.4 + intensity * 1.6;
    strokePath(ctx, lm, path, W, H);
    ctx.globalAlpha = 1;
  }

  ctx.restore();
}

function drawEyeColor(ctx: CanvasRenderingContext2D, lm: LandmarkPoint[], W: number, H: number, color: string, intensity: number) {
  const eyes = [
    { iris: IRIS_LEFT, fallback: MAKEUP_PATHS.eye[0], pupil: 468 },
    { iris: IRIS_RIGHT, fallback: MAKEUP_PATHS.eye[1], pupil: 473 },
  ];

  ctx.save();
  ctx.globalCompositeOperation = 'source-over';

  for (const [index, eye] of eyes.entries()) {
    const iris = getIrisGeometry(lm, eye.iris, eye.fallback);
    const eyeBounds = getEyeBounds(lm, eye.fallback);
    if (!iris || !eyeBounds) {
      continue;
    }

    const darkIris = isVeryDarkHex(color);

    const eyeShift = index === 0 ? -0.012 : -0.034;
    const cx = clamp(
      iris.center.x + eyeShift,
      eyeBounds.center.x - eyeBounds.width * 0.10,
      eyeBounds.center.x + eyeBounds.width * 0.18,
    ) * W;
    const cy = clamp(iris.center.y, eyeBounds.center.y - eyeBounds.height * 0.18, eyeBounds.center.y + eyeBounds.height * 0.18) * H;
    const irisRadius = iris.radius * Math.min(W, H) * (darkIris ? 1.74 + intensity * 0.08 : 1.58 + intensity * 0.10);
    const eyeCap = Math.min(eyeBounds.width * W * (darkIris ? 0.34 : 0.30), eyeBounds.height * H * (darkIris ? 0.48 : 0.44));
    const outerRadius = Math.max(4, Math.min(irisRadius, eyeCap));
    const innerRadius = Math.max(2, outerRadius * (darkIris ? 0.07 + intensity * 0.008 : 0.10 + intensity * 0.010));

    ctx.save();
    ctx.beginPath();
    ctx.ellipse(cx, cy, outerRadius, outerRadius * 1.02, 0, 0, Math.PI * 2);
    ctx.ellipse(cx, cy, innerRadius, innerRadius * 1.02, 0, 0, Math.PI * 2);
    ctx.clip('evenodd');

    const ring = ctx.createRadialGradient(cx, cy, innerRadius * 0.85, cx, cy, outerRadius);
    ring.addColorStop(0, 'rgba(0,0,0,0)');
    ring.addColorStop(0.22, hexToRgba(color, darkIris ? 0.38 + intensity * 0.08 : 0.26 + intensity * 0.08));
    ring.addColorStop(0.58, hexToRgba(color, darkIris ? 0.90 + intensity * 0.04 : 0.72 + intensity * 0.06));
    ring.addColorStop(1, hexToRgba(color, darkIris ? 0.99 : 0.96));

    ctx.globalCompositeOperation = darkIris ? 'source-over' : 'soft-light';
    ctx.globalAlpha = darkIris ? 0.90 : 0.96;
    ctx.fillStyle = ring;
    ctx.fillRect(cx - outerRadius, cy - outerRadius, outerRadius * 2, outerRadius * 2);

    ctx.restore();
  }

  ctx.restore();
}

function fillLipMakeup(ctx: CanvasRenderingContext2D, lm: LandmarkPoint[], W: number, H: number) {
  const outerLip = MAKEUP_PATHS.lip[0];
  const innerMouth = MAKEUP_PATHS.lip[1];

  ctx.beginPath();
  drawPath(ctx, lm, outerLip, W, H);
  ctx.closePath();

  const innerPoints = innerMouth.map((idx) => lm[idx]).filter(Boolean) as LandmarkPoint[];
  if (innerPoints.length > 0) {
    innerPoints
      .slice()
      .reverse()
      .forEach((point, index) => {
        const x = point.x * W;
        const y = point.y * H;
        if (index === 0) {
          ctx.moveTo(x, y);
        } else {
          const prev = innerPoints[innerPoints.length - index];
          ctx.lineTo((prev.x * W + x) / 2, (prev.y * H + y) / 2);
        }
      });
    ctx.closePath();
  }

  ctx.fill('evenodd');
}

function drawMakeup(
  ctx: CanvasRenderingContext2D,
  lm: LandmarkPoint[],
  W: number,
  H: number,
  target: MakeupTarget,
  color: string,
  intensity: number,
) {
  ctx.save();
  ctx.setTransform(1, 0, 0, 1, 0, 0);

  if (target === 'brow') {
    drawBrowMakeup(ctx, lm, W, H, color, intensity);
    ctx.restore();
    return;
  }

  if (target === 'eye') {
    drawEyeColor(ctx, lm, W, H, color, intensity);
    ctx.restore();
    return;
  }

  if (target === 'lash') {
    ctx.globalCompositeOperation = 'multiply';
    ctx.globalAlpha = Math.min(0.82, 0.26 + intensity * 0.52);
    ctx.strokeStyle = hexToRgba(color, 0.88);
    ctx.lineCap = 'round';
    ctx.lineJoin = 'round';
    ctx.lineWidth = 1.2 + intensity * 2.0;
    for (const path of MAKEUP_PATHS.lash) {
      strokePath(ctx, lm, path, W, H);
    }
    ctx.restore();
    return;
  }

  ctx.globalCompositeOperation = target === 'teeth' ? 'screen' : 'soft-light';
  ctx.fillStyle = hexToRgba(color, target === 'teeth' ? 0.52 : 0.62);
  ctx.strokeStyle = hexToRgba(color, 0.72);
  ctx.globalAlpha = Math.min(0.66, 0.16 + intensity * 0.52);

  if (target === 'lip') {
    fillLipMakeup(ctx, lm, W, H);
  } else {
    for (const path of MAKEUP_PATHS[target]) {
      fillPath(ctx, lm, path, W, H);
    }
  }

  ctx.globalAlpha = 1;
  ctx.restore();
}

function drawAgingOverlay(ctx: CanvasRenderingContext2D, lm: LandmarkPoint[], W: number, H: number, intensity: number) {
  const face = getFaceBounds(lm);
  const faceCx = ((face.minX + face.maxX) / 2) * W;
  const faceCy = ((face.minY + face.maxY) / 2) * H;
  const faceW = face.width * W;
  const faceH = face.height * H;

  ctx.save();
  ctx.setTransform(1, 0, 0, 1, 0, 0);
  ctx.globalCompositeOperation = 'multiply';

  const faceGlow = ctx.createRadialGradient(faceCx, faceCy * 0.96, 0, faceCx, faceCy, Math.max(faceW, faceH) * 0.55);
  faceGlow.addColorStop(0, `rgba(118, 76, 45, ${0.03 + intensity * 0.05})`);
  faceGlow.addColorStop(0.72, `rgba(92, 56, 34, ${0.02 + intensity * 0.03})`);
  faceGlow.addColorStop(1, 'rgba(0,0,0,0)');
  ctx.fillStyle = faceGlow;
  ctx.beginPath();
  ctx.ellipse(faceCx, faceCy, faceW * 0.43, faceH * 0.49, 0, 0, Math.PI * 2);
  ctx.fill();

  ctx.globalCompositeOperation = 'source-over';
  ctx.strokeStyle = `rgba(74, 42, 28, ${0.10 + intensity * 0.18})`;
  ctx.lineWidth = 0.7 + intensity * 1.0;
  ctx.lineCap = 'round';

  const wrinkleSets = [
    [70, 63, 105, 107, 65],
    [300, 293, 334, 336, 295],
    [33, 160, 158, 133],
    [263, 387, 385, 362],
    [205, 187, 147, 123],
    [425, 411, 376, 352],
  ];

  wrinkleSets.forEach((path, index) => {
    ctx.globalAlpha = clamp(0.08 + intensity * 0.14 - index * 0.012, 0.04, 0.28);
    strokePath(ctx, lm, path, W, H);
  });

  const cheekSpots = [50, 101, 118, 280, 330, 347];
  cheekSpots.forEach((idx, index) => {
    const point = lm[idx];
    if (!point) return;

    const radius = (faceW + faceH) * (0.004 + intensity * 0.0025) * (1 + index * 0.15);
    ctx.globalAlpha = 0.08 + intensity * 0.09;
    ctx.fillStyle = index % 2 === 0 ? 'rgba(105, 64, 43, 0.8)' : 'rgba(88, 52, 32, 0.7)';
    ctx.beginPath();
    ctx.arc(point.x * W, point.y * H, radius, 0, Math.PI * 2);
    ctx.fill();
  });

  ctx.restore();
}

function drawDeagingOverlay(ctx: CanvasRenderingContext2D, lm: LandmarkPoint[], W: number, H: number, intensity: number) {
  const face = getFaceBounds(lm);
  const faceCx = ((face.minX + face.maxX) / 2) * W;
  const faceCy = ((face.minY + face.maxY) / 2) * H;
  const faceW = face.width * W;
  const faceH = face.height * H;

  ctx.save();
  ctx.setTransform(1, 0, 0, 1, 0, 0);
  ctx.globalCompositeOperation = 'screen';

  const glows = [
    { x: faceCx, y: faceCy * 0.88, rx: faceW * 0.26, ry: faceH * 0.16, alpha: 0.06 + intensity * 0.04 },
    { x: faceCx - faceW * 0.12, y: faceCy * 1.02, rx: faceW * 0.18, ry: faceH * 0.12, alpha: 0.05 + intensity * 0.03 },
    { x: faceCx + faceW * 0.12, y: faceCy * 1.02, rx: faceW * 0.18, ry: faceH * 0.12, alpha: 0.05 + intensity * 0.03 },
    { x: faceCx, y: faceCy * 1.14, rx: faceW * 0.24, ry: faceH * 0.14, alpha: 0.04 + intensity * 0.03 },
  ];

  for (const glow of glows) {
    const grad = ctx.createRadialGradient(glow.x, glow.y, 0, glow.x, glow.y, Math.max(glow.rx, glow.ry));
    grad.addColorStop(0, `rgba(255, 236, 229, ${glow.alpha})`);
    grad.addColorStop(0.65, `rgba(255, 236, 229, ${glow.alpha * 0.32})`);
    grad.addColorStop(1, 'rgba(255, 236, 229, 0)');
    ctx.fillStyle = grad;
    ctx.beginPath();
    ctx.ellipse(glow.x, glow.y, glow.rx, glow.ry, 0, 0, Math.PI * 2);
    ctx.fill();
  }

  ctx.restore();
}

type LiveWarpCameraProps = {
  onCapture?: (dataUrl: string, width: number, height: number) => void;
  isDark?: boolean;
};

export default function LiveWarpCamera({ onCapture, isDark = true }: LiveWarpCameraProps) {
  const videoRef = useRef<any>(null);
  const canvasRef = useRef<any>(null);
  const previewCanvasRef = useRef<HTMLCanvasElement | null>(null);
  const agingPreviewImageRef = useRef<HTMLImageElement | null>(null);
  const streamRef = useRef<any>(null);
  const landmarkerRef = useRef<any>(null);
  const rafRef = useRef<number | null>(null);
  const fpsTimeRef = useRef<number>(performance.now());
  const fpsCountRef = useRef<number>(0);
  const lastDetectionAtRef = useRef<number>(0);
  const lastPreviewRequestAtRef = useRef<number>(0);
  const lastPreviewStartAtRef = useRef<number>(0);
  const previewRequestIdRef = useRef<number>(0);
  const previewInFlightRef = useRef(false);
  const lastFaceSeenAtRef = useRef<number>(0);

  const [running, setRunning] = useState(false);
  const [message, setMessage] = useState('Hazır. Başlat butonuna bas.');
  const [showLandmarks, setShowLandmarks] = useState(false);
  const [fps, setFps] = useState(0);
  const [activeProOperations, setActiveProOperations] = useState<ProLiveOperation[]>([]);
  const [proOperationIntensity, setProOperationIntensity] = useState<Record<ProLiveOperation, number>>({
    smile_enhancement: LAB_DEFAULT_INTENSITY,
    brow_lift: LAB_DEFAULT_INTENSITY,
    lip_plump: LAB_DEFAULT_INTENSITY,
    slim_face: LAB_DEFAULT_INTENSITY,
    aging: LAB_DEFAULT_INTENSITY,
    deaging: LAB_DEFAULT_INTENSITY,
  });
  const [hoveredProOperation, setHoveredProOperation] = useState<ProLiveOperation | null>(null);
  const [proLabEnabled, setProLabEnabled] = useState(false);
  const [makeupTarget, setMakeupTarget] = useState<MakeupTarget>('lip');
  const [makeupProfiles, setMakeupProfiles] = useState<Record<MakeupTarget, MakeupProfile>>(DEFAULT_MAKEUP_PROFILE);
  const [makeupEnabled, setMakeupEnabled] = useState(false);

  const [intensities, setIntensities] = useState<Record<EffectId, number>>({
    smile: 0,
    slim: 0,
    brow: 0,
    lip: 0,
  });
  const [splitScreen, setSplitScreen] = useState(true);
  const intensitiesRef = useRef(intensities);
  const proRef = useRef({ operations: activeProOperations, intensities: proOperationIntensity, smooth: LAB_SMOOTH });
  const proLabEnabledRef = useRef(proLabEnabled);
  const makeupRef = useRef({ target: makeupTarget, profiles: makeupProfiles, enabled: makeupEnabled });
  const showLandmarksRef = useRef(showLandmarks);
  const smoothedLandmarksRef = useRef<LandmarkPoint[] | null>(null);

  useEffect(() => { intensitiesRef.current = intensities; }, [intensities]);
  useEffect(() => {
    proRef.current = { operations: activeProOperations, intensities: proOperationIntensity, smooth: LAB_SMOOTH };
  }, [activeProOperations, proOperationIntensity]);
  useEffect(() => {
    proLabEnabledRef.current = proLabEnabled;
  }, [proLabEnabled]);
  useEffect(() => {
    makeupRef.current = { target: makeupTarget, profiles: makeupProfiles, enabled: makeupEnabled };
  }, [makeupTarget, makeupProfiles, makeupEnabled]);
  useEffect(() => { showLandmarksRef.current = showLandmarks; }, [showLandmarks]);

  useEffect(() => {
    if (!proLabEnabled || !activeProOperations.some((operation) => operation === 'aging' || operation === 'deaging')) {
      agingPreviewImageRef.current = null;
      previewInFlightRef.current = false;
      lastPreviewStartAtRef.current = 0;
    }
  }, [activeProOperations, proLabEnabled]);

  const init = async () => {
    if (landmarkerRef.current) return;
    setMessage('Yüz modeli yükleniyor...');

    const { FaceLandmarker, FilesetResolver } = await eval(
      `import("https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@latest/vision_bundle.js")`,
    );

    const vision = await FilesetResolver.forVisionTasks(
      'https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@latest/wasm',
    );

    landmarkerRef.current = await FaceLandmarker.createFromOptions(vision, {
      baseOptions: {
        modelAssetPath:
          'https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/latest/face_landmarker.task',
        delegate: 'GPU',
      },
      runningMode: 'VIDEO',
      numFaces: 1,
    });
  };

  const drawFrame = () => {
    const video = videoRef.current;
    const canvas = canvasRef.current;
    const landmarker = landmarkerRef.current;
    if (!video || !canvas || !landmarker) return;

    const W = video.videoWidth || 640;
    const H = video.videoHeight || 480;
    const ctx = canvas.getContext('2d', { alpha: false });
    if (!ctx) return;

    const canvasW = splitScreen ? W * 2 : W;
    if (canvas.width !== canvasW) canvas.width = canvasW;
    if (canvas.height !== H) canvas.height = H;

    const now = performance.now();
    const pro = proRef.current;
    const proEnabled = proLabEnabledRef.current;
    const activeLiveOperations = proEnabled
      ? pro.operations.filter((operation) => (pro.intensities[operation] ?? LAB_DEFAULT_INTENSITY) > 0)
      : [];
    const agingOperation = proEnabled
      ? (activeLiveOperations.find((operation) => operation === 'aging' || operation === 'deaging') as 'aging' | 'deaging' | undefined) ?? null
      : null;
    const agingMode = agingOperation !== null;
    const agingIntensity = agingOperation ? (pro.intensities[agingOperation] ?? LAB_DEFAULT_INTENSITY) / 100 : 0;
    const agingInterval = agingOperation === 'deaging' ? LIVE_DEAGING_INTERVAL_MS : LIVE_AGING_INTERVAL_MS;

    if (agingMode && previewInFlightRef.current && now - lastPreviewStartAtRef.current > LIVE_AGING_TIMEOUT_MS) {
      previewInFlightRef.current = false;
      lastPreviewStartAtRef.current = 0;
      previewRequestIdRef.current += 1;
    }

    ctx.setTransform(1, 0, 0, 1, 0, 0);

    if (agingMode) {
      if (splitScreen) {
        if (agingPreviewImageRef.current) {
          ctx.drawImage(agingPreviewImageRef.current, 0, 0, W, H);
        } else {
          ctx.filter = getLiveFilter(agingOperation, agingIntensity);
          ctx.drawImage(video, 0, 0, W, H);
          ctx.filter = 'none';
        }
        ctx.drawImage(video, W, 0, W, H);
      } else if (agingPreviewImageRef.current) {
        ctx.drawImage(agingPreviewImageRef.current, 0, 0, W, H);
      } else {
        ctx.filter = getLiveFilter(agingOperation, agingIntensity);
        ctx.drawImage(video, 0, 0, W, H);
        ctx.filter = 'none';
      }

      if (agingIntensity > 0.05 && !previewInFlightRef.current && now - lastPreviewRequestAtRef.current > agingInterval) {
        lastPreviewRequestAtRef.current = now;
        previewInFlightRef.current = true;
        lastPreviewStartAtRef.current = now;

        const previewCanvas = previewCanvasRef.current ?? document.createElement('canvas');
        previewCanvasRef.current = previewCanvas;
        const previewWidth = agingOperation === 'deaging'
          ? Math.min(168, Math.max(128, Math.round(W * 0.18)))
          : Math.min(180, Math.max(128, Math.round(W * 0.20)));
        const previewHeight = agingOperation === 'deaging'
          ? Math.min(180, Math.max(96, Math.round((H / Math.max(1, W)) * previewWidth)))
          : Math.min(192, Math.max(96, Math.round((H / Math.max(1, W)) * previewWidth)));
        previewCanvas.width = previewWidth;
        previewCanvas.height = previewHeight;

        const previewCtx = previewCanvas.getContext('2d');
        if (previewCtx) {
          previewCtx.setTransform(1, 0, 0, 1, 0, 0);
          previewCtx.drawImage(video, 0, 0, previewWidth, previewHeight);
          const requestId = ++previewRequestIdRef.current;

          previewCanvas.toBlob((blob) => {
            void (async () => {
              try {
                if (!blob) return;
                const previewBase64 = await blobToBase64(blob);

                if (!agingOperation) return;

                const data = await frequencyProFromBase64(previewBase64, agingOperation, agingIntensity, {
                  landmarkBackend: 'hybrid',
                  temporalSmoothing: false,
                  emaAlpha: 0.5,
                  streamId: 'live-pro-aging',
                });

                if (requestId !== previewRequestIdRef.current || !data?.success || !data.result_image_b64) {
                  return;
                }

                const image = new Image();
                image.onload = () => {
                  if (requestId !== previewRequestIdRef.current) {
                    return;
                  }

                  agingPreviewImageRef.current = image;
                };
                image.src = `data:image/png;base64,${data.result_image_b64}`;
              } catch {
                // Keep the current frame if the backend preview misses.
              } finally {
                if (requestId === previewRequestIdRef.current) {
                  previewInFlightRef.current = false;
                  lastPreviewStartAtRef.current = 0;
                }
              }
            })();
          }, 'image/jpeg', 0.34);
        } else {
          previewInFlightRef.current = false;
          lastPreviewStartAtRef.current = 0;
        }
      }
      return;
    }

    let lm = smoothedLandmarksRef.current;
    if (!lm || now - lastDetectionAtRef.current >= LIVE_DETECT_INTERVAL_MS) {
      const result = landmarker.detectForVideo(video, now);
      const detected = result.faceLandmarks?.[0] ?? null;
      lastDetectionAtRef.current = now;

      if (detected) {
        lm = smoothLandmarks(smoothedLandmarksRef.current, detected);
        smoothedLandmarksRef.current = lm;
        lastFaceSeenAtRef.current = now;
      } else if (lastFaceSeenAtRef.current && now - lastFaceSeenAtRef.current > 420) {
        lm = null;
        smoothedLandmarksRef.current = null;
      }
    }

    if (splitScreen) {
      // Sol taraf: orijinal kamera
      ctx.filter = 'none';
      ctx.drawImage(video, 0, 0, W, H);
      // Sağ taraf: efektli kamera (başlangıç)
      ctx.filter = 'none';
      ctx.drawImage(video, W, 0, W, H);
      ctx.filter = 'none';
      ctx.filter = 'none';
      ctx.drawImage(video, 0, 0, W, H);
      ctx.filter = 'none';
      ctx.drawImage(video, W, 0, W, H);
    } else {
      ctx.filter = 'none';
      ctx.drawImage(video, 0, 0, W, H);
      ctx.filter = 'none';
    }

    if (!lm) {
      if (now - lastFaceSeenAtRef.current > 420) {
        setMessage('Yüz aranıyor...');
      }
      return;
    }

    const cur = intensitiesRef.current;
    const liveEffects: Record<EffectId, number> = { ...cur };
    if (proEnabled) {
      for (const operation of activeLiveOperations) {
        const activeEffect = PRO_EFFECT_MAP[operation];
        if (activeEffect) {
          liveEffects[activeEffect] = Math.max(liveEffects[activeEffect], (pro.intensities[operation] ?? LAB_DEFAULT_INTENSITY) / 100);
        }
      }
    }

    const controls: { effect: EffectId; sx: number; sy: number; dxn: number; dyn: number; spread: number }[] = [];
    (Object.keys(EFFECTS) as EffectId[]).forEach((effect) => {
      const intensity = liveEffects[effect];
      if (intensity < 0.005) return;
      for (const anchor of EFFECTS[effect]) {
        const lp = lm[anchor.idx];
        if (!lp) continue;
        controls.push({
          effect,
          sx: lp.x,
          sy: lp.y,
          dxn: anchor.dx * intensity,
          dyn: anchor.dy * intensity,
          spread: EFFECT_SPREAD[effect],
        });
      }
    });

    if (controls.length > 0 && !agingMode) {
      let minX = 1, maxX = 0, minY = 1, maxY = 0;
      for (const p of lm) {
        if (p.x < minX) minX = p.x;
        if (p.x > maxX) maxX = p.x;
        if (p.y < minY) minY = p.y;
        if (p.y > maxY) maxY = p.y;
      }
      const pad = 0.12;
      const gMinX = Math.max(0, minX - pad);
      const gMaxX = Math.min(1, maxX + pad);
      const gMinY = Math.max(0, minY - pad);
      const gMaxY = Math.min(1, maxY + pad);

      const N = GRID_N;
      const cellU = (gMaxX - gMinX) / N;
      const cellV = (gMaxY - gMinY) / N;
      const sigma = SIGMA * (1.5 - Math.min(1, Math.max(0, pro.smooth / 10)));
      const sig2 = sigma * sigma;

      const srcG: number[][] = new Array((N + 1) * (N + 1));
      const dstG: number[][] = new Array((N + 1) * (N + 1));

      for (let i = 0; i <= N; i++) {
        for (let j = 0; j <= N; j++) {
          const sx = gMinX + j * cellU;
          const sy = gMinY + i * cellV;

          let dxAccum = 0;
          let dyAccum = 0;
          for (let k = 0; k < controls.length; k++) {
            const c = controls[k];
            const rdx = sx - c.sx;
            const rdy = sy - c.sy;

            let w = 0;
            if (c.effect === 'brow') {
              const lowerBleedLimit = c.sy + 0.010;
              if (sy > lowerBleedLimit) {
                continue;
              }

              const sigX = Math.max(0.018, sigma * 0.22);
              const sigY = Math.max(0.008, sigma * 0.08);
              w = Math.exp(-((rdx * rdx) / (sigX * sigX) + (rdy * rdy) / (sigY * sigY)));
            } else if (c.effect === 'smile') {
              if (sy < c.sy - 0.040 || sy > c.sy + 0.030) {
                continue;
              }

              const sigX = Math.max(0.020, sigma * 0.22);
              const sigY = Math.max(0.012, sigma * 0.14);
              w = Math.exp(-((rdx * rdx) / (sigX * sigX) + (rdy * rdy) / (sigY * sigY)));
            } else {
              const localSig = sigma * c.spread;
              const localSig2 = localSig * localSig;
              w = Math.exp(-(rdx * rdx + rdy * rdy) / localSig2);
            }

            dxAccum += c.dxn * w;
            dyAccum += c.dyn * w;
          }

          const k = i * (N + 1) + j;
          srcG[k] = [sx * W, sy * H];
          dstG[k] = [(sx + dxAccum) * W, (sy + dyAccum) * H];
        }
      }

      for (let i = 0; i < N; i++) {
        for (let j = 0; j < N; j++) {
          const k00 = i * (N + 1) + j;
          const k01 = k00 + 1;
          const k10 = k00 + (N + 1);
          const k11 = k10 + 1;

          const offsetX = 0;
          drawTri(ctx, video, srcG[k00], srcG[k01], srcG[k11], dstG[k00], dstG[k01], dstG[k11], W, H, activeLiveOperations[0] ?? 'smile_enhancement', 0, offsetX);
          drawTri(ctx, video, srcG[k00], srcG[k11], srcG[k10], dstG[k00], dstG[k11], dstG[k10], W, H, activeLiveOperations[0] ?? 'smile_enhancement', 0, offsetX);
        }
      }
    }

    const makeup = makeupRef.current;
    if (makeup.enabled) {
      (Object.keys(makeup.profiles) as MakeupTarget[]).forEach((target) => {
        const profile = makeup.profiles[target];
        if (profile.active && profile.intensity > 0.005) {
          drawMakeup(ctx, lm, W, H, target, profile.color, profile.intensity);
        }
      });
    }

    if (showLandmarksRef.current) {
      ctx.setTransform(1, 0, 0, 1, 0, 0);
      ctx.fillStyle = '#22d3ee';
      for (const p of lm) {
        ctx.beginPath();
        ctx.arc(p.x * W, p.y * H, 1.4, 0, Math.PI * 2);
        ctx.fill();
      }
    }

    fpsCountRef.current += 1;
    const fpsNow = performance.now();
    if (fpsNow - fpsTimeRef.current > 800) {
      const f = (fpsCountRef.current * 1000) / (fpsNow - fpsTimeRef.current);
      fpsTimeRef.current = fpsNow;
      fpsCountRef.current = 0;
      setFps(Math.round(f));
      setMessage(`Yüz tespit ✔ • ${lm.length} landmark`);
    }
  };

  const loop = () => {
    drawFrame();
    rafRef.current = requestAnimationFrame(loop);
  };

  const start = async () => {
    try {
      await init();
      const stream = await navigator.mediaDevices.getUserMedia({
        video: { width: { ideal: 960 }, height: { ideal: 720 }, facingMode: 'user' },
        audio: false,
      });
      streamRef.current = stream;
      const video = videoRef.current;
      video.srcObject = stream;
      await video.play();
      setRunning(true);
      setMessage('Kamera çalışıyor');
      fpsTimeRef.current = performance.now();
      fpsCountRef.current = 0;
      loop();
    } catch (err: any) {
      setMessage('Kamera açılamadı: ' + (err?.message ?? 'bilinmeyen hata'));
    }
  };

  const stop = () => {
    if (rafRef.current) {
      cancelAnimationFrame(rafRef.current);
      rafRef.current = null;
    }
    if (streamRef.current) {
      streamRef.current.getTracks().forEach((t: any) => t.stop());
      streamRef.current = null;
    }
    const video = videoRef.current;
    if (video) {
      video.pause();
      video.srcObject = null;
    }
    setRunning(false);
    setMessage('Durduruldu');
    setFps(0);
  };

  const capture = () => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    
    if (splitScreen) {
      // Split screen mode'da sadece sağ taraf (efektli) capture et
      const tempCanvas = document.createElement('canvas');
      const origW = canvas.width / 2;
      const H = canvas.height;
      tempCanvas.width = origW;
      tempCanvas.height = H;
      const tempCtx = tempCanvas.getContext('2d');
      if (tempCtx) {
        tempCtx.drawImage(canvas, 0, 0, origW, H, 0, 0, origW, H);
        const dataUrl = tempCanvas.toDataURL('image/png');
        onCapture?.(dataUrl, origW, H);
      }
    } else {
      const dataUrl = canvas.toDataURL('image/png');
      onCapture?.(dataUrl, canvas.width, canvas.height);
    }
  };

  const resetSliders = () => {
    setIntensities({ smile: 0, slim: 0, brow: 0, lip: 0 });
    setActiveProOperations([]);
    setHoveredProOperation(null);
    setProOperationIntensity({
      smile_enhancement: LAB_DEFAULT_INTENSITY,
      brow_lift: LAB_DEFAULT_INTENSITY,
      lip_plump: LAB_DEFAULT_INTENSITY,
      slim_face: LAB_DEFAULT_INTENSITY,
      aging: LAB_DEFAULT_INTENSITY,
      deaging: LAB_DEFAULT_INTENSITY,
    });
    setProLabEnabled(false);
    setMakeupTarget('lip');
    setMakeupProfiles(DEFAULT_MAKEUP_PROFILE);
    setMakeupEnabled(false);
    agingPreviewImageRef.current = null;
    smoothedLandmarksRef.current = null;
  };

  const activateProOperation = (operation: ProLiveOperation) => {
    setActiveProOperations((current) => {
      if (current.includes(operation)) {
        const next = current.filter((item) => item !== operation);
        setProLabEnabled(next.length > 0);
        return next;
      }

      setProOperationIntensity((values) => ({
        ...values,
        [operation]: LAB_DEFAULT_INTENSITY,
      }));
      const next = [...current, operation];
      setProLabEnabled(next.length > 0);
      return next;
    });
  };

  const adjustProOperationIntensity = (operation: ProLiveOperation, delta: number) => {
    setProOperationIntensity((values) => {
      const nextValue = clamp((values[operation] ?? LAB_DEFAULT_INTENSITY) + delta, 0, 100);
      if (nextValue === 0) {
        setActiveProOperations((current) => {
          const next = current.filter((item) => item !== operation);
          setProLabEnabled(next.length > 0);
          return next;
        });
      }

      return {
        ...values,
        [operation]: nextValue,
      };
    });
  };

  const updateProOperationIntensity = (operation: ProLiveOperation, value: number) => {
    setProOperationIntensity((values) => {
      const nextValue = clamp(value, 0, 100);
      if (nextValue === 0) {
        setActiveProOperations((current) => {
          const next = current.filter((item) => item !== operation);
          setProLabEnabled(next.length > 0);
          return next;
        });
      }

      return {
        ...values,
        [operation]: nextValue,
      };
    });
  };

  const toggleMakeup = () => {
    setMakeupEnabled((enabled) => {
      const nextEnabled = !enabled;
      if (nextEnabled) {
        setMakeupProfiles((current) => {
          const hasActiveProfile = (Object.keys(current) as MakeupTarget[]).some((target) => current[target].active);
          if (hasActiveProfile) {
            return current;
          }

          return {
            ...current,
            [makeupTarget]: {
              ...current[makeupTarget],
              active: true,
            },
          };
        });
      }
      return nextEnabled;
    });
  };

  useEffect(() => {
    return () => {
      if (rafRef.current) cancelAnimationFrame(rafRef.current);
      streamRef.current?.getTracks().forEach((t: any) => t.stop());
    };
  }, []);

  if (Platform.OS !== 'web') {
    return (
      <View style={[styles.container, { backgroundColor: isDark ? '#0A0B0D' : '#F7F4FB' }]}>
        <Text style={{ color: isDark ? '#fff' : '#111' }}>
          Canlı kamera modu şu an sadece web tarayıcıda kullanılabilir.
        </Text>
      </View>
    );
  }

  const accent = '#A020F0';
  const panelBg = isDark ? 'rgba(35,32,39,0.92)' : 'rgba(255,255,255,0.92)';
  const panelBorder = isDark ? 'rgba(255,255,255,0.10)' : 'rgba(20,20,20,0.08)';
  const text = isDark ? '#F4F1F6' : '#111217';
  const muted = isDark ? '#7D88A0' : '#657086';

  return (
    <View style={styles.container}>
      <View style={styles.layout}>
        <View style={[styles.stage, { backgroundColor: '#000', borderColor: panelBorder }, Platform.OS === 'web' ? ({ order: 2 } as any) : null]}>
          {/* @ts-ignore */}
          <video ref={videoRef} style={{ display: 'none' }} muted playsInline />
          {/* @ts-ignore */}
          <canvas ref={canvasRef} style={{ width: '100%', height: '100%', objectFit: 'contain', transform: 'scaleX(-1)' }} />

          <View style={styles.stageOverlay} pointerEvents="box-none">
            <View style={styles.topPills}>
              <View style={[styles.pill, { backgroundColor: running ? '#ef4444' : 'rgba(0,0,0,0.5)' }]}>
                {running ? <View style={styles.recDot} /> : null}
                <Text style={styles.pillText}>{running ? 'CANLI' : 'KAPALI'}</Text>
              </View>
              {running && (
                <View style={[styles.pill, { backgroundColor: 'rgba(0,0,0,0.5)' }]}>
                  <Text style={styles.pillText}>{fps} FPS</Text>
                </View>
              )}
            </View>

            <View style={styles.bottomBar} pointerEvents="box-none">
              <View style={[styles.statusChip, { backgroundColor: 'rgba(0,0,0,0.55)' }]}>
                <Text style={styles.statusChipText}>{message}</Text>
              </View>
            </View>
          </View>
        </View>

        <View style={[styles.controls, { backgroundColor: panelBg, borderColor: panelBorder }, Platform.OS === 'web' ? ({ order: 1 } as any) : null]}>
          <View style={[styles.actionRow, { paddingHorizontal: 20, paddingTop: 20, paddingBottom: 15, borderBottomWidth: 1, borderBottomColor: panelBorder }]}>
            <Pressable
              onPress={running ? stop : start}
              style={[styles.actionBtn, { backgroundColor: running ? '#ef4444' : accent }]}
            >
              <Ionicons
                name={running ? 'stop-circle-outline' : 'play-circle-outline'}
                size={18}
                color="#fff"
              />
              <Text style={styles.actionBtnText}>{running ? 'Durdur' : 'Başlat'}</Text>
            </Pressable>

            <Pressable
              onPress={capture}
              disabled={!running}
              style={[
                styles.actionBtn,
                {
                  backgroundColor: running ? '#0F172A' : 'rgba(0,0,0,0.30)',
                  opacity: running ? 1 : 0.5,
                  borderWidth: 1,
                  borderColor: panelBorder,
                },
              ]}
            >
              <Ionicons name="camera-outline" size={18} color="#fff" />
              <Text style={styles.actionBtnText}>Yakala & Düzenle</Text>
            </Pressable>
          </View>
          <ScrollView
            contentContainerStyle={styles.controlsScroll}
            showsVerticalScrollIndicator={false}
            showsHorizontalScrollIndicator={false}>
            <View style={styles.controlsHeader}>
              <View>
                <Text style={[styles.controlsTitle, { color: text }]}>Anlık Efektler</Text>
                <Text style={[styles.controlsSub, { color: accent }]}>Kaydırıcıyı çevir, yüzün canlı değişsin</Text>
              </View>
              <Pressable onPress={resetSliders} style={[styles.smallBtn, { borderColor: panelBorder }]}>
                <Ionicons name="refresh-outline" size={14} color={text} />
                <Text style={[styles.smallBtnText, { color: text }]}>Sıfırla</Text>
              </Pressable>
            </View>

            <View style={styles.sectionRow}>
              <Text style={[styles.sectionLabel, { color: muted }]}>LAB</Text>
            </View>
            <View style={styles.operationGrid}>
              {PRO_OPERATIONS.map((operation) => {
                const active = proLabEnabled && activeProOperations.includes(operation);
                const hovered = hoveredProOperation === operation;
                const intensity = proOperationIntensity[operation] ?? LAB_DEFAULT_INTENSITY;
                return (
                  <Pressable
                    key={operation}
                    onPress={active ? undefined : () => activateProOperation(operation)}
                    onHoverIn={() => setHoveredProOperation(operation)}
                    onHoverOut={() => setHoveredProOperation((current) => current === operation ? null : current)}
                    style={[
                      styles.operationButton,
                      {
                        backgroundColor: active ? accent : isDark ? 'rgba(255,255,255,0.07)' : 'rgba(0,0,0,0.05)',
                        borderColor: active ? active ? accent : isDark ? 'rgba(255,255,255,0.07)' : 'rgba(0,0,0,0.05)' : panelBorder,
                        paddingHorizontal: active ? 12 : 34,
                        paddingVertical: active ? 12 : 8,
                        minHeight: 98,
                      },
                    ]}>
                    <Pressable
                      style={styles.liveLabLabelWrap}
                      onPress={() => activateProOperation(operation)}>
                      <Ionicons name={PRO_ICON[operation]} size={15} color={active ? '#fff' : accent} />
                      <Text style={[styles.operationText, { color: active ? '#fff' : text }]}>{PRO_LABEL[operation]}</Text>
                      {active ? (
                        <Text style={[styles.liveLabValue, { color: '#fff' }]}>{intensity}%</Text>
                      ) : null}
                    </Pressable>
                    {active ? (
                      <Pressable
                        pointerEvents={hovered ? 'auto' : 'none'}
                        onHoverIn={() => setHoveredProOperation(operation)}
                        onPress={(event) => {
                          event.stopPropagation();
                          adjustProOperationIntensity(operation, -LAB_INTENSITY_STEP);
                        }}
                        style={[styles.liveLabAdjust, styles.liveLabAdjustLeft, { opacity: hovered ? 1 : 0 }]}>
                        <Text style={styles.liveLabAdjustText}>-</Text>
                      </Pressable>
                    ) : null}
                    {active ? (
                      <Pressable
                        pointerEvents={hovered ? 'auto' : 'none'}
                        onHoverIn={() => setHoveredProOperation(operation)}
                        onPress={(event) => {
                          event.stopPropagation();
                          adjustProOperationIntensity(operation, LAB_INTENSITY_STEP);
                        }}
                        style={[styles.liveLabAdjust, styles.liveLabAdjustRight, { opacity: hovered ? 1 : 0 }]}>
                        <Text style={styles.liveLabAdjustText}>+</Text>
                      </Pressable>
                    ) : null}
                    {active && (
                      <Slider
                        style={{ width: '100%', height: 30, marginTop: 4 }}
                        minimumValue={0}
                        maximumValue={100}
                        step={1}
                        value={intensity}
                        onValueChange={(val) => {
                          updateProOperationIntensity(operation, val);
                        }}
                        minimumTrackTintColor="#ffffff"
                        maximumTrackTintColor={isDark ? 'rgba(0,0,0,0.25)' : 'rgba(255,255,255,0.45)'}
                        thumbTintColor="#ffffff"
                      />
                    )}
                  </Pressable>
                );
              })}
            </View>

            <Text style={[styles.sectionLabel, { color: muted }]}>MANUEL KARIŞIM</Text>
            <View style={styles.manualGrid}>
              {(Object.keys(EFFECTS) as EffectId[]).map((id) => (
                <Pressable
                  key={id}
                  onPress={() => setIntensities((s) => ({ ...s, [id]: s[id] > 0 ? 0 : 0.45 }))}
                  style={[
                    styles.manualButton,
                    {
                      backgroundColor: intensities[id] > 0 ? 'rgba(160,32,240,0.18)' : isDark ? 'rgba(255,255,255,0.06)' : 'rgba(0,0,0,0.04)',
                      borderColor: intensities[id] > 0 ? accent : panelBorder,
                    },
                  ]}>
                  <Ionicons name={EFFECT_META[id].icon} size={14} color={accent} />
                  <Text style={[styles.manualText, { color: text }]}>{EFFECT_META[id].label}</Text>
                </Pressable>
              ))}
            </View>

            <View style={styles.sectionRow}>
              <Text style={[styles.sectionLabel, { color: muted }]}>MAKEUP</Text>
              <Pressable
                onPress={toggleMakeup}
                style={[
                  styles.miniSwitch,
                  { backgroundColor: makeupEnabled ? accent : isDark ? 'rgba(255,255,255,0.10)' : 'rgba(0,0,0,0.08)' },
                ]}>
                <View style={[styles.miniSwitchThumb, makeupEnabled ? styles.miniSwitchThumbOn : null]} />
              </Pressable>
            </View>

            <View style={styles.makeupGrid}>
              {MAKEUP_PRESETS.map((preset) => {
                const profile = makeupProfiles[preset.key];
                const active = profile.active;
                return (
                  <Pressable
                    key={preset.key}
                    onPress={() => {
                      setMakeupEnabled(true);
                      setMakeupTarget((current) => (current === preset.key ? current : preset.key));
                      setMakeupProfiles((current) => ({
                        ...current,
                        [preset.key]: {
                          ...current[preset.key],
                          active: !current[preset.key].active,
                          color: current[preset.key].color || preset.defaultColor,
                        },
                      }));
                    }}
                    style={[
                      styles.makeupButton,
                      {
                        backgroundColor: active ? accent : isDark ? 'rgba(255,255,255,0.07)' : 'rgba(0,0,0,0.05)',
                        borderColor: active ? accent : panelBorder,
                      },
                    ]}>
                    <Ionicons name={preset.icon} size={14} color={active ? '#fff' : accent} />
                    <Text style={[styles.makeupText, { color: active ? '#fff' : text }]}>{preset.label}</Text>
                  </Pressable>
                );
              })}
            </View>

            <View style={styles.swatchRow}>
              {(MAKEUP_SWATCHES[makeupTarget] ?? []).map((swatch) => {
                const profile = makeupProfiles[makeupTarget];
                const active = profile.active && profile.color.toUpperCase() === swatch.toUpperCase();
                return (
                  <Pressable
                    key={swatch}
                    onPress={() => {
                      setMakeupEnabled(true);
                      setMakeupProfiles((current) => ({
                        ...current,
                        [makeupTarget]: {
                          ...current[makeupTarget],
                          active: !(current[makeupTarget].active && current[makeupTarget].color.toUpperCase() === swatch.toUpperCase()),
                          color: swatch,
                        },
                      }));
                    }}
                    style={[
                      styles.swatchButton,
                      {
                        backgroundColor: swatch,
                        borderColor: active ? accent : isDark ? 'rgba(255,255,255,0.70)' : 'rgba(17,18,23,0.22)',
                        transform: [{ scale: active ? 1.08 : 1 }],
                      },
                    ]}
                  />
                );
              })}
            </View>

            <View style={styles.sliderBlock}>
              <View style={styles.sliderHeader}>
                <Text style={[styles.sliderLabel, { color: text }]}>Makeup Intensity</Text>
                <Text style={[styles.sliderValue, { color: muted }]}>{Math.round(makeupProfiles[makeupTarget].intensity * 100)}%</Text>
              </View>
              <Slider
                value={makeupProfiles[makeupTarget].intensity}
                onValueChange={(value) => {
                  setMakeupProfiles((current) => ({
                    ...current,
                    [makeupTarget]: {
                      ...current[makeupTarget],
                      active: true,
                      intensity: value,
                    },
                  }));
                }}
                minimumValue={0}
                maximumValue={1}
                step={0.01}
                minimumTrackTintColor={accent}
                maximumTrackTintColor={isDark ? 'rgba(255,255,255,0.12)' : 'rgba(0,0,0,0.10)'}
                thumbTintColor={accent}
              />
            </View>

            <View style={[styles.toggleRow, { borderColor: panelBorder }]}>
              <Text style={[styles.toggleLabel, { color: text }]}>Landmark Göster</Text>
              <Pressable
                onPress={() => setShowLandmarks((v) => !v)}
                style={[
                  styles.toggleSwitch,
                  { backgroundColor: showLandmarks ? accent : isDark ? 'rgba(255,255,255,0.10)' : 'rgba(0,0,0,0.08)' },
                ]}
              >
                <View style={[styles.toggleThumb, showLandmarks ? styles.toggleThumbOn : null]} />
              </Pressable>
            </View>

            <Text style={[styles.footnote, { color: muted, marginTop: 20 }]}>
              “Yakala & Düzenle”ye basınca anlık görüntü, fotoğraf düzenleme sekmesine aktarılır
              ve yaşlandırma, ifade transferi gibi HQ efektler oradan uygulanır.
            </Text>
          </ScrollView>
        </View>
      </View>
    </View>
  );
}

function drawTri(
  ctx: CanvasRenderingContext2D,
  video: any,
  s0: number[], s1: number[], s2: number[],
  d0: number[], d1: number[], d2: number[],
  W: number, H: number,
  operation: ProLiveOperation,
  intensity: number,
  offsetX: number = 0,
) {
  const m = getAffine(
    s0[0], s0[1], s1[0], s1[1], s2[0], s2[1],
    d0[0], d0[1], d1[0], d1[1], d2[0], d2[1],
  );
  if (!m) return;

  ctx.save();
  ctx.beginPath();
  ctx.moveTo(d0[0] + offsetX, d0[1]);
  ctx.lineTo(d1[0] + offsetX, d1[1]);
  ctx.lineTo(d2[0] + offsetX, d2[1]);
  ctx.closePath();
  ctx.clip();
  ctx.setTransform(m[0], m[1], m[2], m[3], m[4] + offsetX, m[5]);
  ctx.filter = getLiveFilter(operation, intensity);
  ctx.drawImage(video, 0, 0, W, H);
  ctx.filter = 'none';
  ctx.restore();
}

function getLiveFilter(operation: ProLiveOperation, intensity: number) {
  if (operation === 'aging') {
    const contrast = 1 + intensity * 0.18;
    const saturate = 1 - intensity * 0.28;
    const brightness = 1 - intensity * 0.04;
    return `contrast(${contrast}) saturate(${saturate}) brightness(${brightness})`;
  }

  if (operation === 'deaging') {
    const saturate = 1 + intensity * 0.12;
    const brightness = 1 + intensity * 0.06;
    return `brightness(${brightness}) saturate(${saturate})`;
  }

  return 'none';
}

const styles = StyleSheet.create({
  container: {
    flex: 1,
    width: '100%',
  },
  layout: {
    flex: 1,
    flexDirection: 'row',
    gap: 18,
    width: '100%',
  },
  stage: {
    flex: 1.6,
    borderRadius: 28,
    borderWidth: 1,
    overflow: 'hidden',
    minHeight: 420,
    position: 'relative',
  },
  stageOverlay: {
    ...StyleSheet.absoluteFillObject,
    padding: 16,
    justifyContent: 'space-between',
  },
  topPills: {
    flexDirection: 'row',
    gap: 8,
  },
  pill: {
    flexDirection: 'row',
    alignItems: 'center',
    gap: 6,
    paddingHorizontal: 12,
    paddingVertical: 6,
    borderRadius: 999,
  },
  recDot: {
    width: 8,
    height: 8,
    borderRadius: 4,
    backgroundColor: '#fff',
  },
  pillText: {
    color: '#fff',
    fontSize: 11,
    fontWeight: '900',
    letterSpacing: 0.6,
  },
  bottomBar: {
    flexDirection: 'row',
    justifyContent: 'center',
  },
  statusChip: {
    paddingHorizontal: 14,
    paddingVertical: 8,
    borderRadius: 999,
  },
  statusChipText: {
    color: '#fff',
    fontSize: 12,
    fontWeight: '600',
  },
  controls: {
    width: 360,
    minWidth: 320,
    borderRadius: 28,
    borderWidth: 1,
    padding: 0,
    overflow: 'hidden',
    alignSelf: 'stretch',
  },
  controlsScroll: {
    padding: 18,
    gap: 14,
    paddingBottom: 22,
  },
  controlsHeader: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'flex-start',
    paddingBottom: 10,
  },
  controlsTitle: {
    fontSize: 18,
    fontWeight: '900',
  },
  controlsSub: {
    fontSize: 11,
    fontWeight: '800',
    letterSpacing: 0.6,
    textTransform: 'uppercase',
    marginTop: 4,
  },
  smallBtn: {
    flexDirection: 'row',
    alignItems: 'center',
    gap: 6,
    paddingHorizontal: 10,
    paddingVertical: 6,
    borderRadius: 999,
    borderWidth: 1,
  },
  smallBtnText: {
    fontSize: 11,
    fontWeight: '700',
  },
  sliderBlock: {
    gap: 6,
  },
  sliderHeader: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'center',
  },
  sliderLabelGroup: {
    flexDirection: 'row',
    alignItems: 'center',
    gap: 8,
  },
  iconBubble: {
    width: 24,
    height: 24,
    borderRadius: 12,
    alignItems: 'center',
    justifyContent: 'center',
  },
  sliderLabel: {
    fontSize: 13,
    fontWeight: '700',
  },
  sliderValue: {
    fontSize: 11,
    fontWeight: '700',
    minWidth: 24,
    textAlign: 'right',
  },
  sectionLabel: {
    fontSize: 10,
    fontWeight: '900',
    letterSpacing: 0.9,
    marginTop: 2,
  },
  sectionRow: {
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'space-between',
    marginTop: 2,
  },
  operationGrid: {
    flexDirection: 'row',
    flexWrap: 'wrap',
    gap: 8,
  },
  operationButton: {
    width: '48%',
    minHeight: 58,
    borderRadius: 14,
    borderWidth: 1,
    alignItems: 'center',
    justifyContent: 'center',
    paddingHorizontal: 34,
    paddingVertical: 8,
    position: 'relative',
  },
  operationText: {
    fontSize: 10,
    lineHeight: 12,
    fontWeight: '900',
  },
  liveLabLabelWrap: {
    minHeight: 38,
    alignItems: 'center',
    justifyContent: 'center',
    gap: 3,
  },
  liveLabValue: {
    fontSize: 10,
    lineHeight: 12,
    fontWeight: '900',
  },
  liveLabAdjust: {
    position: 'absolute',
    top: '50%',
    marginTop: -13,
    width: 26,
    height: 26,
    borderRadius: 13,
    backgroundColor: 'rgba(0,0,0,0.20)',
    alignItems: 'center',
    justifyContent: 'center',
    zIndex: 2,
  },
  liveLabAdjustLeft: {
    left: 7,
  },
  liveLabAdjustRight: {
    right: 7,
  },
  liveLabAdjustText: {
    color: '#fff',
    fontSize: 17,
    lineHeight: 20,
    fontWeight: '900',
  },
  manualGrid: {
    flexDirection: 'row',
    flexWrap: 'wrap',
    gap: 8,
  },
  manualButton: {
    width: '48%',
    minHeight: 38,
    borderRadius: 13,
    borderWidth: 1,
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'center',
    gap: 6,
    paddingHorizontal: 8,
  },
  manualText: {
    fontSize: 10,
    fontWeight: '800',
  },
  makeupGrid: {
    flexDirection: 'row',
    flexWrap: 'wrap',
    gap: 8,
  },
  makeupButton: {
    width: '31%',
    minHeight: 38,
    borderRadius: 13,
    borderWidth: 1,
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'center',
    gap: 5,
    paddingHorizontal: 6,
  },
  makeupText: {
    fontSize: 10,
    fontWeight: '900',
  },
  swatchRow: {
    flexDirection: 'row',
    flexWrap: 'wrap',
    gap: 8,
  },
  swatchButton: {
    width: 30,
    height: 30,
    borderRadius: 15,
    borderWidth: 2,
  },
  miniSwitch: {
    width: 34,
    height: 20,
    borderRadius: 10,
    padding: 2,
  },
  miniSwitchThumb: {
    width: 16,
    height: 16,
    borderRadius: 8,
    backgroundColor: '#fff',
  },
  miniSwitchThumbOn: {
    transform: [{ translateX: 14 }],
  },
  toggleRow: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'center',
    paddingTop: 8,
    paddingBottom: 4,
    borderTopWidth: 1,
    marginTop: 4,
  },
  toggleLabel: {
    fontSize: 12,
    fontWeight: '700',
  },
  toggleSwitch: {
    width: 38,
    height: 22,
    borderRadius: 11,
    padding: 2,
  },
  toggleThumb: {
    width: 18,
    height: 18,
    borderRadius: 9,
    backgroundColor: '#fff',
  },
  toggleThumbOn: {
    transform: [{ translateX: 16 }],
  },
  actionRow: {
    flexDirection: 'row',
    gap: 10,
    marginTop: 4,
  },
  actionBtn: {
    flex: 1,
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'center',
    gap: 6,
    paddingVertical: 12,
    borderRadius: 14,
  },
  actionBtnText: {
    color: '#fff',
    fontWeight: '800',
    fontSize: 13,
  },
  footnote: {
    fontSize: 10,
    fontWeight: '500',
    lineHeight: 14,
    marginTop: 4,
  },
});
