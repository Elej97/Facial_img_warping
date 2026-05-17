import Slider from '@react-native-community/slider';
import { useEffect, useRef, useState } from 'react';
import { Platform, Pressable, ScrollView, StyleSheet, Text, View } from 'react-native';

import { Ionicons } from '@expo/vector-icons';

type EffectId = 'smile' | 'slim' | 'brow' | 'lip';
type ProLiveOperation = 'smile_enhancement' | 'brow_lift' | 'lip_plump' | 'slim_face' | 'aging' | 'deaging';
type ProPreset = 'natural' | 'balanced' | 'strong';
type MakeupTarget = 'lip' | 'cheek' | 'bronzer' | 'lash' | 'brow';

type Anchor = { idx: number; dx: number; dy: number };

// Landmark indices follow MediaPipe FaceMesh 468-point spec.
// Deltas are in normalized image coords [0,1] at full intensity.
const EFFECTS: Record<EffectId, Anchor[]> = {
  smile: [
    { idx: 61, dx: -0.018, dy: -0.020 },
    { idx: 291, dx: 0.018, dy: -0.020 },
    { idx: 84, dx: -0.010, dy: -0.014 },
    { idx: 314, dx: 0.010, dy: -0.014 },
    { idx: 78, dx: -0.012, dy: -0.012 },
    { idx: 308, dx: 0.012, dy: -0.012 },
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
    { idx: 70, dx: 0, dy: -0.022 },
    { idx: 63, dx: 0, dy: -0.020 },
    { idx: 105, dx: 0, dy: -0.020 },
    { idx: 107, dx: 0, dy: -0.018 },
    { idx: 300, dx: 0, dy: -0.022 },
    { idx: 293, dx: 0, dy: -0.020 },
    { idx: 334, dx: 0, dy: -0.020 },
    { idx: 336, dx: 0, dy: -0.018 },
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
  smile_enhancement: 'Pro Smile',
  brow_lift: 'Pro Brow Lift',
  lip_plump: 'Pro Lip Plump',
  slim_face: 'Pro Slim Face',
  aging: 'Pro Aging',
  deaging: 'Pro De-Aging',
};

const PRO_ICON: Record<ProLiveOperation, keyof typeof Ionicons.glyphMap> = {
  smile_enhancement: 'happy-outline',
  brow_lift: 'arrow-up-outline',
  lip_plump: 'water-outline',
  slim_face: 'remove-outline',
  aging: 'time-outline',
  deaging: 'sparkles-outline',
};

const PRO_EFFECT_MAP: Partial<Record<ProLiveOperation, EffectId>> = {
  smile_enhancement: 'smile',
  brow_lift: 'brow',
  lip_plump: 'lip',
  slim_face: 'slim',
};

const PRO_PRESET_VALUES: Record<ProPreset, { intensity: number; smooth: number }> = {
  natural: { intensity: 0.3, smooth: 4.0 },
  balanced: { intensity: 0.6, smooth: 3.0 },
  strong: { intensity: 0.85, smooth: 2.0 },
};

const PRO_PRESET_LABEL: Record<ProPreset, string> = {
  natural: 'Natural',
  balanced: 'Balanced',
  strong: 'Strong',
};

const MAKEUP_PRESETS: { key: MakeupTarget; label: string; icon: keyof typeof Ionicons.glyphMap; defaultColor: string }[] = [
  { key: 'lip', label: 'Ruj', icon: 'water-outline', defaultColor: '#D45A73' },
  { key: 'cheek', label: 'Allık', icon: 'ellipse-outline', defaultColor: '#F29AAF' },
  { key: 'bronzer', label: 'Bronzer', icon: 'sunny-outline', defaultColor: '#B97A4C' },
  { key: 'lash', label: 'Kirpik', icon: 'eye-outline', defaultColor: '#1D1D1F' },
  { key: 'brow', label: 'Kaş', icon: 'remove-outline', defaultColor: '#5E4735' },
];

const MAKEUP_SWATCHES: Record<MakeupTarget, string[]> = {
  lip: ['#D45A73', '#A83253', '#F18FA7', '#BE4369', '#FF7AA2'],
  cheek: ['#F29AAF', '#F2B2A6', '#E88E7A', '#DB6F93', '#F7C1C8'],
  bronzer: ['#B97A4C', '#9F6642', '#D09A6B', '#7F5337', '#C88557'],
  lash: ['#1D1D1F', '#2E2E33', '#505057'],
  brow: ['#5E4735', '#463427', '#7A5B43', '#2D221A'],
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

type LiveWarpCameraProps = {
  onCapture?: (dataUrl: string, width: number, height: number) => void;
  isDark?: boolean;
};

export default function LiveWarpCamera({ onCapture, isDark = true }: LiveWarpCameraProps) {
  const videoRef = useRef<any>(null);
  const canvasRef = useRef<any>(null);
  const streamRef = useRef<any>(null);
  const landmarkerRef = useRef<any>(null);
  const rafRef = useRef<number | null>(null);
  const fpsTimeRef = useRef<number>(performance.now());
  const fpsCountRef = useRef<number>(0);

  const [running, setRunning] = useState(false);
  const [message, setMessage] = useState('Hazır. Başlat butonuna bas.');
  const [showLandmarks, setShowLandmarks] = useState(false);
  const [fps, setFps] = useState(0);
  const [proOperation, setProOperation] = useState<ProLiveOperation>('smile_enhancement');
  const [proPreset, setProPreset] = useState<ProPreset>('balanced');
  const [proIntensity, setProIntensity] = useState(PRO_PRESET_VALUES.balanced.intensity);
  const [proSmooth, setProSmooth] = useState(PRO_PRESET_VALUES.balanced.smooth);
  const [makeupTarget, setMakeupTarget] = useState<MakeupTarget>('lip');
  const [makeupColor, setMakeupColor] = useState(MAKEUP_PRESETS[0].defaultColor);
  const [makeupIntensity, setMakeupIntensity] = useState(0.48);
  const [makeupEnabled, setMakeupEnabled] = useState(true);

  const [intensities, setIntensities] = useState<Record<EffectId, number>>({
    smile: 0,
    slim: 0,
    brow: 0,
    lip: 0,
  });
  const intensitiesRef = useRef(intensities);
  const proRef = useRef({ operation: proOperation, intensity: proIntensity, smooth: proSmooth });
  const makeupRef = useRef({ target: makeupTarget, color: makeupColor, intensity: makeupIntensity, enabled: makeupEnabled });
  const showLandmarksRef = useRef(showLandmarks);

  useEffect(() => { intensitiesRef.current = intensities; }, [intensities]);
  useEffect(() => {
    proRef.current = { operation: proOperation, intensity: proIntensity, smooth: proSmooth };
  }, [proOperation, proIntensity, proSmooth]);
  useEffect(() => {
    makeupRef.current = { target: makeupTarget, color: makeupColor, intensity: makeupIntensity, enabled: makeupEnabled };
  }, [makeupTarget, makeupColor, makeupIntensity, makeupEnabled]);
  useEffect(() => { showLandmarksRef.current = showLandmarks; }, [showLandmarks]);

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

    if (canvas.width !== W) canvas.width = W;
    if (canvas.height !== H) canvas.height = H;

    const result = landmarker.detectForVideo(video, performance.now());
    const lm = result.faceLandmarks?.[0];

    const pro = proRef.current;
    ctx.setTransform(1, 0, 0, 1, 0, 0);
    ctx.filter = getLiveFilter(pro.operation, pro.intensity);
    ctx.drawImage(video, 0, 0, W, H);
    ctx.filter = 'none';

    if (!lm) {
      setMessage('Yüz aranıyor...');
      return;
    }

    const cur = intensitiesRef.current;
    const activeEffect = PRO_EFFECT_MAP[pro.operation];
    const liveEffects: Record<EffectId, number> = { ...cur };
    if (activeEffect) {
      liveEffects[activeEffect] = Math.max(liveEffects[activeEffect], pro.intensity);
    }

    const controls: { sx: number; sy: number; dxn: number; dyn: number }[] = [];
    (Object.keys(EFFECTS) as EffectId[]).forEach((effect) => {
      const intensity = liveEffects[effect];
      if (intensity < 0.005) return;
      for (const anchor of EFFECTS[effect]) {
        const lp = lm[anchor.idx];
        if (!lp) continue;
        controls.push({
          sx: lp.x,
          sy: lp.y,
          dxn: anchor.dx * intensity,
          dyn: anchor.dy * intensity,
        });
      }
    });

    if (controls.length > 0) {
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
            const w = Math.exp(-(rdx * rdx + rdy * rdy) / sig2);
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

          drawTri(ctx, video, srcG[k00], srcG[k01], srcG[k11], dstG[k00], dstG[k01], dstG[k11], W, H, pro.operation, pro.intensity);
          drawTri(ctx, video, srcG[k00], srcG[k11], srcG[k10], dstG[k00], dstG[k11], dstG[k10], W, H, pro.operation, pro.intensity);
        }
      }
    }

    if (pro.operation === 'aging') {
      drawAgingOverlay(ctx, lm, W, H, pro.intensity);
    } else if (pro.operation === 'deaging') {
      drawDeagingOverlay(ctx, lm, W, H, pro.intensity);
    }

    const makeup = makeupRef.current;
    if (makeup.enabled && makeup.intensity > 0.005) {
      drawMakeup(ctx, lm, W, H, makeup.target, makeup.color, makeup.intensity);
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
    const now = performance.now();
    if (now - fpsTimeRef.current > 800) {
      const f = (fpsCountRef.current * 1000) / (now - fpsTimeRef.current);
      fpsTimeRef.current = now;
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
    const dataUrl = canvas.toDataURL('image/png');
    onCapture?.(dataUrl, canvas.width, canvas.height);
  };

  const resetSliders = () => {
    setIntensities({ smile: 0, slim: 0, brow: 0, lip: 0 });
    setProOperation('smile_enhancement');
    setProPreset('balanced');
    setProIntensity(PRO_PRESET_VALUES.balanced.intensity);
    setProSmooth(PRO_PRESET_VALUES.balanced.smooth);
    setMakeupTarget('lip');
    setMakeupColor(MAKEUP_PRESETS[0].defaultColor);
    setMakeupIntensity(0.48);
    setMakeupEnabled(true);
  };

  const applyPreset = (preset: ProPreset) => {
    setProPreset(preset);
    setProIntensity(PRO_PRESET_VALUES[preset].intensity);
    setProSmooth(PRO_PRESET_VALUES[preset].smooth);
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
        <View style={[styles.stage, { backgroundColor: '#000', borderColor: panelBorder }]}>
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

        <View style={[styles.controls, { backgroundColor: panelBg, borderColor: panelBorder }]}>
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

            <Text style={[styles.sectionLabel, { color: muted }]}>PRO LAB</Text>
            <View style={styles.operationGrid}>
              {PRO_OPERATIONS.map((operation) => {
                const active = proOperation === operation;
                return (
                  <Pressable
                    key={operation}
                    onPress={() => setProOperation(operation)}
                    style={[
                      styles.operationButton,
                      {
                        backgroundColor: active ? accent : isDark ? 'rgba(255,255,255,0.07)' : 'rgba(0,0,0,0.05)',
                        borderColor: active ? accent : panelBorder,
                      },
                    ]}>
                    <Ionicons name={PRO_ICON[operation]} size={15} color={active ? '#fff' : accent} />
                    <Text style={[styles.operationText, { color: active ? '#fff' : text }]}>{PRO_LABEL[operation]}</Text>
                  </Pressable>
                );
              })}
            </View>

            <View style={styles.presetRow}>
              {(['natural', 'balanced', 'strong'] as ProPreset[]).map((preset) => {
                const active = proPreset === preset;
                return (
                  <Pressable
                    key={preset}
                    onPress={() => applyPreset(preset)}
                    style={[
                      styles.presetButton,
                      {
                        backgroundColor: active ? accent : isDark ? 'rgba(255,255,255,0.07)' : 'rgba(0,0,0,0.05)',
                        borderColor: active ? accent : panelBorder,
                      },
                    ]}>
                    <Text style={[styles.presetText, { color: active ? '#fff' : text }]}>{PRO_PRESET_LABEL[preset]}</Text>
                  </Pressable>
                );
              })}
            </View>

            <View style={styles.sliderBlock}>
              <View style={styles.sliderHeader}>
                <Text style={[styles.sliderLabel, { color: text }]}>Intensity</Text>
                <Text style={[styles.sliderValue, { color: muted }]}>{proIntensity.toFixed(2)}</Text>
              </View>
              <Slider
                value={proIntensity}
                onValueChange={setProIntensity}
                minimumValue={0}
                maximumValue={1}
                step={0.01}
                minimumTrackTintColor={accent}
                maximumTrackTintColor={isDark ? 'rgba(255,255,255,0.12)' : 'rgba(0,0,0,0.10)'}
                thumbTintColor={accent}
              />
            </View>

            <View style={styles.sliderBlock}>
              <View style={styles.sliderHeader}>
                <Text style={[styles.sliderLabel, { color: text }]}>RBF Smooth</Text>
                <Text style={[styles.sliderValue, { color: muted }]}>{proSmooth.toFixed(1)}</Text>
              </View>
              <Slider
                value={proSmooth}
                onValueChange={setProSmooth}
                minimumValue={0.8}
                maximumValue={10}
                step={0.1}
                minimumTrackTintColor={accent}
                maximumTrackTintColor={isDark ? 'rgba(255,255,255,0.12)' : 'rgba(0,0,0,0.10)'}
                thumbTintColor={accent}
              />
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
                onPress={() => setMakeupEnabled((value) => !value)}
                style={[
                  styles.miniSwitch,
                  { backgroundColor: makeupEnabled ? accent : isDark ? 'rgba(255,255,255,0.10)' : 'rgba(0,0,0,0.08)' },
                ]}>
                <View style={[styles.miniSwitchThumb, makeupEnabled ? styles.miniSwitchThumbOn : null]} />
              </Pressable>
            </View>

            <View style={styles.makeupGrid}>
              {MAKEUP_PRESETS.map((preset) => {
                const active = makeupTarget === preset.key;
                return (
                  <Pressable
                    key={preset.key}
                    onPress={() => {
                      setMakeupTarget(preset.key);
                      setMakeupColor(preset.defaultColor);
                      setMakeupEnabled(true);
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
                const active = makeupColor.toUpperCase() === swatch.toUpperCase();
                return (
                  <Pressable
                    key={swatch}
                    onPress={() => {
                      setMakeupColor(swatch);
                      setMakeupEnabled(true);
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
                <Text style={[styles.sliderValue, { color: muted }]}>{Math.round(makeupIntensity * 100)}%</Text>
              </View>
              <Slider
                value={makeupIntensity}
                onValueChange={(value) => {
                  setMakeupIntensity(value);
                  setMakeupEnabled(true);
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

            <View style={styles.actionRow}>
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

            <Text style={[styles.footnote, { color: muted }]}>
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
) {
  const m = getAffine(
    s0[0], s0[1], s1[0], s1[1], s2[0], s2[1],
    d0[0], d0[1], d1[0], d1[1], d2[0], d2[1],
  );
  if (!m) return;

  ctx.save();
  ctx.beginPath();
  ctx.moveTo(d0[0], d0[1]);
  ctx.lineTo(d1[0], d1[1]);
  ctx.lineTo(d2[0], d2[1]);
  ctx.closePath();
  ctx.clip();
  ctx.setTransform(m[0], m[1], m[2], m[3], m[4], m[5]);
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

function drawAgingOverlay(ctx: CanvasRenderingContext2D, lm: any[], W: number, H: number, intensity: number) {
  const alpha = 0.08 + intensity * 0.18;
  const drawLine = (indices: number[], width = 1) => {
    ctx.beginPath();
    indices.forEach((idx, index) => {
      const p = lm[idx];
      if (!p) return;
      if (index === 0) ctx.moveTo(p.x * W, p.y * H);
      else ctx.lineTo(p.x * W, p.y * H);
    });
    ctx.strokeStyle = `rgba(25,18,14,${alpha})`;
    ctx.lineWidth = width;
    ctx.lineCap = 'round';
    ctx.stroke();
  };

  ctx.save();
  ctx.setTransform(1, 0, 0, 1, 0, 0);
  drawLine([10, 338, 297, 332, 284], 1 + intensity);
  drawLine([10, 109, 67, 103, 54], 1 + intensity);
  drawLine([50, 101, 118, 117, 123], 0.8 + intensity * 0.8);
  drawLine([280, 330, 347, 346, 352], 0.8 + intensity * 0.8);
  drawLine([205, 187, 147, 123], 0.8 + intensity * 0.8);
  drawLine([425, 411, 376, 352], 0.8 + intensity * 0.8);
  ctx.restore();
}

function drawDeagingOverlay(ctx: CanvasRenderingContext2D, lm: any[], W: number, H: number, intensity: number) {
  let minX = 1, maxX = 0, minY = 1, maxY = 0;
  for (const p of lm) {
    minX = Math.min(minX, p.x);
    maxX = Math.max(maxX, p.x);
    minY = Math.min(minY, p.y);
    maxY = Math.max(maxY, p.y);
  }

  const x = minX * W;
  const y = minY * H;
  const width = (maxX - minX) * W;
  const height = (maxY - minY) * H;

  ctx.save();
  ctx.setTransform(1, 0, 0, 1, 0, 0);
  ctx.globalAlpha = 0.05 + intensity * 0.08;
  ctx.fillStyle = '#FFE7F2';
  ctx.beginPath();
  ctx.ellipse(x + width / 2, y + height * 0.48, width * 0.42, height * 0.34, 0, 0, Math.PI * 2);
  ctx.fill();
  ctx.globalAlpha = 1;
  ctx.restore();
}

function drawMakeup(
  ctx: CanvasRenderingContext2D,
  lm: any[],
  W: number,
  H: number,
  target: MakeupTarget,
  color: string,
  intensity: number,
) {
  ctx.save();
  ctx.setTransform(1, 0, 0, 1, 0, 0);
  ctx.globalCompositeOperation = target === 'lash' || target === 'brow' ? 'multiply' : 'soft-light';
  ctx.fillStyle = color;
  ctx.strokeStyle = color;

  if (target === 'lash' || target === 'brow') {
    ctx.globalAlpha = Math.min(0.82, 0.25 + intensity * 0.55);
    ctx.lineCap = 'round';
    ctx.lineJoin = 'round';
    ctx.lineWidth = target === 'lash' ? 1.2 + intensity * 2.2 : 3 + intensity * 4;
    for (const path of MAKEUP_PATHS[target]) {
      strokeLandmarkPath(ctx, lm, path, W, H);
    }
  } else {
    ctx.globalAlpha = Math.min(0.62, 0.12 + intensity * 0.48);
    if (target === 'lip') {
      fillLipMakeup(ctx, lm, W, H);
    } else {
      for (const path of MAKEUP_PATHS[target]) {
        fillLandmarkPath(ctx, lm, path, W, H);
      }
    }
  }

  ctx.globalCompositeOperation = 'source-over';
  ctx.globalAlpha = 1;
  ctx.restore();
}

function fillLipMakeup(ctx: CanvasRenderingContext2D, lm: any[], W: number, H: number) {
  const outerLip = MAKEUP_PATHS.lip[0];
  const innerMouth = MAKEUP_PATHS.lip[1];

  ctx.beginPath();
  outerLip.forEach((idx, index) => {
    const p = lm[idx];
    if (!p) return;
    if (index === 0) ctx.moveTo(p.x * W, p.y * H);
    else ctx.lineTo(p.x * W, p.y * H);
  });
  ctx.closePath();

  innerMouth
    .slice()
    .reverse()
    .forEach((idx, index) => {
      const p = lm[idx];
      if (!p) return;
      if (index === 0) ctx.moveTo(p.x * W, p.y * H);
      else ctx.lineTo(p.x * W, p.y * H);
    });
  ctx.closePath();
  ctx.fill('evenodd');
}

function fillLandmarkPath(ctx: CanvasRenderingContext2D, lm: any[], path: number[], W: number, H: number) {
  ctx.beginPath();
  path.forEach((idx, index) => {
    const p = lm[idx];
    if (!p) return;
    if (index === 0) ctx.moveTo(p.x * W, p.y * H);
    else ctx.lineTo(p.x * W, p.y * H);
  });
  ctx.closePath();
  ctx.fill();
}

function strokeLandmarkPath(ctx: CanvasRenderingContext2D, lm: any[], path: number[], W: number, H: number) {
  ctx.beginPath();
  path.forEach((idx, index) => {
    const p = lm[idx];
    if (!p) return;
    if (index === 0) ctx.moveTo(p.x * W, p.y * H);
    else ctx.lineTo(p.x * W, p.y * H);
  });
  ctx.stroke();
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
    minHeight: 42,
    borderRadius: 14,
    borderWidth: 1,
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'center',
    gap: 6,
    paddingHorizontal: 8,
  },
  operationText: {
    fontSize: 10,
    fontWeight: '900',
  },
  presetRow: {
    flexDirection: 'row',
    gap: 8,
  },
  presetButton: {
    flex: 1,
    height: 36,
    borderRadius: 13,
    borderWidth: 1,
    alignItems: 'center',
    justifyContent: 'center',
  },
  presetText: {
    fontSize: 10,
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
