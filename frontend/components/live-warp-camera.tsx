import Slider from '@react-native-community/slider';
import { useEffect, useRef, useState } from 'react';
import { Platform, Pressable, StyleSheet, Text, View } from 'react-native';

import { Ionicons } from '@expo/vector-icons';

type EffectId = 'smile' | 'slim' | 'brow' | 'lip';

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

  const [intensities, setIntensities] = useState<Record<EffectId, number>>({
    smile: 0,
    slim: 0,
    brow: 0,
    lip: 0,
  });
  const intensitiesRef = useRef(intensities);
  const showLandmarksRef = useRef(showLandmarks);

  useEffect(() => { intensitiesRef.current = intensities; }, [intensities]);
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

    ctx.setTransform(1, 0, 0, 1, 0, 0);
    ctx.drawImage(video, 0, 0, W, H);

    if (!lm) {
      setMessage('Yüz aranıyor...');
      return;
    }

    const cur = intensitiesRef.current;
    const controls: { sx: number; sy: number; dxn: number; dyn: number }[] = [];
    (Object.keys(EFFECTS) as EffectId[]).forEach((effect) => {
      const intensity = cur[effect];
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
      const sig2 = SIGMA * SIGMA;

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

          drawTri(ctx, video, srcG[k00], srcG[k01], srcG[k11], dstG[k00], dstG[k01], dstG[k11], W, H);
          drawTri(ctx, video, srcG[k00], srcG[k11], srcG[k10], dstG[k00], dstG[k11], dstG[k10], W, H);
        }
      }
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

          {(Object.keys(EFFECTS) as EffectId[]).map((id) => (
            <View key={id} style={styles.sliderBlock}>
              <View style={styles.sliderHeader}>
                <View style={styles.sliderLabelGroup}>
                  <View style={[styles.iconBubble, { backgroundColor: 'rgba(160,32,240,0.16)' }]}>
                    <Ionicons name={EFFECT_META[id].icon} size={14} color={accent} />
                  </View>
                  <Text style={[styles.sliderLabel, { color: text }]}>{EFFECT_META[id].label}</Text>
                </View>
                <Text style={[styles.sliderValue, { color: muted }]}>
                  {Math.round(intensities[id] * 100)}
                </Text>
              </View>
              <Slider
                value={intensities[id]}
                onValueChange={(v) => setIntensities((s) => ({ ...s, [id]: v }))}
                minimumValue={0}
                maximumValue={1}
                step={0.01}
                minimumTrackTintColor={accent}
                maximumTrackTintColor={isDark ? 'rgba(255,255,255,0.12)' : 'rgba(0,0,0,0.10)'}
                thumbTintColor={accent}
              />
            </View>
          ))}

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
  ctx.drawImage(video, 0, 0, W, H);
  ctx.restore();
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
    padding: 18,
    gap: 14,
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
