import { useEffect, useRef, useState } from 'react';
import { Platform, Pressable, StyleSheet, Text, View } from 'react-native';
import { AREngine, GlassesStyle } from './ar-engine';

type LandmarkPoint = { x: number; y: number; z?: number };

const DETECT_INTERVAL_MS = 60;
const SMOOTH_ALPHA = 0.68;

const smoothLandmarks = (previous: LandmarkPoint[] | null, next: LandmarkPoint[]) => {
  if (!previous || previous.length !== next.length) {
    return next.map((point) => ({ ...point }));
  }

  return next.map((point, index) => {
    const prev = previous[index];
    if (!prev) return { ...point };
    return {
      x: prev.x * SMOOTH_ALPHA + point.x * (1 - SMOOTH_ALPHA),
      y: prev.y * SMOOTH_ALPHA + point.y * (1 - SMOOTH_ALPHA),
      z: typeof prev.z === 'number' || typeof point.z === 'number'
        ? (prev.z ?? 0) * SMOOTH_ALPHA + (point.z ?? 0) * (1 - SMOOTH_ALPHA)
        : undefined,
    };
  });
};

type AccessoryState = { glasses: boolean; hat: boolean; earrings: boolean };

export default function WebRealtimeFace() {
  const videoRef = useRef<any>(null);
  const canvasRef = useRef<any>(null);
  const arCanvasRef = useRef<any>(null);
  const streamRef = useRef<any>(null);
  const landmarkerRef = useRef<any>(null);
  const arEngineRef = useRef<AREngine | null>(null);
  const rafRef = useRef<number | null>(null);

  const lastFrameRef = useRef<any>(null);
  const lastPointsRef = useRef<LandmarkPoint[] | null>(null);
  const lastSizeRef = useRef({ width: 800, height: 450 });
  const lastDetectAtRef = useRef(0);
  const lastSeenAtRef = useRef(0);

  const [running, setRunning] = useState(false);
  const [message, setMessage] = useState('Hazır');
  const [showLandmarks, setShowLandmarks] = useState(false);
  const showLandmarksRef = useRef(false);

  const [accessories, setAccessories] = useState<AccessoryState>({
    glasses: false,
    hat: false,
    earrings: false,
  });
  const accessoriesRef = useRef<AccessoryState>({ glasses: false, hat: false, earrings: false });

  const [glassesStyle, setGlassesStyleState] = useState<GlassesStyle>('classic');

  const changeGlassesStyle = (style: GlassesStyle) => {
    setGlassesStyleState(style);
    arEngineRef.current?.setGlassesStyle(style);
  };

  // Keep ref in sync and update the AR engine
  useEffect(() => {
    showLandmarksRef.current = showLandmarks;
  }, [showLandmarks]);

  useEffect(() => {
    accessoriesRef.current = accessories;
    arEngineRef.current?.setAccessories(accessories.glasses, accessories.hat, accessories.earrings);
  }, [accessories]);

  // Initialise Three.js AR engine once the AR canvas is in the DOM
  const initAR = () => {
    if (arEngineRef.current || !arCanvasRef.current) return;
    const engine = new AREngine(arCanvasRef.current as unknown as HTMLCanvasElement);
    const acc = accessoriesRef.current;
    engine.setAccessories(acc.glasses, acc.hat, acc.earrings);
    arEngineRef.current = engine;
  };

  const init = async () => {
    if (landmarkerRef.current) return;

    setMessage('Model yükleniyor...');

    const { FaceLandmarker, FilesetResolver } = await eval(
      `import("https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@latest/vision_bundle.js")`
    );

    const vision = await FilesetResolver.forVisionTasks(
      'https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@latest/wasm'
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

    setMessage('Model hazır');
  };

  const drawLandmarks = (ctx: any, points: any[], width: number, height: number) => {
    if (!showLandmarksRef.current) return;
    ctx.fillStyle = '#00ffff';
    for (const p of points) {
      ctx.beginPath();
      ctx.arc(p.x * width, p.y * height, 1.5, 0, Math.PI * 2);
      ctx.fill();
    }
  };

  const redrawFrozenFrame = () => {
    const canvas = canvasRef.current;
    const frame = lastFrameRef.current;
    const points = lastPointsRef.current;
    const { width, height } = lastSizeRef.current;
    if (!canvas || !frame) return;
    const ctx = canvas.getContext('2d');
    if (!ctx) return;
    canvas.width = width;
    canvas.height = height;
    ctx.clearRect(0, 0, width, height);
    ctx.drawImage(frame, 0, 0, width, height);
    if (points) drawLandmarks(ctx, points, width, height);
  };

  const drawLiveFrame = () => {
    const video = videoRef.current;
    const canvas = canvasRef.current;
    const landmarker = landmarkerRef.current;
    if (!video || !canvas || !landmarker) return;

    const width = video.videoWidth || 800;
    const height = video.videoHeight || 450;
    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    canvas.width = width;
    canvas.height = height;
    lastSizeRef.current = { width, height };

    ctx.clearRect(0, 0, width, height);
    ctx.drawImage(video, 0, 0, width, height);

    const now = performance.now();
    let points = lastPointsRef.current;

    if (!points || now - lastDetectAtRef.current >= DETECT_INTERVAL_MS) {
      const result = landmarker.detectForVideo(video, now);
      const detected = result.faceLandmarks?.[0] ?? null;
      lastDetectAtRef.current = now;

      if (detected) {
        points = smoothLandmarks(lastPointsRef.current, detected);
        lastPointsRef.current = points;
        lastSeenAtRef.current = now;

        // Update 3D accessories — pass z values (MediaPipe provides them)
        const arEngine = arEngineRef.current;
        if (arEngine && points) {
          const lm3d = points.map(p => ({ x: p.x, y: p.y, z: p.z ?? 0 }));
          arEngine.update(lm3d, width, height);
        }
      } else if (now - lastSeenAtRef.current > 400) {
        points = null;
        lastPointsRef.current = null;
        arEngineRef.current?.clear();
      }
    } else if (points) {
      // Re-render accessories at last known position every frame for smoothness
      const arEngine = arEngineRef.current;
      if (arEngine) {
        const lm3d = points.map(p => ({ x: p.x, y: p.y, z: p.z ?? 0 }));
        arEngine.update(lm3d, width, height);
      }
    }

    lastFrameRef.current = video;

    if (points) {
      drawLandmarks(ctx, points, width, height);
      setMessage(`Yüz bulundu · ${points.length} nokta`);
    } else {
      setMessage('Yüz aranıyor...');
    }
  };

  const freezeCurrentFrame = async () => {
    const video = videoRef.current;
    if (!video) return;
    const width = video.videoWidth || 800;
    const height = video.videoHeight || 450;
    lastSizeRef.current = { width, height };
    const bitmap = await createImageBitmap(video);
    lastFrameRef.current = bitmap;
    redrawFrozenFrame();
  };

  const loop = () => {
    drawLiveFrame();
    rafRef.current = requestAnimationFrame(loop);
  };

  const start = async () => {
    await init();
    initAR();

    const stream = await navigator.mediaDevices.getUserMedia({
      video: { width: { ideal: 1280 }, height: { ideal: 720 } },
      audio: false,
    });

    streamRef.current = stream;
    const video = videoRef.current;
    video.srcObject = stream;
    await video.play();

    setRunning(true);
    loop();
  };

  const stop = async () => {
    if (rafRef.current) {
      cancelAnimationFrame(rafRef.current);
      rafRef.current = null;
    }
    await freezeCurrentFrame();
    arEngineRef.current?.clear();

    if (streamRef.current) {
      streamRef.current.getTracks().forEach((track: any) => track.stop());
      streamRef.current = null;
    }

    const video = videoRef.current;
    if (video) {
      video.pause();
      video.srcObject = null;
    }

    setRunning(false);
    setMessage('Durduruldu');
  };

  useEffect(() => {
    return () => {
      if (rafRef.current) cancelAnimationFrame(rafRef.current);
      streamRef.current?.getTracks().forEach((track: any) => track.stop());
      arEngineRef.current?.dispose();
    };
  }, []);

  if (Platform.OS !== 'web') {
    return (
      <View style={styles.center}>
        <Text>Web demo sadece tarayıcı için</Text>
      </View>
    );
  }

  const toggleAccessory = (key: keyof AccessoryState) => {
    setAccessories(prev => ({ ...prev, [key]: !prev[key] }));
  };

  return (
    <View style={styles.container}>
      <Text style={styles.title}>AR Yüz Filtresi</Text>
      <Text style={styles.message}>{message}</Text>

      {/* Camera + AR overlay */}
      <View style={styles.stage}>
        {/* @ts-ignore */}
        <video ref={videoRef} style={{ display: 'none' }} muted playsInline />
        {/* 2D canvas: video feed + optional landmark dots */}
        {/* @ts-ignore */}
        <canvas ref={canvasRef} style={styles.canvas} />
        {/* Three.js WebGL canvas: 3D accessories, overlaid transparently */}
        {/* @ts-ignore */}
        <canvas ref={arCanvasRef} style={styles.arCanvas} />
      </View>

      {/* Accessory toggles */}
      <View style={styles.accessoryRow}>
        <Pressable
          onPress={() => toggleAccessory('glasses')}
          style={[styles.accButton, accessories.glasses && styles.accButtonActive]}
        >
          <Text style={styles.accIcon}>👓</Text>
          <Text style={styles.accLabel}>Gözlük</Text>
        </Pressable>

        <Pressable
          onPress={() => toggleAccessory('hat')}
          style={[styles.accButton, accessories.hat && styles.accButtonActive]}
        >
          <Text style={styles.accIcon}>🎩</Text>
          <Text style={styles.accLabel}>Şapka</Text>
        </Pressable>

        <Pressable
          onPress={() => toggleAccessory('earrings')}
          style={[styles.accButton, accessories.earrings && styles.accButtonActive]}
        >
          <Text style={styles.accIcon}>💎</Text>
          <Text style={styles.accLabel}>Küpe</Text>
        </Pressable>
      </View>

      {/* Glasses style picker — visible only when glasses is active */}
      {accessories.glasses && (
        <View style={styles.styleRow}>
          {([
            { key: 'classic', label: 'Klasik', color: '#111122' },
            { key: 'round',   label: 'Altın',  color: '#d4a030' },
            { key: 'aviator', label: 'Aviator', color: '#b0b8c8' },
            { key: 'square',  label: 'Kare',    color: '#1a2233' },
          ] as { key: GlassesStyle; label: string; color: string }[]).map(({ key, label, color }) => (
            <Pressable
              key={key}
              onPress={() => changeGlassesStyle(key)}
              style={[styles.styleButton, glassesStyle === key && styles.styleButtonActive]}
            >
              <View style={[styles.styleColorDot, { backgroundColor: color }]} />
              <Text style={styles.styleLabel}>{label}</Text>
            </Pressable>
          ))}
        </View>
      )}

      {/* Camera + debug controls */}
      <View style={styles.actions}>
        <Pressable onPress={running ? stop : start} style={styles.button}>
          <Text style={styles.buttonText}>{running ? 'Durdur' : 'Başlat'}</Text>
        </Pressable>

        <Pressable
          onPress={() => {
            const next = !showLandmarks;
            showLandmarksRef.current = next;
            setShowLandmarks(next);
            if (!running) redrawFrozenFrame();
          }}
          style={[styles.button, styles.buttonSecondary]}
        >
          <Text style={styles.buttonText}>
            {showLandmarks ? 'Noktaları Gizle' : 'Noktaları Göster'}
          </Text>
        </Pressable>
      </View>
    </View>
  );
}

const styles = StyleSheet.create({
  container: {
    flex: 1,
    alignItems: 'center',
    justifyContent: 'center',
    gap: 14,
    backgroundColor: '#0d0d14',
    paddingVertical: 24,
  },
  title: {
    color: '#ffffff',
    fontSize: 22,
    fontWeight: '900',
    letterSpacing: 0.5,
  },
  message: {
    color: 'rgba(255,255,255,0.6)',
    fontSize: 13,
    fontWeight: '600',
  },
  stage: {
    width: 800,
    height: 450,
    borderRadius: 18,
    overflow: 'hidden',
    backgroundColor: '#000',
    position: 'relative',
  },
  canvas: {
    position: 'absolute',
    top: 0,
    left: 0,
    width: '100%',
    height: '100%',
  },
  // Three.js canvas sits on top — transparent bg, pointer-events: none so
  // clicks still reach the controls behind it.
  arCanvas: {
    position: 'absolute',
    top: 0,
    left: 0,
    width: '100%',
    height: '100%',
    // @ts-ignore — React Native Web passes unknown CSS props through
    pointerEvents: 'none',
  },
  accessoryRow: {
    flexDirection: 'row',
    gap: 12,
  },
  styleRow: {
    flexDirection: 'row',
    gap: 8,
  },
  styleButton: {
    flexDirection: 'row',
    alignItems: 'center',
    gap: 6,
    paddingHorizontal: 14,
    paddingVertical: 8,
    borderRadius: 10,
    backgroundColor: 'rgba(255,255,255,0.08)',
    borderWidth: 1.5,
    borderColor: 'rgba(255,255,255,0.12)',
  },
  styleButtonActive: {
    backgroundColor: 'rgba(160,32,240,0.30)',
    borderColor: '#a020f0',
  },
  styleColorDot: {
    width: 12,
    height: 12,
    borderRadius: 6,
    borderWidth: 1,
    borderColor: 'rgba(255,255,255,0.3)',
  },
  styleLabel: {
    color: '#ffffff',
    fontSize: 12,
    fontWeight: '600',
  },
  accButton: {
    width: 88,
    paddingVertical: 10,
    borderRadius: 14,
    alignItems: 'center',
    gap: 4,
    backgroundColor: 'rgba(255,255,255,0.08)',
    borderWidth: 1.5,
    borderColor: 'rgba(255,255,255,0.12)',
  },
  accButtonActive: {
    backgroundColor: 'rgba(160,32,240,0.35)',
    borderColor: '#a020f0',
  },
  accIcon: {
    fontSize: 22,
  },
  accLabel: {
    color: '#ffffff',
    fontSize: 12,
    fontWeight: '700',
  },
  actions: {
    flexDirection: 'row',
    gap: 12,
  },
  button: {
    backgroundColor: '#7c3aed',
    paddingHorizontal: 20,
    paddingVertical: 12,
    borderRadius: 12,
  },
  buttonSecondary: {
    backgroundColor: 'rgba(255,255,255,0.12)',
  },
  buttonText: {
    color: '#ffffff',
    fontWeight: '700',
    fontSize: 14,
  },
  center: {
    flex: 1,
    justifyContent: 'center',
    alignItems: 'center',
  },
});
