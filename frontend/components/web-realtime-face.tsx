import { useEffect, useRef, useState } from 'react';
import { Platform, Pressable, StyleSheet, Text, View } from 'react-native';

type LandmarkPoint = { x: number; y: number; z?: number };

const DETECT_INTERVAL_MS = 80;
const SMOOTH_ALPHA = 0.72;

const smoothLandmarks = (previous: LandmarkPoint[] | null, next: LandmarkPoint[]) => {
  if (!previous || previous.length !== next.length) {
    return next.map((point) => ({ ...point }));
  }

  return next.map((point, index) => {
    const prev = previous[index];
    if (!prev) {
      return { ...point };
    }

    return {
      x: prev.x * SMOOTH_ALPHA + point.x * (1 - SMOOTH_ALPHA),
      y: prev.y * SMOOTH_ALPHA + point.y * (1 - SMOOTH_ALPHA),
      z: typeof prev.z === 'number' || typeof point.z === 'number'
        ? (prev.z ?? 0) * SMOOTH_ALPHA + (point.z ?? 0) * (1 - SMOOTH_ALPHA)
        : undefined,
    };
  });
};

export default function WebRealtimeFace() {
  const videoRef = useRef<any>(null);
  const canvasRef = useRef<any>(null);
  const streamRef = useRef<any>(null);
  const landmarkerRef = useRef<any>(null);
  const rafRef = useRef<number | null>(null);

  const lastFrameRef = useRef<any>(null);
  const lastPointsRef = useRef<LandmarkPoint[] | null>(null);
  const lastSizeRef = useRef({ width: 800, height: 450 });
  const lastDetectAtRef = useRef(0);
  const lastSeenAtRef = useRef(0);

  const [running, setRunning] = useState(false);
  const [message, setMessage] = useState('Hazır');
  const [showLandmarks, setShowLandmarks] = useState(true);
  const showLandmarksRef = useRef(true);

  useEffect(() => {
    showLandmarksRef.current = showLandmarks;
  }, [showLandmarks]);

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

    if (points) {
      drawLandmarks(ctx, points, width, height);
    }
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
      } else if (now - lastSeenAtRef.current > 400) {
        points = null;
        lastPointsRef.current = null;
      }
    }

    lastFrameRef.current = video;

    if (points) {
      drawLandmarks(ctx, points, width, height);
      setMessage(`✔ ${points.length} landmark`);
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

    const stream = await navigator.mediaDevices.getUserMedia({
      video: true,
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
    };
  }, []);

  if (Platform.OS !== 'web') {
    return (
      <View style={styles.center}>
        <Text>Web demo sadece laptop için</Text>
      </View>
    );
  }

  return (
    <View style={styles.container}>
      <Text style={styles.title}>Realtime Face Tracking</Text>
      <Text>{message}</Text>

      <View style={styles.stage}>
        {/* @ts-ignore */}
        <video ref={videoRef} style={{ display: 'none' }} muted playsInline />
        {/* @ts-ignore */}
        <canvas ref={canvasRef} style={styles.canvas} />
      </View>

      <View style={styles.actions}>
        <Pressable onPress={running ? stop : start} style={styles.button}>
          <Text style={{ color: '#fff' }}>{running ? 'Durdur' : 'Başlat'}</Text>
        </Pressable>

        <Pressable
          onPress={() => {
            const next = !showLandmarks;
            showLandmarksRef.current = next;
            setShowLandmarks(next);

            if (!running) {
              redrawFrozenFrame();
            }
          }}
          style={[
            styles.button,
            { backgroundColor: showLandmarks ? '#0891B2' : '#6B7280' },
          ]}
        >
          <Text style={{ color: '#fff' }}>
            {showLandmarks ? 'Landmark Gizle' : 'Landmark Göster'}
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
    gap: 10,
    backgroundColor: '#F7F4FB',
  },
  stage: {
    width: 800,
    height: 450,
    backgroundColor: '#000',
    borderRadius: 20,
    overflow: 'hidden',
  },
  canvas: {
    width: '100%',
    height: '100%',
  },
  actions: {
    flexDirection: 'row',
    gap: 12,
  },
  button: {
    backgroundColor: '#7c3aed',
    paddingHorizontal: 16,
    paddingVertical: 12,
    borderRadius: 10,
  },
  title: {
    fontSize: 22,
    fontWeight: 'bold',
  },
  center: {
    flex: 1,
    justifyContent: 'center',
    alignItems: 'center',
  },
});