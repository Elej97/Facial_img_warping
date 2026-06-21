import Slider from '@react-native-community/slider';
import * as FileSystem from 'expo-file-system/legacy';
import { Image } from 'expo-image';
import * as ImageManipulator from 'expo-image-manipulator';
import * as ImagePicker from 'expo-image-picker';
import * as Sharing from 'expo-sharing';
import { useCallback, useEffect, useMemo, useRef, useState } from 'react';
import {
    ActivityIndicator,
    Alert,
    Animated,
    Modal,
    PanResponder,
    Platform,
    Pressable,
    ScrollView,
    StyleSheet,
    Switch,
    Text,
    TextInput,
    View,
    useWindowDimensions
} from 'react-native';

import type { EarringStyle, GlassesStyle, HatStyle, MaskStyle, NecklaceStyle, TieStyle } from '@/components/ar-engine';
import { AREngine, EARRING_URLS, GLB_URLS, HAT_URLS, MASK_URLS, NECKLACE_URLS, TIE_URLS } from '@/components/ar-engine';
import LiveWarpCamera, {
    EARRING_VARIANTS,
    GLASSES_VARIANTS,
    HAT_VARIANTS,
    MASK_VARIANTS,
    NECKLACE_VARIANTS,
    TIE_VARIANTS
} from '@/components/live-warp-camera';
import { STUDIO, StudioScreen } from '@/components/studio-shell';
import { ThemedText } from '@/components/themed-text';
import { Colors } from '@/constants/theme';
import { useColorScheme } from '@/hooks/use-color-scheme';
import {
    accessoryAssetUrl,
    agingSamFromBase64,
    applyAccessoryFromBase64,
    applyHairColorFromBase64,
    applyMakeupFromBase64,
    estimateAgeFromBase64,
    estimateAgeFromUri,
    exportEvaluationReportFromBase64,
    landmarksFromBase64,
    preprocessFromUri,
    transferExpressionFromBase64,
    warpProFromBase64,
    type AccessoryStyle,
    type AccessoryType,
    type ProMetrics,
    type ProWarpOperation
} from '@/services/facial-api';
import { Ionicons } from '@expo/vector-icons';
import * as THREE from 'three';
import { GLTFLoader } from 'three/examples/jsm/loaders/GLTFLoader.js';
import { MTLLoader } from 'three/examples/jsm/loaders/MTLLoader.js';
import { OBJLoader } from 'three/examples/jsm/loaders/OBJLoader.js';

const MIN_WIDTH = 512;
const MIN_HEIGHT = 512;
const MIN_CROP_SIZE = 24;
const WEB_NO_SELECT_STYLE: any =
  Platform.OS === 'web'
    ? ({
        touchAction: 'none',
        userSelect: 'none',
        WebkitUserSelect: 'none',
      } as const)
    : null;

type ProcessState = 'idle' | 'selected' | 'error';
type AgeTarget = 'before' | 'after';

type CropBox = {
  x: number;
  y: number;
  width: number;
  height: number;
};

type StageLayout = {
  width: number;
  height: number;
};

type ContainLayout = {
  scale: number;
  renderWidth: number;
  renderHeight: number;
  offsetX: number;
  offsetY: number;
};

type PreprocessMeta = {
  faceBBox: [number, number, number, number];
  processedSize: { width: number; height: number };
};

type ResizeMode = 'left' | 'right' | 'top' | 'bottom' | 'topLeft' | 'topRight' | 'bottomLeft' | 'bottomRight';

type ProOperation = ProWarpOperation | 'aging' | 'deaging';
type MakeupUiTarget = 'lip' | 'cheek' | 'bronzer' | 'lash' | 'brow' | 'eye' | 'teeth';
type MakeupBackendRegion = 'lip' | 'cheek' | 'brow' | 'lash' | 'eye' | 'teeth';
type MakeupPreviewKind = 'hair' | 'makeup' | 'accessory';
type AccessoryUiTarget = AccessoryType;

type MakeupPreset = {
  key: MakeupUiTarget;
  label: string;
  backendRegion: MakeupBackendRegion;
  defaultColor: string;
};

type AccessoryPreset = {
  key: AccessoryUiTarget;
  label: string;
  styles: { key: AccessoryStyle; label: string; thumbnail: string }[];
  defaultStyle: AccessoryStyle;
  defaultColor: string;
};

type MetricStatus = 'good' | 'warn' | 'bad' | 'neutral';

type EvalMetricRow = {
  metric: string;
  value: string;
  purposeRange: string;
  status: MetricStatus;
};

const METRIC_STATUS_COLOR: Record<MetricStatus, string> = {
  good: '#4ADE80',
  warn: '#FBBF24',
  bad: '#F87171',
  neutral: '#CBD5E1',
};

const formatEnergyValue = (value?: number | null) => {
  if (value == null || !Number.isFinite(value)) {
    return 'N/A';
  }

  return value.toExponential(3);
};

const PRO_OPERATIONS: ProOperation[] = [
  'smile_enhancement',
  'brow_lift',
  'lip_plump',
  'slim_face',
  'enlarge_eyes',
  'sharpen_jaw',
  'resize_nose',
  'aging',
  'deaging',
];

const PRO_LABEL: Record<ProOperation, string> = {
  smile_enhancement: 'Smile',
  brow_lift: 'Brow Lift',
  lip_plump: 'Lip Plump',
  slim_face: 'Slim Face',
  enlarge_eyes: 'Enlarge Eyes',
  sharpen_jaw: 'Sharpen Jaw',
  resize_nose: 'Resize Nose',
  aging: 'Aging',
  deaging: 'De-Aging',
};

const LAB_DEFAULT_INTENSITY = 50;
const LAB_INTENSITY_STEP = 5;
const LAB_RBF_SMOOTH = 2.8;

const MANUAL_MAKEUP_PRESETS: MakeupPreset[] = [
  { key: 'lip', label: 'Ruj', backendRegion: 'lip', defaultColor: '#D45A73' },
  { key: 'cheek', label: 'Allık', backendRegion: 'cheek', defaultColor: '#F29AAF' },
  { key: 'bronzer', label: 'Bronzer', backendRegion: 'cheek', defaultColor: '#B97A4C' },
  { key: 'lash', label: 'Far', backendRegion: 'lash', defaultColor: '#8E5CF7' },
  { key: 'brow', label: 'Kaş', backendRegion: 'brow', defaultColor: '#5E4735' },
  { key: 'eye', label: 'Göz Rengi', backendRegion: 'eye', defaultColor: '#8B4513' },
  { key: 'teeth', label: 'Diş Beyazlatma', backendRegion: 'teeth', defaultColor: '#FFFFFF' },
];

const MANUAL_MAKEUP_SWATCHES: Record<MakeupUiTarget, string[]> = {
  lip: ['#D45A73', '#A83253', '#F18FA7', '#BE4369', '#FF7AA2'],
  cheek: ['#F29AAF', '#F2B2A6', '#E88E7A', '#DB6F93', '#F7C1C8'],
  bronzer: ['#B97A4C', '#9F6642', '#D09A6B', '#7F5337', '#C88557'],
  lash: ['#8E5CF7', '#D86AD8', '#4ECDC4', '#5B8DEF', '#C9A227'],
  brow: ['#5E4735', '#463427', '#7A5B43', '#2D221A'],
  eye: ['#8B4513', '#1C3A70', '#2F5233', '#704214', '#1A1A2E'],
  teeth: ['#FFFFFF', '#F5F5F5', '#FFFACD', '#F0E68C', '#FAFAF0'],
};

const MANUAL_MAKEUP_LABELS: Record<MakeupUiTarget, string> = {
  lip: 'Ruj',
  cheek: 'Allık',
  bronzer: 'Bronzer',
  lash: 'Far',
  brow: 'Kaş',
  eye: 'Göz Rengi',
  teeth: 'Diş Beyazlatma',
};

const ACCESSORY_PRESETS: AccessoryPreset[] = [
  {
    key: 'glasses',
    label: 'Gozluk',
    defaultStyle: 'classic',
    defaultColor: '#111827',
    styles: [
      { key: 'classic', label: 'Klasik', thumbnail: 'glasses/user_black_square_clean.png' },
      { key: 'round', label: 'Altin', thumbnail: 'glasses/user_gold_frame.png' },
      { key: 'heart', label: 'Kalp', thumbnail: 'glasses/user_pink_heart.png' },
    ],
  },
  {
    key: 'mustache',
    label: 'Biyik',
    defaultStyle: 'handlebar',
    defaultColor: '#3A2618',
    styles: [
      { key: 'handlebar', label: 'Kivrik Biyik', thumbnail: 'mustache/handlebar_asset.png' },
      { key: 'chevron', label: 'Sakal', thumbnail: 'mustache/full_beard_asset.png' },
    ],
  },
  {
    key: 'hat',
    label: 'Sapka',
    defaultStyle: 'cowboy',
    defaultColor: '#8B4513',
    styles: [
      { key: 'cowboy', label: 'Kovboy', thumbnail: 'hats/CowboyHat.jpg' },
      { key: 'cap', label: 'Kasket', thumbnail: 'hats/cap.jpg' },
      { key: 'asian', label: 'Asya', thumbnail: 'hats/asianHat.jpg' },
      { key: 'newasian', label: 'Asya 2', thumbnail: 'hats/newAsianHat.png' },
      { key: 'pink', label: 'Pembe', thumbnail: 'hats/pinkHat.jpg' },
    ],
  },
];

const normalizeHexColor = (value: string, fallback: string) => {
  const cleaned = value.trim().replace(/^#/, '');
  if (/^[0-9a-fA-F]{3}$/.test(cleaned)) {
    return `#${cleaned
      .split('')
      .map((part) => part + part)
      .join('')}`.toUpperCase();
  }

  if (/^[0-9a-fA-F]{6}$/.test(cleaned)) {
    return `#${cleaned.toUpperCase()}`;
  }

  return fallback;
};

const getContrastTextColor = (hex: string) => {
  const normalized = normalizeHexColor(hex, '#FFFFFF').replace('#', '');
  const red = parseInt(normalized.slice(0, 2), 16);
  const green = parseInt(normalized.slice(2, 4), 16);
  const blue = parseInt(normalized.slice(4, 6), 16);
  const brightness = red * 0.299 + green * 0.587 + blue * 0.114;
  return brightness > 150 ? '#111111' : '#FFFFFF';
};

const clamp = (value: number, min: number, max: number) => Math.max(min, Math.min(max, value));

const getContainLayout = (stage: StageLayout, imageSize: { width: number; height: number }): ContainLayout => {
  const scale = Math.min(stage.width / imageSize.width, stage.height / imageSize.height);
  const renderWidth = imageSize.width * scale;
  const renderHeight = imageSize.height * scale;

  return {
    scale,
    renderWidth,
    renderHeight,
    offsetX: (stage.width - renderWidth) / 2,
    offsetY: (stage.height - renderHeight) / 2,
  };
};

const toWebImageUri = (asset: { uri: string; base64?: string | null; mimeType?: string | null }) => {
  if (Platform.OS !== 'web' || !asset.base64) {
    return asset.uri;
  }

  const mimeType = asset.mimeType || 'image/jpeg';
  return `data:${mimeType};base64,${asset.base64}`;
};

const imageElementToPngDataUrl = (image: HTMLImageElement, width: number, height: number) => {
  const canvas = document.createElement('canvas');
  canvas.width = width;
  canvas.height = height;
  const ctx = canvas.getContext('2d');
  if (!ctx) {
    return null;
  }

  ctx.drawImage(image, 0, 0, width, height);
  return canvas.toDataURL('image/png');
};

const createInitialCropBox = (stage: StageLayout, imageSize: { width: number; height: number }): CropBox => {
  const contain = getContainLayout(stage, imageSize);
  const cropWidth = clamp(contain.renderWidth * 0.5, MIN_CROP_SIZE, contain.renderWidth);
  const cropHeight = clamp(contain.renderHeight * 0.5, MIN_CROP_SIZE, contain.renderHeight);

  return {
    x: contain.offsetX + (contain.renderWidth - cropWidth) / 2,
    y: contain.offsetY + (contain.renderHeight - cropHeight) / 2,
    width: cropWidth,
    height: cropHeight,
  };
};

const clampCropBox = (box: CropBox, stage: StageLayout, imageSize: { width: number; height: number }) => {
  const contain = getContainLayout(stage, imageSize);
  const cropWidth = clamp(box.width, MIN_CROP_SIZE, contain.renderWidth);
  const cropHeight = clamp(box.height, MIN_CROP_SIZE, contain.renderHeight);
  const x = clamp(box.x, contain.offsetX, contain.offsetX + contain.renderWidth - cropWidth);
  const y = clamp(box.y, contain.offsetY, contain.offsetY + contain.renderHeight - cropHeight);

  return { x, y, width: cropWidth, height: cropHeight };
};

let sharedRenderer: THREE.WebGLRenderer | null = null;
let sharedCanvas: HTMLCanvasElement | null = null;

const getSharedRenderer = (): { renderer: THREE.WebGLRenderer; canvas: HTMLCanvasElement } => {
  if (!sharedRenderer) {
    sharedCanvas = document.createElement('canvas');
    sharedCanvas.width = 128;
    sharedCanvas.height = 128;
    sharedRenderer = new THREE.WebGLRenderer({ canvas: sharedCanvas, alpha: true, antialias: true });
    sharedRenderer.setPixelRatio(1);
    sharedRenderer.setSize(128, 128);
    sharedRenderer.setClearColor(0x000000, 0);
  }
  return { renderer: sharedRenderer, canvas: sharedCanvas! };
};

const renderThumbnail = async (url: string, isObj: boolean, mtlUrl?: string): Promise<string> => {
  return new Promise((resolve) => {
    const { renderer, canvas } = getSharedRenderer();

    if (url === 'sombrero') {
      const scene = new THREE.Scene();
      const camera = new THREE.PerspectiveCamera(45, 1, 0.1, 100);
      camera.position.set(0, 0.6, 3.2);

      const ambient = new THREE.AmbientLight(0xffffff, 0.7);
      const key = new THREE.DirectionalLight(0xffffff, 0.9);
      key.position.set(1, 2, 3);
      scene.add(ambient, key);

      const model = new THREE.Group();
      const mkMat = (c: number) => new THREE.MeshStandardMaterial({ color: c, roughness: 0.78 });
      const tan = 0xc4891a;
      const dark = 0x1a0900;
      const brim = new THREE.Mesh(new THREE.CylinderGeometry(1.50, 1.50, 0.07, 64), mkMat(tan));
      brim.position.y = 0.035;
      const crownLow = new THREE.Mesh(new THREE.CylinderGeometry(0.52, 0.80, 0.80, 32), mkMat(tan));
      crownLow.position.y = 0.47;
      const crownTop = new THREE.Mesh(new THREE.SphereGeometry(0.52, 32, 16, 0, Math.PI * 2, 0, Math.PI * 0.55), mkMat(tan));
      crownTop.position.y = 0.87;
      const band = new THREE.Mesh(new THREE.TorusGeometry(0.70, 0.04, 8, 48), mkMat(dark));
      band.rotation.x = Math.PI / 2;
      band.position.y = 0.12;
      model.add(brim, crownLow, crownTop, band);
      scene.add(model);

      renderer.render(scene, camera);
      const dataUrl = canvas.toDataURL('image/png');
      
      scene.remove(model);
      brim.geometry.dispose();
      (brim.material as THREE.Material).dispose();
      crownLow.geometry.dispose();
      (crownLow.material as THREE.Material).dispose();
      crownTop.geometry.dispose();
      (crownTop.material as THREE.Material).dispose();
      band.geometry.dispose();
      (band.material as THREE.Material).dispose();

      resolve(dataUrl);
      return;
    }

    const scene = new THREE.Scene();
    const camera = new THREE.PerspectiveCamera(45, 1, 0.1, 100);
    camera.position.set(0, 0, 8);

    const ambient = new THREE.AmbientLight(0xffffff, 0.75);
    const key = new THREE.DirectionalLight(0xffffff, 0.85);
    key.position.set(1, 2, 3);
    scene.add(ambient, key);

    const loadAndRender = (model: THREE.Object3D) => {
      const box = new THREE.Box3().setFromObject(model);
      const size = box.getSize(new THREE.Vector3());
      const center = box.getCenter(new THREE.Vector3());

      const maxDim = Math.max(size.x, size.y, size.z);
      const s = 4.0 / Math.max(maxDim, 0.001);
      model.scale.setScalar(s);
      model.position.copy(center).multiplyScalar(-s);

      scene.add(model);
      renderer.render(scene, camera);
      const dataUrl = canvas.toDataURL('image/png');
      
      model.traverse((child) => {
        if (child instanceof THREE.Mesh) {
          child.geometry.dispose();
          if (Array.isArray(child.material)) {
            child.material.forEach((m) => m.dispose());
          } else if (child.material) {
            child.material.dispose();
          }
        }
      });

      scene.remove(model);
      resolve(dataUrl);
    };

    if (isObj) {
      const mtlLoader = new MTLLoader();
      const objLoader = new OBJLoader();
      mtlLoader.load(mtlUrl!, (materials) => {
        materials.preload();
        objLoader.setMaterials(materials);
        objLoader.load(url, (obj) => {
          loadAndRender(obj);
        }, undefined, () => resolve(''));
      }, undefined, () => resolve(''));
    } else {
      const gltfLoader = new GLTFLoader();
      gltfLoader.load(url, (gltf) => {
        loadAndRender(gltf.scene);
      }, undefined, () => resolve(''));
    }
  });
};

export default function CreateScreen() {
  const colorScheme = useColorScheme() ?? 'light';
  const colors = Colors[colorScheme];
  // In dark mode tint is #fff — text on tint buttons must be dark to be visible
  const tintTextColor = colorScheme === 'dark' ? '#11181C' : '#FFFFFF';
  const { width, height } = useWindowDimensions();
  const isWide = width >= 960;
  const cropStageHeight = Math.min(560, Math.max(360, height - 220));
  const [mode, setMode] = useState<'photo' | 'live'>('photo');
  const [selectedImageName, setSelectedImageName] = useState<string | null>(null);
  const [selectedImageUri, setSelectedImageUri] = useState<string | null>(null);
  const [selectedImageSize, setSelectedImageSize] = useState<{ width: number; height: number } | null>(null);
  const [selectedImageB64, setSelectedImageB64] = useState<string | null>(null);
  const [statusMessage, setStatusMessage] = useState<string>('Henüz görsel seçilmedi.');
  const [processState, setProcessState] = useState<ProcessState>('idle');
  const [cropApplied, setCropApplied] = useState(false);
  const [cropEditorVisible, setCropEditorVisible] = useState(false);
  const [cropStageLayout, setCropStageLayout] = useState<StageLayout | null>(null);
  const [cropBox, setCropBox] = useState<CropBox | null>(null);
  const cropBoxRef = useRef<CropBox | null>(null);
  const dragStartRef = useRef<CropBox | null>(null);
  const resizeStartRef = useRef<CropBox | null>(null);

  // CV pipeline state
  const [preprocessLoading, setPreprocessLoading] = useState(false);
  const [preprocessError, setPreprocessError] = useState<string | null>(null);
  const [preprocessedB64, setPreprocessedB64] = useState<string | null>(null);
  const [preprocessMeta, setPreprocessMeta] = useState<PreprocessMeta | null>(null);

  const [landmarkLoading, setLandmarkLoading] = useState(false);
  const [landmarkError, setLandmarkError] = useState<string | null>(null);
  const [landmarkCount, setLandmarkCount] = useState<number | null>(null);
  const [landmarkPoints, setLandmarkPoints] = useState<number[][] | null>(null);
  const [showLandmarks, setShowLandmarks] = useState(false);
  const preprocessRunRef = useRef(0);
  const landmarkRunRef = useRef(0);
  const [landmarkPreviewLayout, setLandmarkPreviewLayout] = useState<StageLayout | null>(null);


  const [referenceExpressionName, setReferenceExpressionName] = useState<string | null>(null);
  const [referenceExpressionUri, setReferenceExpressionUri] = useState<string | null>(null);
  const [referenceExpressionSize, setReferenceExpressionSize] = useState<{ width: number; height: number } | null>(null);
  const [expressionTransferIntensity, setExpressionTransferIntensity] = useState(0.75);
  const [expressionTransferLoading, setExpressionTransferLoading] = useState(false);
  const [expressionTransferError, setExpressionTransferError] = useState<string | null>(null);
  const [expressionTransferResultB64, setExpressionTransferResultB64] = useState<string | null>(null);
  const [manualLipWarpResultB64, setManualLipWarpResultB64] = useState<string | null>(null);
  const [manualLipWarpError, setManualLipWarpError] = useState<string | null>(null);

  const [landmarkBackend, setLandmarkBackend] = useState<'mediapipe' | 'dlib' | 'hybrid'>('hybrid');
  const [activeProOperations, setActiveProOperations] = useState<ProOperation[]>([]);
  const [proOperationIntensity, setProOperationIntensity] = useState<Record<ProOperation, number>>({
    smile_enhancement: LAB_DEFAULT_INTENSITY,
    brow_lift: LAB_DEFAULT_INTENSITY,
    lip_plump: LAB_DEFAULT_INTENSITY,
    slim_face: LAB_DEFAULT_INTENSITY,
    resize_nose: LAB_DEFAULT_INTENSITY,
    aging: 80,
    deaging: 80,
  });
  const [hoveredProOperation, setHoveredProOperation] = useState<ProOperation | null>(null);
  const [proLoading, setProLoading] = useState(false);
  const [proError, setProError] = useState<string | null>(null);
  const [proResultB64, setProResultB64] = useState<string | null>(null);
  const [proMetrics, setProMetrics] = useState<ProMetrics | null>(null);
  
  // Pro operations layers system
  type ProLayer = {
    id: string;
    operation: ProOperation;
    intensity: number;
    resultB64: string;
    locked: boolean;
  };
  
  const [proLayers, setProLayers] = useState<ProLayer[]>([]);
  const [makeupTarget, setMakeupTarget] = useState<MakeupUiTarget>('lip');
  const [makeupHexColor, setMakeupHexColor] = useState(MANUAL_MAKEUP_PRESETS[0].defaultColor);
  const [makeupIntensity, setMakeupIntensity] = useState(0.48);
  const [makeupLoading, setMakeupLoading] = useState(false);
  const [makeupError, setMakeupError] = useState<string | null>(null);
  const [makeupResultB64, setMakeupResultB64] = useState<string | null>(null);
  const [accessoryEnabled, setAccessoryEnabled] = useState(false);
  const [accessoryTarget, setAccessoryTarget] = useState<AccessoryUiTarget>('glasses');
  const [accessoryStyle, setAccessoryStyle] = useState<AccessoryStyle>('classic');
  const [accessoryIntensity, setAccessoryIntensity] = useState(0.72);
  const [accessoryScale, setAccessoryScale] = useState(1);
  const [accessoryOffsetX, setAccessoryOffsetX] = useState(0);
  const [accessoryOffsetY, setAccessoryOffsetY] = useState(0);
  const [accessoryLoading, setAccessoryLoading] = useState(false);
  const [accessoryError, setAccessoryError] = useState<string | null>(null);
  const [accessoryResultB64, setAccessoryResultB64] = useState<string | null>(null);
  
  // Makeup layers system - keep multiple makeup layers
  type MakeupLayer = {
    id: string;
    region: MakeupBackendRegion;
    color: string;
    intensity: number;
    resultB64: string;
    locked: boolean;
  };
  
  const [makeupLayers, setMakeupLayers] = useState<MakeupLayer[]>([]);

  // --- Hair Color state ---
  const [hairColorHex, setHairColorHex] = useState('#3b1f0f');
  const [hairColorIntensity, setHairColorIntensity] = useState(0.85);
  const [hairColorLoading, setHairColorLoading] = useState(false);
  const [hairColorError, setHairColorError] = useState<string | null>(null);
  const [hairColorResultB64, setHairColorResultB64] = useState<string | null>(null);
  const [makeupPreviewKind, setMakeupPreviewKind] = useState<MakeupPreviewKind | null>(null);

  const [evalMetrics, setEvalMetrics] = useState<ProMetrics | null>(null);
  const [evalSourceLabel, setEvalSourceLabel] = useState<string | null>(null);
  const [evalResultB64, setEvalResultB64] = useState<string | null>(null);
  const [spectrumGrayB64, setSpectrumGrayB64] = useState<string | null>(null);
  const [spectrumBlueB64, setSpectrumBlueB64] = useState<string | null>(null);
  const [spectrumRedB64, setSpectrumRedB64] = useState<string | null>(null);
  const [ageBefore, setAgeBefore] = useState<number | null>(null);
  const [ageAfter, setAgeAfter] = useState<number | null>(null);
  const [ageLoading, setAgeLoading] = useState(false);
  const [ageError, setAgeError] = useState<string | null>(null);
  const proDebounceRef = useRef<ReturnType<typeof setTimeout> | null>(null);
  const accessoryDebounceRef = useRef<ReturnType<typeof setTimeout> | null>(null);
  const ageRequestRef = useRef(0);
  const [proCompareHeld, setProCompareHeld] = useState(false);
  const proCompareOpacity = useRef(new Animated.Value(0)).current;
  const [lightboxUri, setLightboxUri] = useState<string | null>(null);
  const [lightboxCompareUri, setLightboxCompareUri] = useState<string | null>(null);

  type TabKey = 'analysis' | 'expression' | 'prolab' | 'makeup' | 'accessory' | 'accessory2';
  const [activeTab, setActiveTab] = useState<TabKey>('analysis');
  const [landmarkPoints3d, setLandmarkPoints3d] = useState<{ x: number; y: number; z: number }[] | null>(null);
  const [accessory2ResultB64, setAccessory2ResultB64] = useState<string | null>(null);
  const [activeCategory2, setActiveCategory2] = useState<'glasses' | 'hat' | 'earrings' | 'necklace' | 'tie' | 'mask'>('hat');
  const [accessories2, setAccessories2] = useState({
    glasses: false,
    hat: false,
    earrings: false,
    necklace: false,
    tie: false,
    mask: false,
  });
  const [glassesStyle2, setGlassesStyle2] = useState<GlassesStyle>('ski');
  const [hatStyle2, setHatStyle2] = useState<HatStyle>('top-hat');
  const [earringStyle2, setEarringStyle2] = useState<EarringStyle>('hoop-earrings');
  const [necklaceStyle2, setNecklaceStyle2] = useState<NecklaceStyle>('necklace');
  const [tieStyle2, setTieStyle2] = useState<TieStyle>('necktie');
  const [maskStyle2, setMaskStyle2] = useState<MaskStyle>('anon-mask');
  const arEngineRef2 = useRef<AREngine | null>(null);
  const canvasRef2 = useRef<HTMLCanvasElement | null>(null);
  const [lightboxZoom, setLightboxZoom] = useState(1);
  const [lightboxOffset, setLightboxOffset] = useState({ x: 0, y: 0 });
  const lightboxPanStartRef = useRef({ x: 0, y: 0 });

  const [thumbnails2, setThumbnails2] = useState<Record<string, string>>({});
  useEffect(() => {
    if (activeTab !== 'accessory2' || !activeCategory2) return;
    let active = true;

    const generate = async () => {
      let variants: { key: string; url?: string; isObj?: boolean; mtlUrl?: string }[] = [];
      if (activeCategory2 === 'glasses') {
        variants = GLASSES_VARIANTS.map((v) => ({ key: v.key, url: GLB_URLS[v.key] }));
      } else if (activeCategory2 === 'hat') {
        variants = HAT_VARIANTS.map((v) => ({ key: v.key, url: v.key === 'sombrero' ? 'sombrero' : HAT_URLS[v.key] }));
      } else if (activeCategory2 === 'mask') {
        variants = MASK_VARIANTS.map((v) => ({ key: v.key, url: MASK_URLS[v.key] }));
      } else if (activeCategory2 === 'earrings') {
        variants = EARRING_VARIANTS.map((v) => ({ key: v.key, url: EARRING_URLS[v.key] }));
      } else if (activeCategory2 === 'necklace') {
        variants = NECKLACE_VARIANTS.map((v) => ({ key: v.key, url: NECKLACE_URLS[v.key] }));
      } else if (activeCategory2 === 'tie') {
        variants = TIE_VARIANTS.map((v) => {
          const config = TIE_URLS[v.key];
          return {
            key: v.key,
            url: config.glb || config.obj,
            isObj: !!config.obj,
            mtlUrl: config.mtl,
          };
        });
      }

      for (const v of variants) {
        if (!active) break;
        const cacheKey = `${activeCategory2}_${v.key}`;
        if (thumbnails2[cacheKey]) continue;

        if (v.url) {
          try {
            const dataUrl = await renderThumbnail(v.url, !!v.isObj, v.mtlUrl);
            if (dataUrl && active) {
              setThumbnails2((prev) => ({ ...prev, [cacheKey]: dataUrl }));
            }
          } catch (e) {
            console.error('Failed to render thumbnail for', cacheKey, e);
          }
        }
      }
    };

    void generate();

    return () => {
      active = false;
    };
  }, [activeCategory2, activeTab]);

  const metricTableRows = useMemo<EvalMetricRow[]>(() => {
    if (!evalMetrics) {
      return [];
    }

    const mse = evalMetrics.mse;
    const psnr = evalMetrics.psnr;
    const ssim = evalMetrics.ssim;

    const mseStatus: MetricStatus = !Number.isFinite(mse) ? 'neutral'
      : mse < 0.01 ? 'good' : mse < 0.05 ? 'warn' : 'bad';
    const psnrStatus: MetricStatus = !Number.isFinite(psnr) ? 'good'
      : psnr > 30 ? 'good' : psnr > 25 ? 'warn' : 'bad';
    const ssimStatus: MetricStatus = !Number.isFinite(ssim) ? 'neutral'
      : ssim >= 0.8 ? 'good' : ssim >= 0.6 ? 'warn' : 'bad';

    const rows: EvalMetricRow[] = [
      {
        metric: 'MSE',
        value: Number.isFinite(mse) ? mse.toFixed(6) : 'N/A',
        purposeRange: 'Pixel diff. Lower is better, ideal 0.',
        status: mseStatus,
      },
      {
        metric: 'PSNR',
        value: Number.isFinite(psnr) ? `${psnr.toFixed(4)} dB` : 'Infinity',
        purposeRange: 'Signal quality. Higher is better, > 30 dB.',
        status: psnrStatus,
      },
      {
        metric: 'SSIM',
        value: Number.isFinite(ssim) ? ssim.toFixed(6) : 'N/A',
        purposeRange: 'Perceptual similarity. Closer to 1, >= 0.80.',
        status: ssimStatus,
      },
    ];

    if (evalMetrics.total_spectral_energy_before != null || evalMetrics.total_spectral_energy_after != null) {
      rows.push(
        {
          metric: 'Total Energy',
          value: `${formatEnergyValue(evalMetrics.total_spectral_energy_before)} → ${formatEnergyValue(evalMetrics.total_spectral_energy_after)}`,
          purposeRange: `Δ ${formatEnergyValue(evalMetrics.total_spectral_energy_delta)}. Freq-domain power.`,
          status: 'neutral',
        },
        {
          metric: 'LF Energy',
          value: `${formatEnergyValue(evalMetrics.low_frequency_energy_before)} → ${formatEnergyValue(evalMetrics.low_frequency_energy_after)}`,
          purposeRange: 'Low-frequency component energy.',
          status: 'neutral',
        },
        {
          metric: 'HF Energy',
          value: `${formatEnergyValue(evalMetrics.high_frequency_energy_before)} → ${formatEnergyValue(evalMetrics.high_frequency_energy_after)}`,
          purposeRange: 'High-frequency component energy.',
          status: 'neutral',
        },
      );
    }

    return rows;
  }, [evalMetrics]);

  const landmarkOverlayPoints = useMemo(() => {
    if (!landmarkPoints || !preprocessMeta || !selectedImageSize || !landmarkPreviewLayout) {
      return [];
    }

    const contain = getContainLayout(landmarkPreviewLayout, selectedImageSize);
    const [bboxX, bboxY, bboxW, bboxH] = preprocessMeta.faceBBox;
    const scaleX = bboxW / preprocessMeta.processedSize.width;
    const scaleY = bboxH / preprocessMeta.processedSize.height;

    return landmarkPoints.map(([x, y]) => ({
      x: contain.offsetX + (bboxX + x * scaleX) * contain.scale,
      y: contain.offsetY + (bboxY + y * scaleY) * contain.scale,
    }));
  }, [landmarkPoints, landmarkPreviewLayout, preprocessMeta, selectedImageSize]);

  const downloadBase64File = async (fileB64: string, fileName: string, mimeType: string) => {
    if (Platform.OS === 'web') {
      const binary = globalThis.atob(fileB64);
      const bytes = new Uint8Array(binary.length);
      for (let i = 0; i < binary.length; i++) {
        bytes[i] = binary.charCodeAt(i);
      }
      const blob = new Blob([bytes], { type: mimeType });
      const url = URL.createObjectURL(blob);
      const link = document.createElement('a');
      link.href = url;
      link.download = fileName;
      document.body.appendChild(link);
      link.click();
      document.body.removeChild(link);
      URL.revokeObjectURL(url);
      return;
    }

    const dir = FileSystem.cacheDirectory ?? FileSystem.documentDirectory;
    if (!dir) {
      throw new Error('Dosya dizini bulunamadi.');
    }

    const uri = `${dir}${fileName}`;
    await FileSystem.writeAsStringAsync(uri, fileB64, { encoding: FileSystem.EncodingType.Base64 });
    if (await Sharing.isAvailableAsync()) {
      await Sharing.shareAsync(uri, { mimeType });
    }
  };

  const updateCropBox = useCallback((nextBox: CropBox | null) => {
    cropBoxRef.current = nextBox;
    setCropBox(nextBox);
  }, []);

  const resetExpressionTransferState = () => {
    setReferenceExpressionName(null);
    setReferenceExpressionUri(null);
    setReferenceExpressionSize(null);
    setExpressionTransferLoading(false);
    setExpressionTransferError(null);
    setExpressionTransferResultB64(null);
    setManualLipWarpResultB64(null);
    setManualLipWarpError(null);
  };

  const resetAgeAnalysis = () => {
    ageRequestRef.current += 1;
    setAgeBefore(null);
    setAgeAfter(null);
    setAgeLoading(false);
    setAgeError(null);
  };

  const runAgeAnalysis = useCallback(async (source: string, target: AgeTarget, kind: 'uri' | 'base64') => {
    const requestId = ++ageRequestRef.current;
    setAgeLoading(true);
    setAgeError(null);

    try {
      const data = kind === 'uri' ? await estimateAgeFromUri(source) : await estimateAgeFromBase64(source);
      if (!data.success) {
        throw new Error(data.message ?? 'Age estimation failed');
      }

      const estimatedAge = Number(data.estimated_age ?? data.age);
      if (!Number.isFinite(estimatedAge)) {
        throw new Error('Age estimation returned an invalid value.');
      }

      if (requestId !== ageRequestRef.current) {
        return;
      }

      if (target === 'before') {
        setAgeBefore(Math.round(estimatedAge));
      } else {
        setAgeAfter(Math.round(estimatedAge));
      }
    } catch (error) {
      if (requestId !== ageRequestRef.current) {
        return;
      }

      const message = error instanceof Error ? error.message : 'Age estimation failed.';
      setAgeError(message);
      Alert.alert('AI Analizinde hata oluştu', message);
    } finally {
      if (requestId === ageRequestRef.current) {
        setAgeLoading(false);
      }
    }
  }, []);

  const closeCropEditor = () => {
    setCropEditorVisible(false);
    setCropStageLayout(null);
    updateCropBox(null);
  };

  const ensureCropBox = useCallback((stage: StageLayout) => {
    if (!selectedImageSize) {
      return;
    }

    updateCropBox(createInitialCropBox(stage, selectedImageSize));
  }, [selectedImageSize, updateCropBox]);

  useEffect(() => {
    cropBoxRef.current = cropBox;
  }, [cropBox]);

  useEffect(() => {
    if (cropEditorVisible && cropStageLayout && selectedImageSize && !cropBoxRef.current) {
      ensureCropBox(cropStageLayout);
    }
  }, [cropEditorVisible, cropStageLayout, selectedImageSize, ensureCropBox]);

  useEffect(() => {
    Animated.timing(proCompareOpacity, {
      toValue: proCompareHeld ? 1 : 0,
      duration: 200,
      useNativeDriver: true,
    }).start();
  }, [proCompareHeld, proCompareOpacity]);

  useEffect(() => {
    if (!lightboxUri) {
      setLightboxZoom(1);
      setLightboxOffset({ x: 0, y: 0 });
    }
  }, [lightboxUri]);

  const openCropEditor = () => {
    if (!selectedImageUri || !selectedImageSize) {
      setProcessState('error');
      setStatusMessage('Kırpma için önce bir görsel seçmelisin.');
      return;
    }

    setCropEditorVisible(true);
    setStatusMessage('Kırpma alanını sürükle ya da köşeden büyüt/küçült.');
  };

  const setCropToAspect = (ratio: number | 'full') => {
    if (!cropStageLayout || !selectedImageSize) {
      return;
    }

    const contain = getContainLayout(cropStageLayout, selectedImageSize);
    if (ratio === 'full') {
      updateCropBox({
        x: contain.offsetX,
        y: contain.offsetY,
        width: contain.renderWidth,
        height: contain.renderHeight,
      });
      return;
    }

    const maxWidth = contain.renderWidth * 0.82;
    const maxHeight = contain.renderHeight * 0.82;
    let width = maxWidth;
    let height = width / ratio;

    if (height > maxHeight) {
      height = maxHeight;
      width = height * ratio;
    }

    updateCropBox({
      x: contain.offsetX + (contain.renderWidth - width) / 2,
      y: contain.offsetY + (contain.renderHeight - height) / 2,
      width,
      height,
    });
  };

  const centerCropBox = () => {
    if (!cropBox || !cropStageLayout || !selectedImageSize) {
      return;
    }

    const contain = getContainLayout(cropStageLayout, selectedImageSize);
    updateCropBox(
      clampCropBox(
        {
          ...cropBox,
          x: contain.offsetX + (contain.renderWidth - cropBox.width) / 2,
          y: contain.offsetY + (contain.renderHeight - cropBox.height) / 2,
        },
        cropStageLayout,
        selectedImageSize,
      ),
    );
  };

  const resetCropSelection = () => {
    if (cropStageLayout) {
      ensureCropBox(cropStageLayout);
    }
  };

  const downloadTextFile = async (content: string, fileName: string, mimeType: string) => {
    if (Platform.OS === 'web') {
      const blob = new Blob([content], { type: mimeType });
      const url = URL.createObjectURL(blob);
      const link = document.createElement('a');
      link.href = url;
      link.download = fileName;
      document.body.appendChild(link);
      link.click();
      document.body.removeChild(link);
      URL.revokeObjectURL(url);
      return;
    }

    const dir = FileSystem.cacheDirectory ?? FileSystem.documentDirectory;
    if (!dir) {
      throw new Error('Dosya dizini bulunamadi.');
    }

    const uri = `${dir}${fileName}`;
    await FileSystem.writeAsStringAsync(uri, content, { encoding: FileSystem.EncodingType.UTF8 });
    if (await Sharing.isAvailableAsync()) {
      await Sharing.shareAsync(uri, { mimeType });
    }
  };

  const moveCropResponder = useMemo(
    () =>
      PanResponder.create({
        onStartShouldSetPanResponder: () => true,
        onMoveShouldSetPanResponder: (_, gestureState) => Math.abs(gestureState.dx) > 2 || Math.abs(gestureState.dy) > 2,
        onPanResponderGrant: () => {
          dragStartRef.current = cropBoxRef.current;
        },
        onPanResponderMove: (_, gestureState) => {
          if (!dragStartRef.current || !cropStageLayout || !selectedImageSize) {
            return;
          }

          const nextBox = clampCropBox(
            {
              x: dragStartRef.current.x + gestureState.dx,
              y: dragStartRef.current.y + gestureState.dy,
              width: dragStartRef.current.width,
              height: dragStartRef.current.height,
            },
            cropStageLayout,
            selectedImageSize
          );

          updateCropBox(nextBox);
        },
        onPanResponderRelease: () => {
          dragStartRef.current = null;
        },
        onPanResponderTerminate: () => {
          dragStartRef.current = null;
        },
        onPanResponderTerminationRequest: () => false,
        onShouldBlockNativeResponder: () => true,
      }),
    [cropStageLayout, selectedImageSize, updateCropBox]
  );

  const createResizeResponder = useMemo(
    () => (mode: ResizeMode) =>
      PanResponder.create({
        onStartShouldSetPanResponder: () => true,
        onMoveShouldSetPanResponder: () => true,
        onPanResponderGrant: () => {
          resizeStartRef.current = cropBoxRef.current;
        },
        onPanResponderMove: (_, gestureState) => {
          if (!resizeStartRef.current || !cropStageLayout || !selectedImageSize) {
            return;
          }

          const contain = getContainLayout(cropStageLayout, selectedImageSize);
          const startBox = resizeStartRef.current;
          const leftLimit = contain.offsetX;
          const topLimit = contain.offsetY;
          const rightLimit = contain.offsetX + contain.renderWidth;
          const bottomLimit = contain.offsetY + contain.renderHeight;

          let nextX = startBox.x;
          let nextY = startBox.y;
          let nextWidth = startBox.width;
          let nextHeight = startBox.height;

          if (mode === 'left' || mode === 'topLeft' || mode === 'bottomLeft') {
            nextX = clamp(startBox.x + gestureState.dx, leftLimit, startBox.x + startBox.width - MIN_CROP_SIZE);
            nextWidth = startBox.x + startBox.width - nextX;
          }

          if (mode === 'right' || mode === 'topRight' || mode === 'bottomRight') {
            nextWidth = clamp(startBox.width + gestureState.dx, MIN_CROP_SIZE, rightLimit - startBox.x);
          }

          if (mode === 'top' || mode === 'topLeft' || mode === 'topRight') {
            nextY = clamp(startBox.y + gestureState.dy, topLimit, startBox.y + startBox.height - MIN_CROP_SIZE);
            nextHeight = startBox.y + startBox.height - nextY;
          }

          if (mode === 'bottom' || mode === 'bottomLeft' || mode === 'bottomRight') {
            nextHeight = clamp(startBox.height + gestureState.dy, MIN_CROP_SIZE, bottomLimit - startBox.y);
          }

          updateCropBox(
            clampCropBox(
              {
                x: nextX,
                y: nextY,
                width: nextWidth,
                height: nextHeight,
              },
              cropStageLayout,
              selectedImageSize
            )
          );
        },
        onPanResponderRelease: () => {
          resizeStartRef.current = null;
        },
        onPanResponderTerminate: () => {
          resizeStartRef.current = null;
        },
        onPanResponderTerminationRequest: () => false,
        onShouldBlockNativeResponder: () => true,
      }),
    [cropStageLayout, selectedImageSize, updateCropBox]
  );

  const resizeLeftResponder = useMemo(() => createResizeResponder('left'), [createResizeResponder]);
  const resizeRightResponder = useMemo(() => createResizeResponder('right'), [createResizeResponder]);
  const resizeTopResponder = useMemo(() => createResizeResponder('top'), [createResizeResponder]);
  const resizeBottomResponder = useMemo(() => createResizeResponder('bottom'), [createResizeResponder]);
  const resizeTopLeftResponder = useMemo(() => createResizeResponder('topLeft'), [createResizeResponder]);
  const resizeTopRightResponder = useMemo(() => createResizeResponder('topRight'), [createResizeResponder]);
  const resizeBottomLeftResponder = useMemo(() => createResizeResponder('bottomLeft'), [createResizeResponder]);
  const resizeBottomRightResponder = useMemo(() => createResizeResponder('bottomRight'), [createResizeResponder]);

  const lightboxPanResponder = useMemo(
    () =>
      PanResponder.create({
        onStartShouldSetPanResponder: () => lightboxZoom > 1,
        onMoveShouldSetPanResponder: (_, gestureState) =>
          lightboxZoom > 1 && (Math.abs(gestureState.dx) > 2 || Math.abs(gestureState.dy) > 2),
        onPanResponderGrant: () => {
          lightboxPanStartRef.current = lightboxOffset;
        },
        onPanResponderMove: (_, gestureState) => {
          if (lightboxZoom <= 1) {
            return;
          }
          const maxOffset = 220 * lightboxZoom;
          setLightboxOffset({
            x: clamp(lightboxPanStartRef.current.x + gestureState.dx, -maxOffset, maxOffset),
            y: clamp(lightboxPanStartRef.current.y + gestureState.dy, -maxOffset, maxOffset),
          });
        },
        onPanResponderTerminationRequest: () => false,
        onShouldBlockNativeResponder: () => true,
      }),
    [lightboxOffset, lightboxZoom],
  );

  const setLightboxZoomLevel = (nextZoom: number) => {
    const zoom = clamp(nextZoom, 1, 4);
    setLightboxZoom(zoom);
    if (zoom === 1) {
      setLightboxOffset({ x: 0, y: 0 });
    }
  };

  const handleLiveCapture = (dataUrl: string, w: number, h: number) => {
    const stamp = new Date().toISOString().replace(/[:.]/g, '-');
    const name = `canli-yakalama-${stamp}.png`;
    setSelectedImageName(name);
    setSelectedImageUri(dataUrl);
    setSelectedImageSize({ width: w, height: h });
    setSelectedImageB64(dataUrl);
    setProcessState('selected');
    setStatusMessage(`Canlı yakalama hazır: ${w}x${h}.`);
    setCropApplied(false);
    resetExpressionTransferState();
    resetAgeAnalysis();
    closeCropEditor();
    setMode('photo');
    void runAgeAnalysis(dataUrl, 'before', 'uri');
  };

  const applySelectedImage = (name: string, uri: string, size: { width: number; height: number }, imageB64: string | null = null) => {
    setSelectedImageName(name);
    setSelectedImageUri(uri);
    setSelectedImageSize(size);
    setSelectedImageB64(imageB64);
    setProcessState('selected');
    setStatusMessage(`Seçilen görsel hazır: ${size.width}x${size.height}.`);
    setCropApplied(false);
    resetExpressionTransferState();
    resetAgeAnalysis();
    setMakeupResultB64(null);
    setHairColorResultB64(null);
    setAccessoryResultB64(null);
    setAccessoryEnabled(false);
    setMakeupPreviewKind(null);
    setMakeupError(null);
    closeCropEditor();
    void runAgeAnalysis(uri, 'before', 'uri');
  };

  const readWebImageFile = (file: File) => {
    const objectUrl = URL.createObjectURL(file);
    const reader = new FileReader();
    const image = document.createElement('img');
    let imageSize: { width: number; height: number } | null = null;

    image.onload = () => {
      const width = image.naturalWidth || image.width;
      const height = image.naturalHeight || image.height;

      if (width < MIN_WIDTH || height < MIN_HEIGHT) {
        URL.revokeObjectURL(objectUrl);
        setSelectedImageName(null);
        setSelectedImageUri(null);
        setSelectedImageSize(null);
        setSelectedImageB64(null);
        setProcessState('error');
        setStatusMessage(`Minimum çözünürlük ${MIN_WIDTH}x${MIN_HEIGHT} olmalı. Seçilen: ${width}x${height}.`);
        return;
      }

      imageSize = { width, height };
      const pngDataUrl = imageElementToPngDataUrl(image, width, height);
      URL.revokeObjectURL(objectUrl);

      if (!pngDataUrl) {
        setProcessState('error');
        setStatusMessage('Gorsel hazirlanamadi. Lutfen baska bir fotograf secin.');
        return;
      }

      applySelectedImage(file.name || 'secilen_gorsel', pngDataUrl, imageSize, pngDataUrl);
    };

    image.onerror = () => {
      URL.revokeObjectURL(objectUrl);
      setProcessState('error');
      setStatusMessage('Görsel yüklenemedi. Lütfen başka bir fotoğraf seçin.');
    };

    reader.onload = () => {
      if (typeof reader.result === 'string') {
        image.src = reader.result;
      } else {
        URL.revokeObjectURL(objectUrl);
        setProcessState('error');
        setStatusMessage('Dosya okunamadi. Lutfen baska bir fotograf secin.');
      }
    };

    reader.onerror = () => {
      URL.revokeObjectURL(objectUrl);
      setProcessState('error');
      setStatusMessage('Dosya okunamadı. Lütfen başka bir fotoğraf seçin.');
    };

    reader.readAsDataURL(file);
  };

  const pickImage = async () => {
    if (Platform.OS === 'web') {
      const input = document.createElement('input');
      input.type = 'file';
      input.accept = 'image/*';
      input.style.position = 'fixed';
      input.style.left = '-10000px';
      input.style.top = '0';

      input.onchange = () => {
        const file = input.files?.[0];
        document.body.removeChild(input);
        if (file) {
          readWebImageFile(file);
        }
      };

      input.oncancel = () => {
        document.body.removeChild(input);
      };

      document.body.appendChild(input);
      input.click();
      return;
    }

    try {
      const result = await ImagePicker.launchImageLibraryAsync({
        mediaTypes: ['images'],
        quality: 1,
        allowsEditing: false,
        selectionLimit: 1,
        base64: true,
      });

      if (result.canceled || result.assets.length === 0) {
        return;
      }

      const asset = result.assets[0];
      const rawName = asset.fileName ?? asset.uri.split('/').pop() ?? 'secilen_gorsel';
      const mimeType = (asset.mimeType ?? '').toLowerCase();
      const isImageMime = mimeType === '' || mimeType.startsWith('image/');

      if (!isImageMime) {
        setSelectedImageName(null);
        setSelectedImageUri(null);
        setSelectedImageSize(null);
        setSelectedImageB64(null);
        setProcessState('error');
        setStatusMessage('Lütfen geçerli bir görsel dosyası seçin.');
        return;
      }

      if ((asset.width ?? 0) < MIN_WIDTH || (asset.height ?? 0) < MIN_HEIGHT) {
        setSelectedImageName(null);
        setSelectedImageUri(null);
        setSelectedImageSize(null);
        setSelectedImageB64(null);
        setProcessState('error');
        setStatusMessage(`Minimum çözünürlük ${MIN_WIDTH}x${MIN_HEIGHT} olmalı. Seçilen: ${asset.width}x${asset.height}.`);
        return;
      }

      const imageB64 = asset.base64 ? `data:${asset.mimeType || 'image/jpeg'};base64,${asset.base64}` : null;
      applySelectedImage(rawName, asset.uri, { width: asset.width ?? MIN_WIDTH, height: asset.height ?? MIN_HEIGHT }, imageB64);
    } catch (error) {
      console.error('Fotoğraf seçme hatası:', error);
      setProcessState('error');
      setStatusMessage('Fotoğraf seçme hatası. Lütfen tekrar deneyin.');
    }
  };

  const pickReferenceExpressionImage = async () => {
    const result = await ImagePicker.launchImageLibraryAsync({
      mediaTypes: ['images'],
      quality: 0.4,
      allowsEditing: false,
      selectionLimit: 1,
      base64: Platform.OS === 'web',
    });

    if (result.canceled || result.assets.length === 0) {
      return;
    }

    const asset = result.assets[0];
    const rawName = asset.fileName ?? asset.uri.split('/').pop() ?? 'reference_expression';
    const mimeType = (asset.mimeType ?? '').toLowerCase();
    const isImageMime = mimeType === '' || mimeType.startsWith('image/');

    if (!isImageMime) {
      setReferenceExpressionName(null);
      setReferenceExpressionUri(null);
      setReferenceExpressionSize(null);
      setExpressionTransferError('Lütfen geçerli bir referans görseli seçin.');
      return;
    }

    if ((asset.width ?? 0) < MIN_WIDTH || (asset.height ?? 0) < MIN_HEIGHT) {
      setReferenceExpressionName(null);
      setReferenceExpressionUri(null);
      setReferenceExpressionSize(null);
      setExpressionTransferError(`Referans görsel en az ${MIN_WIDTH}x${MIN_HEIGHT} olmalı. Seçilen: ${asset.width}x${asset.height}.`);
      return;
    }

    setReferenceExpressionName(rawName);
    setReferenceExpressionUri(toWebImageUri(asset));
    setReferenceExpressionSize({ width: asset.width ?? MIN_WIDTH, height: asset.height ?? MIN_HEIGHT });
    setExpressionTransferError(null);
    setExpressionTransferResultB64(null);
    setManualLipWarpResultB64(null);
    setManualLipWarpError(null);
    setStatusMessage('Referans ifade görseli hazır. Transfer işlemini başlatabilirsin.');
  };

  const cropImage = async () => {
    if (!selectedImageUri) {
      setProcessState('error');
      setStatusMessage('Kırpma için önce bir görsel seçmelisin.');
      return;
    }

    openCropEditor();
  };

  const applyCrop = async () => {
    if (!selectedImageUri || !selectedImageSize || !cropStageLayout || !cropBox) {
      setProcessState('error');
      setStatusMessage('Kırpma alanı hazırlanamadı. Önce görseli aç ve alanı seç.');
      return;
    }

    const contain = getContainLayout(cropStageLayout, selectedImageSize);
    const originX = clamp((cropBox.x - contain.offsetX) / contain.scale, 0, selectedImageSize.width);
    const originY = clamp((cropBox.y - contain.offsetY) / contain.scale, 0, selectedImageSize.height);
    const cropWidth = clamp(cropBox.width / contain.scale, 1, selectedImageSize.width - originX);
    const cropHeight = clamp(cropBox.height / contain.scale, 1, selectedImageSize.height - originY);

    try {
      const result = await ImageManipulator.manipulateAsync(
        selectedImageUri,
        [
          {
            crop: {
              originX: Math.round(originX),
              originY: Math.round(originY),
              width: Math.round(cropWidth),
              height: Math.round(cropHeight),
            },
          },
        ],
        { compress: 1, format: ImageManipulator.SaveFormat.PNG, base64: true }
      );

      setSelectedImageUri(result.uri);
      setSelectedImageSize({ width: result.width ?? Math.round(cropWidth), height: result.height ?? Math.round(cropHeight) });
      setSelectedImageB64(result.base64 ? `data:image/png;base64,${result.base64}` : null);
      setCropApplied(true);
      setProcessState('selected');
      setMakeupResultB64(null);
      setHairColorResultB64(null);
      setAccessoryResultB64(null);
      setAccessoryEnabled(false);
      setMakeupPreviewKind(null);
      setMakeupError(null);
      setStatusMessage('Kırpma uygulandı. Görsel artık seçtiğin kadrajla hazır.');
      resetAgeAnalysis();
      closeCropEditor();
      void runAgeAnalysis(result.uri, 'before', 'uri');
    } catch (error) {
      const message = error instanceof Error ? error.message : 'Kırpma uygulanamadı.';
      setProcessState('error');
      setStatusMessage(message);
    }
  };

  const handlePreprocess = async (imageUri = selectedImageUri) => {
    if (!imageUri) return;
    const runId = ++preprocessRunRef.current;
    setPreprocessLoading(true);
    setPreprocessError(null);
    setPreprocessedB64(null);
    setPreprocessMeta(null);
    setLandmarkCount(null);
    setLandmarkPoints(null);
    setMakeupResultB64(null);
    setHairColorResultB64(null);
    setMakeupPreviewKind(null);
    setMakeupError(null);
    resetExpressionTransferState();
    setProResultB64(null);
    setProMetrics(null);
    setProLayers([]);
    setMakeupLoading(false);
    setMakeupError(null);
    setMakeupResultB64(null);
    setHairColorResultB64(null);
    setMakeupPreviewKind(null);
    setMakeupLayers([]);
    setAccessoryLoading(false);
    setAccessoryError(null);
    setAccessoryResultB64(null);
    setAccessoryEnabled(false);
    setEvalMetrics(null);
    setEvalSourceLabel(null);
    setEvalResultB64(null);
    setSpectrumGrayB64(null);
    setSpectrumBlueB64(null);
    setSpectrumRedB64(null);
    setAgeAfter(null);
    setProCompareHeld(false);
    proCompareOpacity.setValue(0);
    try {
      const data = await preprocessFromUri(imageUri);
      if (runId !== preprocessRunRef.current) return;
      if (!data.success) throw new Error(data.message ?? 'Preprocess failed');
      setPreprocessedB64(data.processed_image_b64);
      const bbox = Array.isArray(data.face_bbox) ? data.face_bbox : null;
      const processedSize = Array.isArray(data.processed_size) ? data.processed_size : null;
      if (bbox?.length === 4) {
        const bboxW = Number(bbox[2]);
        const bboxH = Number(bbox[3]);
        const fallbackScale = Math.min(256 / bboxW, 256 / bboxH);
        setPreprocessMeta({
          faceBBox: [Number(bbox[0]), Number(bbox[1]), Number(bbox[2]), Number(bbox[3])],
          processedSize: processedSize?.length === 2
            ? { width: Number(processedSize[0]), height: Number(processedSize[1]) }
            : { width: Math.max(1, Math.floor(bboxW * fallbackScale)), height: Math.max(1, Math.floor(bboxH * fallbackScale)) },
        });
      }
    } catch (e: any) {
      if (runId !== preprocessRunRef.current) return;
      setPreprocessError(e?.message ?? 'Unknown error');
    } finally {
      if (runId === preprocessRunRef.current) {
        setPreprocessLoading(false);
      }
    }
  };

  const handleLandmarks = async (imageB64 = preprocessedB64, backend = landmarkBackend) => {
    if (!imageB64) return;
    const runId = ++landmarkRunRef.current;
    setLandmarkLoading(true);
    setLandmarkError(null);
    try {
      const data = await landmarksFromBase64(imageB64, { landmarkBackend: backend });
      if (runId !== landmarkRunRef.current) return;
      if (!data.success) throw new Error(data.message ?? 'Landmark detection failed');
      setLandmarkCount(data.landmark_count);
      setLandmarkPoints(Array.isArray(data.landmarks) ? data.landmarks : null);
      setLandmarkPoints3d(Array.isArray(data.landmarks_3d) ? data.landmarks_3d : null);
    } catch (e: any) {
      if (runId !== landmarkRunRef.current) return;
      setLandmarkError(e?.message ?? 'Unknown error');
    } finally {
      if (runId === landmarkRunRef.current) {
        setLandmarkLoading(false);
      }
    }
  };

  useEffect(() => {
    if (!selectedImageUri) {
      preprocessRunRef.current += 1;
      landmarkRunRef.current += 1;
      return;
    }
    void handlePreprocess(selectedImageUri);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [selectedImageUri]);

  useEffect(() => {
    if (!preprocessedB64) return;
    setLandmarkCount(null);
    setLandmarkPoints(null);
    void handleLandmarks(preprocessedB64, landmarkBackend);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [preprocessedB64, landmarkBackend]);

  useEffect(() => {
    if (activeTab !== 'analysis' && showLandmarks) {
      setShowLandmarks(false);
    }
  }, [activeTab, showLandmarks]);

  const exportLandmarks = async (format: 'json' | 'csv') => {
    if (!landmarkPoints || landmarkPoints.length === 0) {
      Alert.alert('Landmark Hazir Degil', 'Once yuz noktalarini tespit etmelisin.');
      return;
    }

    const baseName = selectedImageName?.replace(/\.[^.]+$/, '') || 'landmarks';
    const content =
      format === 'json'
        ? JSON.stringify({ count: landmarkPoints.length, landmarks: landmarkPoints }, null, 2)
        : ['index,x,y', ...landmarkPoints.map(([x, y], index) => `${index},${x},${y}`)].join('\n');

    await downloadTextFile(
      content,
      `${baseName}-landmarks.${format}`,
      format === 'json' ? 'application/json;charset=utf-8' : 'text/csv;charset=utf-8',
    );
    setStatusMessage(`Landmark koordinatlari ${format.toUpperCase()} olarak indirildi.`);
  };

  const handleExpressionTransfer = async () => {
    if (!preprocessedB64 || !referenceExpressionUri) {
      setExpressionTransferError('Önce ana görseli ve referans ifade görselini seçmelisin.');
      return;
    }

    setExpressionTransferLoading(true);
    setExpressionTransferError(null);
    setExpressionTransferResultB64(null);
    setManualLipWarpResultB64(null);
    setManualLipWarpError(null);

    try {
      const effectSourceB64 = selectedImageB64 ?? preprocessedB64;
      const transferData = await transferExpressionFromBase64(effectSourceB64, referenceExpressionUri, expressionTransferIntensity, {
        landmarkBackend,
      });

      if (!transferData.success) {
        throw new Error(transferData.message ?? 'Expression transfer failed');
      }

      setExpressionTransferResultB64(transferData.result_image_b64);
      setAgeAfter(null);
      void runAgeAnalysis(transferData.result_image_b64, 'after', 'base64');

      try {
        const baselineData = await warpProFromBase64(effectSourceB64, 'lip_plump', expressionTransferIntensity, 2.8, { landmarkBackend });
        if (baselineData.success) {
          setManualLipWarpResultB64(baselineData.result_image_b64);
        } else {
          setManualLipWarpError(baselineData.message ?? 'Manual lip warp baseline could not be generated.');
        }
      } catch (baselineError) {
        const message = baselineError instanceof Error ? baselineError.message : 'Manual lip warp baseline could not be generated.';
        setManualLipWarpError(message);
      }
    } catch (error: any) {
      setExpressionTransferError(error?.message ?? 'Unknown expression transfer error');
      Alert.alert('AI Analizinde hata oluştu', error?.message ?? 'Unknown expression transfer error');
    } finally {
      setExpressionTransferLoading(false);
    }
  };

  const mergeWebGLWithImage = useCallback(
    (imageB64: string, webglCanvas: HTMLCanvasElement, width: number, height: number): Promise<string> => {
      return new Promise((resolve) => {
        const img = new (window as any).Image();
        img.onload = () => {
          const canvas = document.createElement('canvas');
          canvas.width = width;
          canvas.height = height;
          const ctx = canvas.getContext('2d');
          if (ctx) {
            ctx.drawImage(img, 0, 0, width, height);
            ctx.drawImage(webglCanvas, 0, 0, width, height);
          }
          resolve(canvas.toDataURL('image/png').split(',')[1] ?? '');
        };
        img.onerror = () => resolve('');
        img.src = `data:image/png;base64,${imageB64}`;
      });
    },
    []
  );

  useEffect(() => {
    if (Platform.OS === 'web' && activeTab === 'accessory2' && canvasRef2.current) {
      arEngineRef2.current = new AREngine(canvasRef2.current);
    }
    return () => {
      if (arEngineRef2.current) {
        arEngineRef2.current.dispose();
        arEngineRef2.current = null;
      }
    };
  }, [activeTab]);

  const [accessory2Loading, setAccessory2Loading] = useState(false);

  const updateAccessory2 = useCallback(async () => {
    const arEngine = arEngineRef2.current;
    if (!arEngine || !landmarkPoints3d || !preprocessMeta || !preprocessedB64) {
      setAccessory2ResultB64(null);
      return;
    }

    setAccessory2Loading(true);
    try {
      arEngine.setAccessories(
        accessories2.glasses,
        accessories2.hat,
        accessories2.earrings,
        accessories2.necklace,
        accessories2.tie,
        accessories2.mask
      );
      arEngine.setGlassesStyle(glassesStyle2);
      arEngine.setHatStyle(hatStyle2);
      arEngine.setEarringStyle(earringStyle2);
      arEngine.setNecklaceStyle(necklaceStyle2);
      arEngine.setTieStyle(tieStyle2);
      arEngine.setMaskStyle(maskStyle2);

      const renderWidth = preprocessMeta.processedSize.width;
      const renderHeight = preprocessMeta.processedSize.height;

      arEngine.update(landmarkPoints3d, renderWidth, renderHeight);

      const merged = await mergeWebGLWithImage(preprocessedB64, canvasRef2.current!, renderWidth, renderHeight);
      setAccessory2ResultB64(merged || null);
    } catch (e) {
      console.error('[Accessory2] Error updating:', e);
      setAccessory2ResultB64(null);
    } finally {
      setAccessory2Loading(false);
    }
  }, [
    accessories2,
    glassesStyle2,
    hatStyle2,
    earringStyle2,
    necklaceStyle2,
    tieStyle2,
    maskStyle2,
    landmarkPoints3d,
    preprocessMeta,
    preprocessedB64,
    mergeWebGLWithImage,
  ]);

  useEffect(() => {
    if (activeTab === 'accessory2') {
      void updateAccessory2();
    }
  }, [updateAccessory2, activeTab]);

  const getProIntensityValue = useCallback((operation: ProOperation) => {
    return (proOperationIntensity[operation] ?? LAB_DEFAULT_INTENSITY) / 100;
  }, [proOperationIntensity]);

  const runProOperation = useCallback(async (override?: { operation?: ProOperation }) => {
    if (!preprocessedB64) return;
    const operations = (override?.operation ? [override.operation] : activeProOperations)
      .filter((operation) => (proOperationIntensity[operation] ?? LAB_DEFAULT_INTENSITY) > 0);
    if (operations.length === 0) {
      setProResultB64(null);
      setEvalResultB64(null);
      setProMetrics(null);
      setEvalMetrics(null);
      return;
    }

    let effectSourceB64 = selectedImageB64 ?? preprocessedB64;
    let lastMetrics: ProMetrics | null = null;
    let lastSourceLabel: string | null = null;
    let lastSpectrumGray: string | null = null;
    let lastSpectrumBlue: string | null = null;
    let lastSpectrumRed: string | null = null;

    setProCompareHeld(false);

    setProLoading(true);
    setProError(null);
    try {
      for (const effectiveOperation of operations) {
        const effectiveIntensity = getProIntensityValue(effectiveOperation);
        if (effectiveOperation === 'aging' || effectiveOperation === 'deaging') {
          const data = await agingSamFromBase64(effectSourceB64, effectiveOperation, effectiveIntensity, {
            landmarkBackend,
          });
          if (!data.success) throw new Error(data.message ?? 'SAM aging failed');
          effectSourceB64 = data.result_image_b64;
          lastMetrics = null;  // SAM frekans metriği döndürmez
          lastSourceLabel = effectiveOperation === 'aging' ? 'SAM / Aging' : 'SAM / De-Aging';
          lastSpectrumGray = null;
          lastSpectrumBlue = null;
          lastSpectrumRed = null;
        } else {
          const data = await warpProFromBase64(effectSourceB64, effectiveOperation, effectiveIntensity, LAB_RBF_SMOOTH, {
            landmarkBackend,
            temporalSmoothing: true,
            emaAlpha: 0.62,
            streamId: 'pro-ui',
          });
          if (!data.success) throw new Error(data.message ?? 'Pro warp failed');
          effectSourceB64 = data.result_image_b64;
          lastMetrics = data.metrics ?? null;
          lastSourceLabel = `Warp / ${operations.map((operation) => PRO_LABEL[operation]).join(' + ')}`;
          lastSpectrumGray = null;
          lastSpectrumBlue = null;
          lastSpectrumRed = null;
        }
      }

      setProResultB64(effectSourceB64);
      setProMetrics(lastMetrics);
      setEvalMetrics(lastMetrics);
      setEvalSourceLabel(lastSourceLabel);
      setEvalResultB64(effectSourceB64);
      setSpectrumGrayB64(lastSpectrumGray);
      setSpectrumBlueB64(lastSpectrumBlue);
      setSpectrumRedB64(lastSpectrumRed);
      setAgeAfter(null);
      void runAgeAnalysis(effectSourceB64, 'after', 'base64');
    } catch (e: any) {
      setProError(e?.message ?? 'Unknown pro error');
    } finally {
      setProLoading(false);
    }
  }, [activeProOperations, getProIntensityValue, landmarkBackend, preprocessedB64, proOperationIntensity, runAgeAnalysis, selectedImageB64]);

  const applyMakeup = async () => {
    if (!preprocessedB64) return;

    const preset = MANUAL_MAKEUP_PRESETS.find((item) => item.key === makeupTarget) ?? MANUAL_MAKEUP_PRESETS[0];
    const hexColor = normalizeHexColor(makeupHexColor, preset.defaultColor);
    
    // Use last layer result if available, else preserve the selected image dimensions.
    const baseImageB64 = makeupLayers.length > 0 ? makeupLayers[makeupLayers.length - 1].resultB64 : (selectedImageB64 ?? preprocessedB64);

    setMakeupLoading(true);
    setMakeupError(null);

    try {
      const data = await applyMakeupFromBase64(baseImageB64, preset.backendRegion, hexColor, makeupIntensity, {
        landmarkBackend: 'hybrid',
        temporalSmoothing: true,
        emaAlpha: 0.62,
        streamId: 'makeup-ui',
      });

      if (!data.success) {
        throw new Error(data.message ?? 'Makeup effect failed');
      }

      // Add to layers array
      const newLayer: MakeupLayer = {
        id: `${Date.now()}-${Math.random()}`,
        region: preset.backendRegion,
        color: hexColor,
        intensity: makeupIntensity,
        resultB64: data.result_image_b64,
        locked: true, // Default to locked
      };
      
      setMakeupLayers([...makeupLayers, newLayer]);
      setMakeupResultB64(data.result_image_b64);
      setMakeupPreviewKind('makeup');
      setAgeAfter(null);
      void runAgeAnalysis(data.result_image_b64, 'after', 'base64');
    } catch (error: any) {
      setMakeupError(error?.message ?? 'Unknown makeup error');
    } finally {
      setMakeupLoading(false);
    }
  };

  const rebuildMakeupLayers = async (layersToApply: MakeupLayer[]) => {
    const baseImageB64 = selectedImageB64 ?? preprocessedB64;
    if (!baseImageB64) return;

    if (layersToApply.length === 0) {
      setMakeupLayers([]);
      setMakeupResultB64(null);
      setMakeupPreviewKind((current) => current === 'makeup' ? null : current);
      return;
    }

    setMakeupLoading(true);
    setMakeupError(null);

    try {
      let currentB64 = baseImageB64;
      const rebuiltLayers: MakeupLayer[] = [];

      for (const layer of layersToApply) {
        const data = await applyMakeupFromBase64(currentB64, layer.region, layer.color, layer.intensity, {
          landmarkBackend: 'hybrid',
          temporalSmoothing: true,
          emaAlpha: 0.62,
          streamId: 'makeup-ui',
        });

        if (!data.success) {
          throw new Error(data.message ?? 'Makeup layer rebuild failed');
        }

        currentB64 = data.result_image_b64;
        rebuiltLayers.push({ ...layer, resultB64: currentB64 });
      }

      setMakeupLayers(rebuiltLayers);
      setMakeupResultB64(currentB64);
      setMakeupPreviewKind('makeup');
      setAgeAfter(null);
      void runAgeAnalysis(currentB64, 'after', 'base64');
    } catch (error: any) {
      setMakeupError(error?.message ?? 'Makeup layers could not be rebuilt');
    } finally {
      setMakeupLoading(false);
    }
  };

  const runHairColor = async () => {
    const base = selectedImageB64 ?? preprocessedB64;
    if (!base) return;
    setHairColorLoading(true);
    setHairColorError(null);
    try {
      const data = await applyHairColorFromBase64(base, hairColorHex, hairColorIntensity);
      if (data.success && data.result_image_b64) {
        setHairColorResultB64(data.result_image_b64);
        setMakeupPreviewKind('hair');
      } else {
        setHairColorError(data.error ?? data.details ?? 'Saç rengi uygulanamadı.');
      }
    } catch (err: any) {
      setHairColorError(err?.message ?? 'Bilinmeyen hata.');
    } finally {
      setHairColorLoading(false);
    }
  };

  const runAccessoryPreview = useCallback(async () => {
    if (!preprocessedB64 || !accessoryEnabled) {
      setAccessoryResultB64(null);
      setMakeupPreviewKind((current) => current === 'accessory' ? null : current);
      return;
    }
    const effectSourceB64 = selectedImageB64 ?? preprocessedB64;

    const preset = ACCESSORY_PRESETS.find((item) => item.key === accessoryTarget) ?? ACCESSORY_PRESETS[0];
    const style = preset.styles.some((item) => item.key === accessoryStyle) ? accessoryStyle : preset.defaultStyle;

    setAccessoryLoading(true);
    setAccessoryError(null);

    try {
      const data = await applyAccessoryFromBase64(
        effectSourceB64,
        preset.key,
        style,
        preset.defaultColor,
        accessoryIntensity,
        accessoryScale,
        accessoryOffsetX,
        accessoryOffsetY,
        {
          landmarkBackend,
          temporalSmoothing: true,
          emaAlpha: 0.62,
          streamId: 'accessory-ui',
        }
      );

      if (!data.success) {
        throw new Error(data.message ?? data.details ?? 'Accessory effect failed');
      }

      setAccessoryResultB64(data.result_image_b64);
      setMakeupPreviewKind('accessory');
    } catch (error: any) {
      setAccessoryError(error?.message ?? 'Unknown accessory error');
    } finally {
      setAccessoryLoading(false);
    }
  }, [
    accessoryIntensity,
    accessoryEnabled,
    accessoryOffsetX,
    accessoryOffsetY,
    accessoryScale,
    accessoryStyle,
    accessoryTarget,
    landmarkBackend,
    preprocessedB64,
    selectedImageB64,
  ]);

  const toggleProOperation = (operation: ProOperation) => {
    setActiveProOperations((current) => {
      if (current.includes(operation)) {
        return current.filter((item) => item !== operation);
      }

      setProOperationIntensity((values) => ({
        ...values,
        [operation]: LAB_DEFAULT_INTENSITY,
      }));
      return [...current, operation];
    });
  };

  const adjustProOperationIntensity = (operation: ProOperation, delta: number) => {
    setProOperationIntensity((values) => {
      const nextValue = clamp((values[operation] ?? LAB_DEFAULT_INTENSITY) + delta, 0, 100);
      if (nextValue === 0) {
        setActiveProOperations((current) => current.filter((item) => item !== operation));
      }

      return {
        ...values,
        [operation]: nextValue,
      };
    });
  };

  const updateProOperationIntensity = (operation: ProOperation, value: number) => {
    setProOperationIntensity((values) => {
      const nextValue = clamp(value, 0, 100);
      if (nextValue === 0) {
        setActiveProOperations((current) => current.filter((item) => item !== operation));
      }

      return {
        ...values,
        [operation]: nextValue,
      };
    });
  };

  useEffect(() => {
    if (!preprocessedB64 || !accessoryEnabled) {
      setAccessoryResultB64(null);
      return;
    }

    if (accessoryDebounceRef.current) {
      clearTimeout(accessoryDebounceRef.current);
    }

    accessoryDebounceRef.current = setTimeout(() => {
      void runAccessoryPreview();
    }, 320);

    return () => {
      if (accessoryDebounceRef.current) {
        clearTimeout(accessoryDebounceRef.current);
      }
    };
  }, [accessoryEnabled, preprocessedB64, runAccessoryPreview]);

  useEffect(() => {
    if (!preprocessedB64) return;

    if (proDebounceRef.current) {
      clearTimeout(proDebounceRef.current);
    }

    proDebounceRef.current = setTimeout(() => {
      runProOperation();
    }, 280);

    return () => {
      if (proDebounceRef.current) {
        clearTimeout(proDebounceRef.current);
      }
    };
  }, [activeProOperations, preprocessedB64, proOperationIntensity, runProOperation]);

  const applyProLayer = async () => {
    // Run pro operation once and add to layers
    if (!preprocessedB64) return;
    const activeOperationsWithIntensity = activeProOperations
      .filter((operation) => (proOperationIntensity[operation] ?? LAB_DEFAULT_INTENSITY) > 0);
    if (activeOperationsWithIntensity.length === 0) return;
    const effectSourceB64 = selectedImageB64 ?? preprocessedB64;

    setProLoading(true);
    setProError(null);
    try {
      let resultB64: string | null = null;
      let workingB64 = effectSourceB64;
      for (const effectiveOperation of activeOperationsWithIntensity) {
        const effectiveIntensity = getProIntensityValue(effectiveOperation);
        if (effectiveOperation === 'aging' || effectiveOperation === 'deaging') {
          const data = await agingSamFromBase64(workingB64, effectiveOperation, effectiveIntensity, {
            landmarkBackend,
          });
          if (!data.success) throw new Error(data.message ?? 'SAM aging failed');
          workingB64 = data.result_image_b64;
        } else {
          const data = await warpProFromBase64(workingB64, effectiveOperation, effectiveIntensity, LAB_RBF_SMOOTH, {
            landmarkBackend,
            temporalSmoothing: true,
            emaAlpha: 0.62,
            streamId: 'pro-ui',
          });
          if (!data.success) throw new Error(data.message ?? 'Pro warp failed');
          workingB64 = data.result_image_b64;
        }
      }
      resultB64 = workingB64;

      if (resultB64) {
        const newLayer: ProLayer = {
          id: `${Date.now()}-${Math.random()}`,
          operation: activeOperationsWithIntensity[0],
          intensity: getProIntensityValue(activeOperationsWithIntensity[0]),
          resultB64,
          locked: true,
        };
        
        setProLayers([...proLayers, newLayer]);
        setProResultB64(resultB64);
      }
    } catch (error: any) {
      setProError(error?.message ?? 'Unknown pro error');
    } finally {
      setProLoading(false);
    }
  };

  const exportCsv = async () => {
    if (!fullQualitySourceB64 || !evalResultB64 || !evalMetrics) {
      Alert.alert('Rapor Hazır Değil', 'Önce bir warp veya frequency işlemi çalıştırmalısın.');
      return;
    }

    try {
      const data = await exportEvaluationReportFromBase64(
        'csv',
        evalSourceLabel ?? 'Unknown Operation',
        fullQualitySourceB64,
        evalResultB64,
        { mse: evalMetrics.mse, psnr: evalMetrics.psnr, ssim: evalMetrics.ssim }
      );
      if (!data.success) throw new Error(data.message ?? 'CSV export failed');

      await downloadBase64File(data.file_b64, data.file_name, data.mime_type ?? 'text/csv');
      setStatusMessage('CSV raporu indirildi.');
    } catch (error: any) {
      Alert.alert('Export Hatası', error?.message ?? 'CSV export basarisiz.');
    }
  };

  const exportPdf = async () => {
    if (!fullQualitySourceB64 || !evalResultB64 || !evalMetrics) {
      Alert.alert('Rapor Hazır Değil', 'Önce bir warp veya frequency işlemi çalıştırmalısın.');
      return;
    }

    try {
      const data = await exportEvaluationReportFromBase64(
        'pdf',
        evalSourceLabel ?? 'Unknown Operation',
        fullQualitySourceB64,
        evalResultB64,
        { mse: evalMetrics.mse, psnr: evalMetrics.psnr, ssim: evalMetrics.ssim }
      );
      if (!data.success) throw new Error(data.message ?? 'PDF export failed');

      await downloadBase64File(data.file_b64, data.file_name, data.mime_type ?? 'application/pdf');
      setStatusMessage('PDF raporu indirildi.');
    } catch (error: any) {
      Alert.alert('Export Hatası', error?.message ?? 'PDF export basarisiz.');
    }
  };

  const clearWorkspace = () => {
    setSelectedImageName(null);
    setSelectedImageUri(null);
    setSelectedImageSize(null);
    setSelectedImageB64(null);
    setStatusMessage('Henüz görsel seçilmedi.');
    setProcessState('idle');
    setCropApplied(false);
    setPreprocessLoading(false);
    setPreprocessError(null);
    setPreprocessedB64(null);
    setPreprocessMeta(null);
    setLandmarkLoading(false);
    setLandmarkError(null);
    setLandmarkCount(null);
    setLandmarkPoints(null);
    setShowLandmarks(false);
    resetExpressionTransferState();
    setProLoading(false);
    setProError(null);
    setProResultB64(null);
    setProMetrics(null);
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
    setEvalMetrics(null);
    setEvalSourceLabel(null);
    setEvalResultB64(null);
    setSpectrumGrayB64(null);
    setSpectrumBlueB64(null);
    setSpectrumRedB64(null);
    resetAgeAnalysis();
    closeCropEditor();
  };

  const renderAgeAnalysisCard = () => (
    <View style={styles.aiAnalysisCard}>
      <View style={styles.aiAnalysisHeaderRow}>
        <View style={{ flex: 1 }}>
          <ThemedText type="defaultSemiBold">AI Analysis</ThemedText>
          <ThemedText style={styles.helperText}>DeepFace age estimation</ThemedText>
        </View>
        {ageLoading ? (
          <ActivityIndicator />
        ) : (
          <View style={styles.aiAnalysisPill}>
            <ThemedText style={styles.aiAnalysisPillText}>Live</ThemedText>
          </View>
        )}
      </View>

      <View style={styles.ageCompareRow}>
        <View style={styles.ageStatCard}>
          <ThemedText style={styles.ageStatLabel}>Before</ThemedText>
          <ThemedText style={styles.ageStatValue}>{ageBefore != null ? String(ageBefore) : '—'}</ThemedText>
        </View>
        <View style={styles.ageStatCard}>
          <ThemedText style={styles.ageStatLabel}>After</ThemedText>
          <ThemedText style={styles.ageStatValue}>{ageAfter != null ? String(ageAfter) : '—'}</ThemedText>
        </View>
      </View>

      <ThemedText style={styles.helperText}>
        {ageBefore != null && ageAfter != null
          ? `Before: ${ageBefore} -> After: ${ageAfter}`
          : ageBefore != null
            ? `Estimated Age: ${ageBefore}`
            : 'Estimated Age: —'}
      </ThemedText>
      {ageError ? <Text style={styles.errorText}>{ageError}</Text> : null}
    </View>
  );

  const renderExpressionComparisonCard = () => (
    <View style={styles.compareCard}>
      <View style={styles.compareHeaderRow}>
        <View style={{ flex: 1 }}>
          <ThemedText type="defaultSemiBold">AI Expression Transfer</ThemedText>
          <ThemedText style={styles.helperText}>{'Reference expression -> target face'}</ThemedText>
        </View>
        <View style={styles.compareHintPill}>
          <ThemedText style={styles.compareHintText}>AI vs Manual Lip Warp</ThemedText>
        </View>
      </View>

      <View style={styles.sideBySide}>
        <View style={styles.sideBox}>
          <ThemedText style={styles.sideLabel}>AI Transfer</ThemedText>
          <Pressable onPress={() => setLightboxUri(`data:image/png;base64,${expressionTransferResultB64}`)}>
            <Image source={{ uri: `data:image/png;base64,${expressionTransferResultB64}` }} style={styles.sideImage} contentFit="contain" />
          </Pressable>
        </View>
        <View style={styles.sideBox}>
          <ThemedText style={styles.sideLabel}>Manual Lip Warp</ThemedText>
          {manualLipWarpResultB64 ? (
            <Pressable onPress={() => setLightboxUri(`data:image/png;base64,${manualLipWarpResultB64}`)}>
              <Image source={{ uri: `data:image/png;base64,${manualLipWarpResultB64}` }} style={styles.sideImage} contentFit="contain" />
            </Pressable>
          ) : (
            <View style={styles.sideImagePlaceholder}>
              <ThemedText style={styles.helperText}>Baseline is generating or unavailable.</ThemedText>
            </View>
          )}
        </View>
      </View>

      {manualLipWarpError ? <Text style={styles.errorText}>{manualLipWarpError}</Text> : null}
    </View>
  );

  const pageBackground = colorScheme === 'dark' ? STUDIO.bg : '#F7F4FB';
  const panelBackground = colorScheme === 'dark' ? '#121313' : '#FFFFFF';
  const previewBackground = colorScheme === 'dark' ? '#101112' : '#FFFFFF';
  const panelBorder = colorScheme === 'dark' ? 'rgba(255,255,255,0.10)' : 'rgba(20,20,20,0.08)';
  const softSurface = colorScheme === 'dark' ? 'rgba(255,255,255,0.06)' : 'rgba(15,15,15,0.05)';
  const accent = '#8B5CF6';
  const mutedText = colorScheme === 'dark' ? '#9CA3AF' : '#64748B';
  const activeAccessoryPreset = ACCESSORY_PRESETS.find((item) => item.key === accessoryTarget) ?? ACCESSORY_PRESETS[0];
  const fullQualitySourceB64 = selectedImageB64 ?? preprocessedB64;
  const getCurrentResultB64 = () => {
    if (activeTab === 'prolab' && proResultB64) return proResultB64;
    if (activeTab === 'makeup') {
      if (makeupPreviewKind === 'makeup' && makeupResultB64) return makeupResultB64;
      if (makeupPreviewKind === 'hair' && hairColorResultB64) return hairColorResultB64;
      return makeupResultB64 ?? hairColorResultB64;
    }
    if (activeTab === 'accessory' && accessoryResultB64) return accessoryResultB64;
    if (activeTab === 'accessory2' && accessory2ResultB64) return accessory2ResultB64;
    if (activeTab === 'expression' && expressionTransferResultB64) return expressionTransferResultB64;
    return null;
  };
  const currentResultB64 = getCurrentResultB64();

  return (
    <StudioScreen style={{ backgroundColor: pageBackground }}>
      <View style={[styles.mainContent, { backgroundColor: 'transparent' }]}>
      <View style={styles.header}>
        <View style={styles.headerTitleGroup}>
          <ThemedText type="title" style={styles.headerTitle}>Oluştur</ThemedText>
          <ThemedText style={[styles.headerSubtitle, { color: mutedText }]}>
            {mode === 'live' ? 'Kameranı aç, slider çevir, yüzün anlık değişsin.' : 'Yüzünüzü dilediğiniz gibi şekillendirin.'}
          </ThemedText>

          <View style={[styles.modeToggle, { backgroundColor: softSurface, borderColor: panelBorder }]}>
            <Pressable
              onPress={() => setMode('photo')}
              style={[
                styles.modeOption,
                mode === 'photo' && { backgroundColor: Colors[colorScheme].tint },
              ]}>
              <Ionicons
                name="image-outline"
                size={14}
                color={mode === 'photo' ? tintTextColor : colors.text}
              />
              <ThemedText
                style={[
                  styles.modeOptionText,
                  { color: mode === 'photo' ? tintTextColor : colors.text },
                ]}>
                Fotoğraf
              </ThemedText>
            </Pressable>
            <Pressable
              onPress={() => setMode('live')}
              style={[
                styles.modeOption,
                mode === 'live' && { backgroundColor: Colors[colorScheme].tint },
              ]}>
              <Ionicons
                name="videocam-outline"
                size={14}
                color={mode === 'live' ? tintTextColor : colors.text}
              />
              <ThemedText
                style={[
                  styles.modeOptionText,
                  { color: mode === 'live' ? tintTextColor : colors.text },
                ]}>
                Anlık Kamera
              </ThemedText>
              <View style={styles.modeOptionDot} />
            </Pressable>
          </View>
        </View>
        <View style={styles.headerActions}>
          <Pressable style={[styles.topActionButton, { backgroundColor: softSurface, borderColor: panelBorder }]} onPress={clearWorkspace}>
            <Ionicons name="trash-outline" size={16} color="#FFFFFF" />
            <ThemedText style={styles.topActionText}>Temizle</ThemedText>
          </Pressable>
          <Pressable style={[styles.topActionButton, styles.topActionPrimary, { opacity: evalMetrics ? 1 : 0.5 }]} onPress={exportPdf} disabled={!evalMetrics}>
            <Ionicons name="download-outline" size={16} color="#000000" />
            <ThemedText style={styles.topActionPrimaryText}>Kaydet</ThemedText>
          </Pressable>
        </View>
      </View>

      <View style={styles.content}>
        {mode === 'live' ? (
          <View style={styles.liveWrap}>
            <LiveWarpCamera onCapture={handleLiveCapture} isDark={colorScheme === 'dark'} />
          </View>
        ) : (
        <View style={[styles.workspace, isWide && styles.workspaceWide]}>
          {/* SOL PANEL - ORIJINAL FOTOĞRAF + SEÇME BUTONU */}
          <View
            style={[
              styles.panel,
              styles.previewPanel,
              {
                backgroundColor: previewBackground,
                borderColor: panelBorder,
                flex: 1,
                padding: 12,
              },
              Platform.OS === 'web' ? ({ order: 2 } as any) : null,
            ]}>
            {selectedImageUri ? (
              <View style={{ flex: 1, position: 'relative', overflow: 'hidden', borderRadius: 16 }}>
                <Image source={{ uri: selectedImageUri }} style={{ width: '100%', height: '100%' }} contentFit="contain" />
                <View style={styles.previewBadge}>
                  <ThemedText style={styles.previewBadgeText}>Önceki</ThemedText>
                </View>
              </View>
            ) : (
              <Pressable
                style={[
                  styles.uploadDropzone,
                  {
                    flex: 1,
                    borderColor: colorScheme === 'dark' ? 'rgba(139,92,246,0.35)' : 'rgba(139,92,246,0.30)',
                    backgroundColor: colorScheme === 'dark' ? 'rgba(139,92,246,0.08)' : 'rgba(139,92,246,0.06)',
                  },
                ]}
                onPress={pickImage}>
                <View style={[styles.uploadIconBubble, { backgroundColor: colorScheme === 'dark' ? 'rgba(139,92,246,0.16)' : 'rgba(139,92,246,0.12)' }]}>
                  <Ionicons name="image-outline" size={40} color={accent} />
                </View>
                <ThemedText style={[styles.uploadDropzoneTitle, { color: accent }]}>Fotoğraf Seç</ThemedText>
              </Pressable>
            )}
          </View>

          {/* SAĞDA PANEL - SONUÇ FOTOĞRAF */}
          <View
            style={[
              styles.panel,
              styles.previewPanel,
              {
                backgroundColor: previewBackground,
                borderColor: panelBorder,
                flex: 1,
                padding: 12,
              },
              Platform.OS === 'web' ? ({ order: 3 } as any) : null,
            ]}>
            {selectedImageUri && showLandmarks && landmarkPoints && preprocessMeta ? (
              <Pressable
                style={{ flex: 1, position: 'relative', overflow: 'hidden', borderRadius: 16 }}
                onLayout={(event) => {
                  const { width: stageWidth, height: stageHeight } = event.nativeEvent.layout;
                  setLandmarkPreviewLayout({ width: stageWidth, height: stageHeight });
                }}
                onPress={() => setLightboxUri(selectedImageUri)}>
                <Image source={{ uri: selectedImageUri }} style={{ width: '100%', height: '100%' }} contentFit="contain" />
                <View pointerEvents="none" style={StyleSheet.absoluteFill}>
                  {landmarkOverlayPoints.map((point, index) => (
                    <View
                      key={`${index}-${Math.round(point.x)}-${Math.round(point.y)}`}
                      style={[
                        styles.landmarkOverlayDot,
                        {
                          left: point.x,
                          top: point.y,
                        },
                      ]}
                    />
                  ))}
                </View>
                <View style={[styles.previewBadge, { backgroundColor: 'rgba(8,145,178,0.86)' }]}>
                  <ThemedText style={styles.previewBadgeText}>Noktalar</ThemedText>
                </View>
              </Pressable>
            ) : selectedImageUri && currentResultB64 ? (
              <View style={{ flex: 1, position: 'relative', overflow: 'hidden', borderRadius: 16 }}>
                <Image source={{ uri: `data:image/png;base64,${currentResultB64}` }} style={{ width: '100%', height: '100%' }} contentFit="contain" />
                <View style={[styles.previewBadge, { backgroundColor: 'rgba(160,32,240,0.8)' }]}>
                  <ThemedText style={styles.previewBadgeText}>Sonraki</ThemedText>
                </View>
              </View>
            ) : (
              <View style={[styles.emptyPreview, { backgroundColor: softSurface }]}>
                <Ionicons name="sparkles-outline" size={32} color={colors.text} />
                <ThemedText style={styles.helperText}>Efekt beklemede</ThemedText>
              </View>
            )}
          </View>

          <View
            style={[
              styles.panel,
              styles.featurePanel,
              {
                backgroundColor: panelBackground,
                borderColor: panelBorder,
                flexDirection: 'row',
                padding: 0,
                overflow: 'hidden',
                minWidth: 360,
              },
              Platform.OS === 'web' ? ({ order: 1 } as any) : null,
            ]}>
            
            {/* Vertical Toolbar */}
            <View style={{ width: 64, borderRightWidth: 1, borderRightColor: panelBorder, backgroundColor: colorScheme === 'dark' ? 'rgba(0,0,0,0.2)' : 'rgba(0,0,0,0.02)', paddingTop: 20, alignItems: 'center', gap: 20 }}>
              {(['analysis', 'expression', 'prolab', 'makeup', 'accessory', 'accessory2'] as TabKey[]).map((tab) => {
                  const labels: Record<TabKey, string> = { analysis: 'Analiz', expression: 'İfade', prolab: 'Lab', makeup: 'Makyaj', accessory: 'Aksesuar', accessory2: 'Aksesuar 2' };
                  const icons: Record<TabKey, any> = {
                    analysis: 'scan-outline',
                    expression: 'happy-outline',
                    prolab: 'flask-outline',
                    makeup: 'color-palette-outline',
                    accessory: 'glasses-outline',
                    accessory2: 'sparkles-outline'
                  };
                  const isActive = activeTab === tab;
                  return (
                    <Pressable
                      key={tab}
                      onPress={() => setActiveTab(tab)}
                      style={[
                        {
                          width: 48,
                          height: 48,
                          borderRadius: 12,
                          alignItems: 'center',
                          justifyContent: 'center',
                          backgroundColor: isActive ? Colors[colorScheme].tint : 'transparent',
                        }
                      ]}>
                      <Ionicons name={icons[tab]} size={22} color={isActive ? tintTextColor : colors.text} />
                      <Text style={{ fontSize: 9, marginTop: 4, fontWeight: '700', color: isActive ? tintTextColor : mutedText, letterSpacing: -0.2 }}>{labels[tab]}</Text>
                    </Pressable>
                  );
              })}
            </View>

            {/* Main Content Area */}
            <View style={{ flex: 1 }}>
            <ScrollView
              style={[styles.featurePanelScroller, { padding: 16 }]}
              contentContainerStyle={styles.featurePanelContent}
              showsVerticalScrollIndicator>
            <View style={styles.featureHeader}>
              <ThemedText type="subtitle" style={styles.panelTitle}>Kontrol Paneli</ThemedText>
              <ThemedText style={[styles.featureHeaderSub, { color: accent }]}>Gelişmiş Parametreler</ThemedText>
            </View>

            {activeTab === 'analysis' && (
              <>
            {/* Landmark model selector (FR-7.5) */}
            <View style={{ marginBottom: 12 }}>
              <ThemedText style={[styles.sideLabel, { marginBottom: 6, textAlign: 'left' }]}>Landmark Modeli</ThemedText>
              <View style={{ flexDirection: 'row', gap: 6 }}>
                {(['mediapipe', 'dlib', 'hybrid'] as const).map((opt) => (
                  <Pressable
                    key={opt}
                    onPress={() => setLandmarkBackend(opt)}
                    style={{
                      flex: 1,
                      paddingVertical: 7,
                      borderRadius: 8,
                      alignItems: 'center',
                      backgroundColor: landmarkBackend === opt ? Colors[colorScheme].tint : softSurface,
                      borderWidth: 1,
                      borderColor: landmarkBackend === opt ? Colors[colorScheme].tint : panelBorder,
                    }}>
                    <ThemedText style={{
                      fontSize: 11,
                      fontWeight: landmarkBackend === opt ? '700' : '400',
                      color: landmarkBackend === opt ? tintTextColor : colors.text,
                    }}>
                      {opt === 'mediapipe' ? 'MediaPipe' : opt === 'dlib' ? 'Dlib' : 'Hybrid'}
                    </ThemedText>
                  </Pressable>
                ))}
              </View>
              <View style={[styles.landmarkRow, { marginTop: 10 }]}>
                <View style={{ flex: 1 }}>
                  <ThemedText style={styles.helperText}>
                    {preprocessLoading
                      ? 'Yüz otomatik tespit ediliyor...'
                      : landmarkLoading
                        ? `${landmarkBackend === 'mediapipe' ? 'MediaPipe' : landmarkBackend === 'dlib' ? 'Dlib' : 'Hybrid'} noktaları hazırlanıyor...`
                        : landmarkCount != null
                          ? `${landmarkCount} nokta bulundu`
                          : selectedImageUri
                            ? 'Fotoğraf yüklenince otomatik tespit başlar'
                            : 'Önce fotoğraf seç'}
                  </ThemedText>
                  <ThemedText style={[styles.helperText, { fontSize: 11, opacity: 0.5 }]}>
                    {landmarkBackend === 'mediapipe' ? 'MediaPipe' : landmarkBackend === 'dlib' ? 'Dlib' : 'Hybrid'} modeli aktif
                  </ThemedText>
                </View>
                <Pressable
                  accessibilityRole="switch"
                  accessibilityState={{ checked: showLandmarks, disabled: !landmarkPoints || landmarkLoading }}
                  onPress={() => setShowLandmarks((value) => !value)}
                  disabled={!landmarkPoints || landmarkLoading}
                  style={[styles.landmarkToggleWrap, { opacity: landmarkPoints && !landmarkLoading ? 1 : 0.45 }]}>
                  <ThemedText style={[styles.landmarkToggleLabel, { color: colors.text }]}>Noktaları göster</ThemedText>
                  <View style={[
                    styles.landmarkToggleTrack,
                    {
                      backgroundColor: showLandmarks ? Colors[colorScheme].tint : softSurface,
                      borderColor: showLandmarks ? Colors[colorScheme].tint : panelBorder,
                    },
                  ]}>
                    <View style={[
                      styles.landmarkToggleThumb,
                      {
                        transform: [{ translateX: showLandmarks ? 18 : 0 }],
                        backgroundColor: showLandmarks ? tintTextColor : colors.text,
                      },
                    ]} />
                  </View>
                </Pressable>
              </View>
            </View>
            {preprocessError ? <Text style={styles.errorText}>{preprocessError}</Text> : null}
            {landmarkError ? <Text style={styles.errorText}>{landmarkError}</Text> : null}
            {landmarkPoints ? (
              <View style={styles.agingRow}>
                <Pressable
                  style={[styles.cvButton, { flex: 1, backgroundColor: softSurface, borderWidth: 1, borderColor: panelBorder }]}
                  onPress={() => exportLandmarks('json')}>
                  <ThemedText style={styles.iconActionText}>JSON Export</ThemedText>
                </Pressable>
                <Pressable
                  style={[styles.cvButton, { flex: 1, backgroundColor: softSurface, borderWidth: 1, borderColor: panelBorder }]}
                  onPress={() => exportLandmarks('csv')}>
                  <ThemedText style={styles.iconActionText}>CSV Export</ThemedText>
                </Pressable>
              </View>
            ) : null}

              </>
            )}

            {activeTab === 'expression' && (
              <>
            {/* Section 3: Expression Transfer */}
            <View style={styles.sectionDivider} />
            <View style={styles.sectionHeader}>
              <View style={[styles.stepBadge, { backgroundColor: accent }]}><Text style={styles.stepBadgeText}>3</Text></View>
              <ThemedText type="defaultSemiBold">İfade Transferi</ThemedText>
            </View>
            <ThemedText style={styles.helperText}>
              Referans ifade görselindeki yüz ifadesini ana görsele aktarıp sonuçta yaş değişimini otomatik ölçer.
            </ThemedText>
            <Pressable
              style={[styles.cvButton, { backgroundColor: Colors[colorScheme].tint, opacity: preprocessedB64 ? 1 : 0.4 }]}
              onPress={pickReferenceExpressionImage}
              disabled={!preprocessedB64}>
              <ThemedText style={[styles.cvButtonText, { color: tintTextColor }]}>Reference Expression Image Seç</ThemedText>
            </Pressable>
            {referenceExpressionName ? (
              <View style={styles.fileCard}>
                <ThemedText type="defaultSemiBold">Referans Görsel</ThemedText>
                <ThemedText style={styles.fileText}>{referenceExpressionName}</ThemedText>
                <ThemedText style={styles.fileText}>
                  {referenceExpressionSize ? `${referenceExpressionSize.width} x ${referenceExpressionSize.height}` : 'Boyut bilinmiyor'}
                </ThemedText>
              </View>
            ) : null}
            {referenceExpressionUri ? (
              <Pressable onPress={() => setLightboxUri(referenceExpressionUri)}>
                <Image source={{ uri: referenceExpressionUri }} style={styles.resultImage} contentFit="contain" />
              </Pressable>
            ) : null}

            <ThemedText style={styles.helperText}>Transfer yoğunluğu: {expressionTransferIntensity.toFixed(2)}</ThemedText>
            <Slider
              style={styles.nativeSlider}
              minimumValue={0}
              maximumValue={1}
              step={0.01}
              value={expressionTransferIntensity}
              onValueChange={setExpressionTransferIntensity}
              minimumTrackTintColor={colors.tint}
              maximumTrackTintColor="rgba(120,120,120,0.25)"
              thumbTintColor={colors.tint}
              disabled={!preprocessedB64 || !referenceExpressionUri}
            />

            <Pressable
              style={[styles.cvButton, { backgroundColor: Colors[colorScheme].tint, opacity: preprocessedB64 && referenceExpressionUri ? 1 : 0.4 }]}
              onPress={handleExpressionTransfer}
              disabled={!preprocessedB64 || !referenceExpressionUri || expressionTransferLoading}>
              {expressionTransferLoading
                ? <ActivityIndicator color={tintTextColor} />
                : <ThemedText style={[styles.cvButtonText, { color: tintTextColor }]}>İfadeyi Aktar</ThemedText>}
            </Pressable>
            {expressionTransferError ? <Text style={styles.errorText}>{expressionTransferError}</Text> : null}

            {expressionTransferResultB64 && preprocessedB64 ? renderExpressionComparisonCard() : null}
            {expressionTransferResultB64 && preprocessedB64 ? renderAgeAnalysisCard() : null}

              </>
            )}

            {activeTab === 'prolab' && (
              <>
            {/* Section 4: Lab */}
            <View style={styles.sectionDivider} />
            <View style={styles.sectionHeader}>
              <View style={[styles.stepBadge, { backgroundColor: accent }]}><Text style={styles.stepBadgeText}>4</Text></View>
              <ThemedText type="defaultSemiBold">Lab (Canlı)</ThemedText>
            </View>
            <ThemedText style={styles.helperText}>
              Operasyon: {activeProOperations.length > 0 ? activeProOperations.map((operation) => PRO_LABEL[operation]).join(' + ') : 'Kapalı'}
            </ThemedText>
            <View style={styles.warpGrid}>
              {PRO_OPERATIONS.map((op) => {
                const isActive = activeProOperations.includes(op);
                const isHovered = hoveredProOperation === op;
                const intensity = proOperationIntensity[op] ?? LAB_DEFAULT_INTENSITY;
                return (
                  <Pressable
                    key={op}
                    style={[
                      styles.warpOpButton,
                      {
                        backgroundColor: isActive ? Colors[colorScheme].tint : 'rgba(120,120,120,0.15)',
                        opacity: preprocessedB64 ? 1 : 0.4,
                        paddingHorizontal: isActive ? 12 : 36,
                        paddingVertical: isActive ? 12 : 10,
                        minHeight: 98,
                      },
                    ]}
                    onHoverIn={() => setHoveredProOperation(op)}
                    onHoverOut={() => setHoveredProOperation((current) => current === op ? null : current)}
                    onPress={isActive ? undefined : () => toggleProOperation(op)}
                    disabled={!preprocessedB64 && !isActive}>
                    <Pressable
                      style={styles.labOperationLabelWrap}
                      onPress={() => toggleProOperation(op)}
                      disabled={!preprocessedB64}>
                      <ThemedText style={[styles.warpOpText, { color: isActive ? tintTextColor : colors.text }]}>
                        {PRO_LABEL[op]}
                      </ThemedText>
                      {isActive ? (
                        <ThemedText style={[styles.labOperationValue, { color: isActive ? tintTextColor : colors.text }]}>
                          {intensity}%
                        </ThemedText>
                      ) : null}
                    </Pressable>
                    {isActive ? (
                      <Pressable
                        pointerEvents={isHovered ? 'auto' : 'none'}
                        style={[styles.labIntensityControl, styles.labIntensityControlLeft, { opacity: isHovered ? 1 : 0 }]}
                        onHoverIn={() => setHoveredProOperation(op)}
                        onPress={(event) => {
                          event.stopPropagation();
                          adjustProOperationIntensity(op, -LAB_INTENSITY_STEP);
                        }}>
                        <ThemedText style={[styles.labIntensityControlText, { color: tintTextColor }]}>-</ThemedText>
                      </Pressable>
                    ) : null}
                    {isActive ? (
                      <Pressable
                        pointerEvents={isHovered ? 'auto' : 'none'}
                        style={[styles.labIntensityControl, styles.labIntensityControlRight, { opacity: isHovered ? 1 : 0 }]}
                        onHoverIn={() => setHoveredProOperation(op)}
                        onPress={(event) => {
                          event.stopPropagation();
                          adjustProOperationIntensity(op, LAB_INTENSITY_STEP);
                        }}>
                        <ThemedText style={[styles.labIntensityControlText, { color: tintTextColor }]}>+</ThemedText>
                      </Pressable>
                    ) : null}
                    {isActive && (
                      <Slider
                        style={{ width: '100%', height: 30, marginTop: 4 }}
                        minimumValue={0}
                        maximumValue={100}
                        step={1}
                        value={intensity}
                        onValueChange={(val) => {
                          updateProOperationIntensity(op, val);
                        }}
                        minimumTrackTintColor={tintTextColor}
                        maximumTrackTintColor={colorScheme === 'dark' ? 'rgba(0,0,0,0.25)' : 'rgba(255,255,255,0.45)'}
                        thumbTintColor={tintTextColor}
                      />
                    )}
                  </Pressable>
                );
              })}
            </View>

            <Pressable
              style={[styles.cvButton, { backgroundColor: Colors[colorScheme].tint, opacity: preprocessedB64 && activeProOperations.length > 0 ? 1 : 0.5 }]}
              onPress={applyProLayer}
              disabled={!preprocessedB64 || activeProOperations.length === 0 || proLoading}>
              <ThemedText style={[styles.cvButtonText, { color: tintTextColor }]}>Efekti Uygula</ThemedText>
            </Pressable>

            {proLoading ? <ActivityIndicator /> : null}
            {proError ? <Text style={styles.errorText}>{proError}</Text> : null}

            {/* Lab layers display */}
            {proLayers.length > 0 && (
              <View style={styles.makeupLayersContainer}>
                <ThemedText style={styles.makeupLayersTitle}>Efekt Katmanları</ThemedText>
                {proLayers.map((layer, idx) => {
                  return (
                    <View key={layer.id} style={styles.makeupLayerRow}>
                      <Switch
                        value={layer.locked}
                        onValueChange={(newVal) => {
                          const updated = [...proLayers];
                          updated[idx].locked = newVal;
                          setProLayers(updated);
                          if (newVal) {
                            setProResultB64(layer.resultB64);
                          }
                        }}
                      />
                      <View style={[styles.makeupLayerColor, { backgroundColor: 'rgba(100,200,255,0.5)' }]} />
                      <ThemedText style={styles.makeupLayerLabel}>{PRO_LABEL[layer.operation]} {Math.round(layer.intensity * 100)}%</ThemedText>
                      <Pressable
                        onPress={() => {
                          // When deleting a layer, also delete all layers after it
                          const updated = proLayers.slice(0, idx);
                          setProLayers(updated);
                          if (updated.length > 0) {
                            setProResultB64(updated[updated.length - 1].resultB64);
                          } else {
                            setProResultB64(null);
                          }
                        }}
                        style={styles.makeupLayerDeleteBtn}>
                      </Pressable>
                    </View>
                  );
                })}
              </View>
            )}

            {proResultB64 && preprocessedB64 ? renderAgeAnalysisCard() : null}

              </>
            )}

            {activeTab === 'accessory' && (
              <>
            {/* Section 5: Accessories */}
            <View style={styles.sectionDivider} />
            <View style={styles.sectionHeader}>
              <View style={[styles.stepBadge, { backgroundColor: accent }]}><Text style={styles.stepBadgeText}>5</Text></View>
              <ThemedText type="defaultSemiBold">Aksesuarlar</ThemedText>
            </View>
            <ThemedText style={styles.helperText}>
              Gozluk ve biyik efektleri yuz noktalarina gore otomatik konumlanir; boyut ve kaydirma ile ince ayar yapabilirsin.
            </ThemedText>

            <View style={styles.warpGrid}>
              {ACCESSORY_PRESETS.map((preset) => {
                const isActive = accessoryEnabled && accessoryTarget === preset.key;
                return (
                  <Pressable
                    key={preset.key}
                    style={[
                      styles.warpOpButton,
                      {
                        backgroundColor: isActive ? Colors[colorScheme].tint : 'rgba(120,120,120,0.15)',
                        opacity: preprocessedB64 ? 1 : 0.4,
                      },
                    ]}
                    onPress={() => {
                      if (isActive) {
                        setAccessoryEnabled(false);
                        setAccessoryResultB64(null);
                        setMakeupPreviewKind((current) => current === 'accessory' ? null : current);
                        setAccessoryError(null);
                        return;
                      }

                      setAccessoryEnabled(true);
                      setAccessoryTarget(preset.key);
                      setAccessoryStyle(preset.defaultStyle);
                      setAccessoryScale(1);
                      setAccessoryOffsetX(0);
                      setAccessoryOffsetY(0);
                    }}
                    disabled={!preprocessedB64}>
                    <ThemedText style={[styles.warpOpText, { color: isActive ? tintTextColor : colors.text }]}>
                      {preset.label}
                    </ThemedText>
                  </Pressable>
                );
              })}
            </View>

            <View style={styles.accessoryModelGrid}>
              {activeAccessoryPreset.styles.map((styleOption) => {
                const isActive = accessoryStyle === styleOption.key;
                return (
                  <Pressable
                    key={styleOption.key}
                    style={[
                      styles.accessoryModelCard,
                      {
                        backgroundColor: isActive ? 'rgba(139,92,246,0.16)' : 'rgba(120,120,120,0.10)',
                        borderColor: isActive ? Colors[colorScheme].tint : panelBorder,
                        opacity: preprocessedB64 ? 1 : 0.4,
                      },
                    ]}
                    onPress={() => {
                      setAccessoryEnabled(true);
                      setAccessoryStyle(styleOption.key);
                    }}
                    disabled={!preprocessedB64}>
                    <View style={styles.accessoryThumbWrap}>
                      <Image
                        source={{ uri: accessoryAssetUrl(styleOption.thumbnail) }}
                        style={styles.accessoryThumb}
                        contentFit="contain"
                      />
                    </View>
                    <ThemedText style={[styles.accessoryModelLabel, { color: colors.text }]}>
                      {styleOption.label}
                    </ThemedText>
                    {isActive ? (
                      <View style={[styles.accessorySelectedDot, { backgroundColor: Colors[colorScheme].tint }]} />
                    ) : null}
                  </Pressable>
                );
              })}
            </View>

            <View style={styles.accessoryLiveStatus}>
              <View style={{ flex: 1 }}>
                <ThemedText style={styles.accessoryLiveTitle}>Canli onizleme</ThemedText>
                <ThemedText style={styles.helperText}>
                  {accessoryEnabled ? 'Model veya slider degisince sonuc otomatik yenilenir.' : 'Bir model secince onizleme otomatik olusur.'}
                </ThemedText>
              </View>
              {accessoryLoading ? <ActivityIndicator /> : <Ionicons name="sparkles-outline" size={18} color={colors.tint} />}
            </View>

            <ThemedText style={styles.helperText}>Gorunurluk: {Math.round(accessoryIntensity * 100)}%</ThemedText>
            <Slider
              style={styles.nativeSlider}
              minimumValue={0}
              maximumValue={1}
              step={0.01}
              value={accessoryIntensity}
              onValueChange={setAccessoryIntensity}
              minimumTrackTintColor={colors.tint}
              maximumTrackTintColor="rgba(120,120,120,0.25)"
              thumbTintColor={colors.tint}
              disabled={!preprocessedB64}
            />

            <ThemedText style={styles.helperText}>Boyut: {accessoryScale.toFixed(2)}x</ThemedText>
            <Slider
              style={styles.nativeSlider}
              minimumValue={0.6}
              maximumValue={1.6}
              step={0.01}
              value={accessoryScale}
              onValueChange={setAccessoryScale}
              minimumTrackTintColor={colors.tint}
              maximumTrackTintColor="rgba(120,120,120,0.25)"
              thumbTintColor={colors.tint}
              disabled={!preprocessedB64}
            />

            <ThemedText style={styles.helperText}>X Kaydirma: {Math.round(accessoryOffsetX)} px</ThemedText>
            <Slider
              style={styles.nativeSlider}
              minimumValue={-80}
              maximumValue={80}
              step={1}
              value={accessoryOffsetX}
              onValueChange={setAccessoryOffsetX}
              minimumTrackTintColor={colors.tint}
              maximumTrackTintColor="rgba(120,120,120,0.25)"
              thumbTintColor={colors.tint}
              disabled={!preprocessedB64}
            />

            <ThemedText style={styles.helperText}>Y Kaydirma: {Math.round(accessoryOffsetY)} px</ThemedText>
            <Slider
              style={styles.nativeSlider}
              minimumValue={-80}
              maximumValue={80}
              step={1}
              value={accessoryOffsetY}
              onValueChange={setAccessoryOffsetY}
              minimumTrackTintColor={colors.tint}
              maximumTrackTintColor="rgba(120,120,120,0.25)"
              thumbTintColor={colors.tint}
              disabled={!preprocessedB64}
            />

            {accessoryError ? <Text style={styles.errorText}>{accessoryError}</Text> : null}
              </>
            )}

            {activeTab === 'accessory2' && (
              <>
                <View style={styles.sectionHeader}>
                  <View style={[styles.stepBadge, { backgroundColor: accent }]}><Text style={styles.stepBadgeText}>5</Text></View>
                  <ThemedText type="defaultSemiBold">3D Aksesuarlar (Aksesuar 2)</ThemedText>
                </View>
                <ThemedText style={styles.helperText}>
                  Gerçek zamanlı 3D aksesuarları doğrudan fotoğrafa ekleyin.
                </ThemedText>

                {/* Categories */}
                <ScrollView horizontal showsHorizontalScrollIndicator={false} style={{ marginVertical: 12 }} contentContainerStyle={{ gap: 8 }}>
                  {(['glasses', 'hat', 'mask', 'earrings', 'necklace', 'tie'] as const).map((cat) => {
                    const isActive = activeCategory2 === cat;
                    const labels = {
                      glasses: 'Gözlük',
                      hat: 'Şapka',
                      mask: 'Maske',
                      earrings: 'Küpe',
                      necklace: 'Kolye',
                      tie: 'Kravat & Papyon'
                    };
                    return (
                      <Pressable
                        key={cat}
                        onPress={() => setActiveCategory2(cat)}
                        style={[
                          styles.warpOpButton,
                          {
                            backgroundColor: isActive ? Colors[colorScheme].tint : 'rgba(120,120,120,0.15)',
                            paddingHorizontal: 12,
                            paddingVertical: 8,
                            borderRadius: 8,
                          }
                        ]}>
                        <ThemedText style={{ color: isActive ? tintTextColor : colors.text, fontSize: 13, fontWeight: '600' }}>
                          {labels[cat]}
                        </ThemedText>
                      </Pressable>
                    );
                  })}
                </ScrollView>

                {/* Category Toggle Switch */}
                <View style={{ flexDirection: 'row', alignItems: 'center', justifyContent: 'space-between', marginVertical: 8, backgroundColor: softSurface, padding: 12, borderRadius: 10 }}>
                  <ThemedText style={{ fontSize: 14, fontWeight: '600' }}>Bu Kategoriyi Aktifleştir</ThemedText>
                  <Switch
                    value={accessories2[activeCategory2]}
                    onValueChange={(val) => setAccessories2(prev => ({ ...prev, [activeCategory2]: val }))}
                    trackColor={{ true: Colors[colorScheme].tint }}
                  />
                </View>

                {/* Styles List */}
                <ScrollView
                  style={{ maxHeight: 300, marginTop: 8 }}
                  contentContainerStyle={styles.variantsGrid}
                  showsVerticalScrollIndicator={true}
                  nestedScrollEnabled={true}>

                  {activeCategory2 === 'glasses' && GLASSES_VARIANTS.map(({ key, label }) => {
                    const isSelected = accessories2.glasses && glassesStyle2 === key;
                    const cacheKey = `glasses_${key}`;
                    const thumbUrl = thumbnails2[cacheKey];
                    return (
                      <Pressable key={key} onPress={() => {
                        setAccessories2(prev => ({ ...prev, glasses: true }));
                        setGlassesStyle2(key);
                      }}
                        style={[styles.variantButton, {
                          backgroundColor: isSelected ? 'rgba(160,32,240,0.30)' : softSurface,
                          borderColor: isSelected ? accent : panelBorder
                        }]}>
                        {thumbUrl ? (
                          <Image source={{ uri: thumbUrl }} style={styles.variantImage} contentFit="contain" />
                        ) : (
                          <View style={styles.variantImagePlaceholder} />
                        )}
                        <Text style={[styles.variantLabel, { color: colors.text }]} numberOfLines={1}>{label}</Text>
                      </Pressable>
                    );
                  })}

                  {activeCategory2 === 'hat' && HAT_VARIANTS.map(({ key, label }) => {
                    const isSelected = accessories2.hat && hatStyle2 === key;
                    const cacheKey = `hat_${key}`;
                    const thumbUrl = thumbnails2[cacheKey];
                    return (
                      <Pressable key={key} onPress={() => {
                        setAccessories2(prev => ({ ...prev, hat: true }));
                        setHatStyle2(key);
                      }}
                        style={[styles.variantButton, {
                          backgroundColor: isSelected ? 'rgba(160,32,240,0.30)' : softSurface,
                          borderColor: isSelected ? accent : panelBorder
                        }]}>
                        {thumbUrl ? (
                          <Image source={{ uri: thumbUrl }} style={styles.variantImage} contentFit="contain" />
                        ) : (
                          <View style={styles.variantImagePlaceholder} />
                        )}
                        <Text style={[styles.variantLabel, { color: colors.text }]} numberOfLines={1}>{label}</Text>
                      </Pressable>
                    );
                  })}

                  {activeCategory2 === 'mask' && MASK_VARIANTS.map(({ key, label }) => {
                    const isSelected = accessories2.mask && maskStyle2 === key;
                    const cacheKey = `mask_${key}`;
                    const thumbUrl = thumbnails2[cacheKey];
                    return (
                      <Pressable key={key} onPress={() => {
                        setAccessories2(prev => ({ ...prev, mask: true }));
                        setMaskStyle2(key);
                      }}
                        style={[styles.variantButton, {
                          backgroundColor: isSelected ? 'rgba(160,32,240,0.30)' : softSurface,
                          borderColor: isSelected ? accent : panelBorder
                        }]}>
                        {thumbUrl ? (
                          <Image source={{ uri: thumbUrl }} style={styles.variantImage} contentFit="contain" />
                        ) : (
                          <View style={styles.variantImagePlaceholder} />
                        )}
                        <Text style={[styles.variantLabel, { color: colors.text }]} numberOfLines={1}>{label}</Text>
                      </Pressable>
                    );
                  })}

                  {activeCategory2 === 'earrings' && EARRING_VARIANTS.map(({ key, label }) => {
                    const isSelected = accessories2.earrings && earringStyle2 === key;
                    const cacheKey = `earrings_${key}`;
                    const thumbUrl = thumbnails2[cacheKey];
                    return (
                      <Pressable key={key} onPress={() => {
                        setAccessories2(prev => ({ ...prev, earrings: true }));
                        setEarringStyle2(key);
                      }}
                        style={[styles.variantButton, {
                          backgroundColor: isSelected ? 'rgba(160,32,240,0.30)' : softSurface,
                          borderColor: isSelected ? accent : panelBorder
                        }]}>
                        {thumbUrl ? (
                          <Image source={{ uri: thumbUrl }} style={styles.variantImage} contentFit="contain" />
                        ) : (
                          <View style={styles.variantImagePlaceholder} />
                        )}
                        <Text style={[styles.variantLabel, { color: colors.text }]} numberOfLines={1}>{label}</Text>
                      </Pressable>
                    );
                  })}

                  {activeCategory2 === 'necklace' && NECKLACE_VARIANTS.map(({ key, label }) => {
                    const isSelected = accessories2.necklace && necklaceStyle2 === key;
                    const cacheKey = `necklace_${key}`;
                    const thumbUrl = thumbnails2[cacheKey];
                    return (
                      <Pressable key={key} onPress={() => {
                        setAccessories2(prev => ({ ...prev, necklace: true }));
                        setNecklaceStyle2(key);
                      }}
                        style={[styles.variantButton, {
                          backgroundColor: isSelected ? 'rgba(160,32,240,0.30)' : softSurface,
                          borderColor: isSelected ? accent : panelBorder
                        }]}>
                        {thumbUrl ? (
                          <Image source={{ uri: thumbUrl }} style={styles.variantImage} contentFit="contain" />
                        ) : (
                          <View style={styles.variantImagePlaceholder} />
                        )}
                        <Text style={[styles.variantLabel, { color: colors.text }]} numberOfLines={1}>{label}</Text>
                      </Pressable>
                    );
                  })}

                  {activeCategory2 === 'tie' && TIE_VARIANTS.map(({ key, label }) => {
                    const isSelected = accessories2.tie && tieStyle2 === key;
                    const cacheKey = `tie_${key}`;
                    const thumbUrl = thumbnails2[cacheKey];
                    return (
                      <Pressable key={key} onPress={() => {
                        setAccessories2(prev => ({ ...prev, tie: true }));
                        setTieStyle2(key);
                      }}
                        style={[styles.variantButton, {
                          backgroundColor: isSelected ? 'rgba(160,32,240,0.30)' : softSurface,
                          borderColor: isSelected ? accent : panelBorder
                        }]}>
                        {thumbUrl ? (
                          <Image source={{ uri: thumbUrl }} style={styles.variantImage} contentFit="contain" />
                        ) : (
                          <View style={styles.variantImagePlaceholder} />
                        )}
                        <Text style={[styles.variantLabel, { color: colors.text }]} numberOfLines={1}>{label}</Text>
                      </Pressable>
                    );
                  })}
                </ScrollView>
              </>
            )}

            {activeTab === 'makeup' && (
              <>
            {/* Section 5: Hair Color */}
            <View style={[styles.sectionHeader, { marginTop: 0 }]}>
              <View style={[styles.stepBadge, { backgroundColor: '#7C3AED' }]}>
                <Text style={styles.stepBadgeText}>5</Text>
              </View>
              <ThemedText type="defaultSemiBold">Saç Rengi</ThemedText>
            </View>
            <ThemedText style={styles.helperText}>
              Doğal veya canlı bir renk seç, yoğunluğu ayarla ve uygula. Tüm doku ve parıltı korunur.
            </ThemedText>

            {/* Natural color presets */}
            <ThemedText style={[styles.sideLabel, { marginBottom: 4 }]}>Doğal</ThemedText>
            <View style={styles.hairSwatchRow}>
              {([
                { hex: '#1a1a1a', label: 'Siyah' },
                { hex: '#3b1f0f', label: 'K. Kahve' },
                { hex: '#6b3a2a', label: 'Kahve' },
                { hex: '#8B5E3C', label: 'A. Kahve' },
                { hex: '#C19A6B', label: 'Kumral' },
                { hex: '#F0C040', label: 'Sarı' },
                { hex: '#F5EDD6', label: 'Platin' },
              ] as { hex: string; label: string }[]).map((item) => (
                <Pressable
                  key={item.hex}
                  onPress={() => setHairColorHex(item.hex)}
                  style={[
                    styles.hairSwatch,
                    { backgroundColor: item.hex },
                    hairColorHex.toUpperCase() === item.hex.toUpperCase() && styles.hairSwatchActive,
                  ]}
                />
              ))}
            </View>

            {/* Vivid color presets */}
            <ThemedText style={[styles.sideLabel, { marginBottom: 4, marginTop: 8 }]}>Canlı</ThemedText>
            <View style={styles.hairSwatchRow}>
              {([
                { hex: '#8B2500', label: 'Kızıl' },
                { hex: '#CC2200', label: 'Kırmızı' },
                { hex: '#B97333', label: 'Bakır' },
                { hex: '#B76E79', label: 'Rose Gold' },
                { hex: '#FF69B4', label: 'Pembe' },
                { hex: '#7B2D8B', label: 'Mor' },
                { hex: '#1565C0', label: 'Mavi' },
                { hex: '#00827F', label: 'Teal' },
              ] as { hex: string; label: string }[]).map((item) => (
                <Pressable
                  key={item.hex}
                  onPress={() => setHairColorHex(item.hex)}
                  style={[
                    styles.hairSwatch,
                    { backgroundColor: item.hex },
                    hairColorHex.toUpperCase() === item.hex.toUpperCase() && styles.hairSwatchActive,
                  ]}
                />
              ))}
            </View>

            {/* Custom hex input + preview dot */}
            <View style={styles.hairHexRow}>
              <View style={[styles.hairHexPreview, { backgroundColor: hairColorHex }]} />
              <TextInput
                style={[
                  styles.makeupHexInput,
                  {
                    color: colors.text,
                    borderColor: panelBorder,
                    flex: 1,
                  },
                ]}
                value={hairColorHex}
                onChangeText={(v) => setHairColorHex(v.startsWith('#') ? v : `#${v}`)}
                placeholder="#RRGGBB"
                placeholderTextColor={mutedText}
                maxLength={7}
                autoCapitalize="characters"
              />
            </View>

            {/* Intensity slider */}
            <View style={styles.sliderRow}>
              <ThemedText style={styles.sideLabel}>Yoğunluk</ThemedText>
              <ThemedText style={[styles.sideLabel, { color: accent }]}>
                {Math.round(hairColorIntensity * 100)}%
              </ThemedText>
            </View>
            <Slider
              value={hairColorIntensity}
              onValueChange={setHairColorIntensity}
              minimumValue={0.1}
              maximumValue={1}
              step={0.05}
              minimumTrackTintColor={accent}
              maximumTrackTintColor={colorScheme === 'dark' ? 'rgba(255,255,255,0.12)' : 'rgba(0,0,0,0.10)'}
              thumbTintColor={accent}
            />

            {/* Apply button */}
            <Pressable
              style={[
                styles.cvButton,
                {
                  backgroundColor: accent,
                  opacity: fullQualitySourceB64 && !hairColorLoading ? 1 : 0.45,
                },
              ]}
              onPress={runHairColor}
              disabled={!fullQualitySourceB64 || hairColorLoading}>
              {hairColorLoading
                ? <ActivityIndicator color="#fff" />
                : <ThemedText style={[styles.cvButtonText, { color: '#fff' }]}>Rengi Uygula</ThemedText>}
            </Pressable>

            {hairColorError ? (
              <Text style={styles.errorText}>{hairColorError}</Text>
            ) : null}

            {/* Divider */}
            <View style={[styles.sectionDivider, { marginVertical: 16 }]} />

            {/* Section 6: Manual Makeup */}
            <View style={[styles.sectionHeader, { marginTop: 0 }]}>
              <View style={[styles.stepBadge, { backgroundColor: accent }]}>
                <Text style={styles.stepBadgeText}>6</Text>
              </View>
              <ThemedText type="defaultSemiBold">Manual Makeup</ThemedText>
            </View>
            {/* Section 6: Manual Makeup */}
            <ThemedText type="defaultSemiBold" style={{ display: 'none' }}>6. Manual Makeup</ThemedText>
            <ThemedText style={styles.helperText}>
              Bölge seç, HEX renk gir ve yoğunluğu ayarla. Uygula ile seçili bölgeyi landmark tabanlı renklendirir.
            </ThemedText>

            <View style={styles.warpGrid}>
              {MANUAL_MAKEUP_PRESETS.map((preset) => {
                const isActive = makeupTarget === preset.key;
                return (
                  <Pressable
                    key={preset.key}
                    style={[
                      styles.warpOpButton,
                      {
                        backgroundColor: isActive ? Colors[colorScheme].tint : 'rgba(120,120,120,0.15)',
                        opacity: preprocessedB64 ? 1 : 0.4,
                      },
                    ]}
                    onPress={() => {
                      setMakeupTarget(preset.key);
                      setMakeupHexColor(preset.defaultColor);
                    }}
                    disabled={!preprocessedB64}>
                    <ThemedText style={[styles.warpOpText, { color: isActive ? tintTextColor : colors.text }]}>
                      {preset.label}
                    </ThemedText>
                  </Pressable>
                );
              })}
            </View>

            <View style={styles.makeupColorRow}>
              <View style={styles.makeupColorPreview}>
                <View
                  style={[
                    styles.makeupColorSwatch,
                    {
                      backgroundColor: normalizeHexColor(
                        makeupHexColor,
                        MANUAL_MAKEUP_PRESETS.find((item) => item.key === makeupTarget)?.defaultColor ?? '#FFFFFF'
                      ),
                    },
                  ]}
                />
              </View>
              <TextInput
                value={makeupHexColor}
                onChangeText={setMakeupHexColor}
                placeholder="#D45A73"
                autoCapitalize="characters"
                autoCorrect={false}
                maxLength={7}
                style={[
                  styles.makeupHexInput,
                  {
                    borderColor: normalizeHexColor(
                      makeupHexColor,
                      MANUAL_MAKEUP_PRESETS.find((item) => item.key === makeupTarget)?.defaultColor ?? '#FFFFFF'
                    ),
                    color: colors.text,
                  },
                ]}
                editable={!!preprocessedB64}
              />
            </View>

            <View style={styles.makeupSwatchGrid}>
              {(MANUAL_MAKEUP_SWATCHES[makeupTarget] ?? []).map((swatch) => {
                const isSelected = normalizeHexColor(makeupHexColor, swatch) === swatch.toUpperCase();
                return (
                  <Pressable
                    key={swatch}
                    style={[
                      styles.makeupSwatch,
                      {
                        backgroundColor: swatch,
                        borderColor: isSelected ? Colors[colorScheme].tint : 'rgba(120,120,120,0.2)',
                      },
                    ]}
                    onPress={() => setMakeupHexColor(swatch)}
                    disabled={!preprocessedB64}>
                    <ThemedText style={[styles.makeupSwatchLabel, { color: getContrastTextColor(swatch) }]}>
                      {swatch}
                    </ThemedText>
                  </Pressable>
                );
              })}
            </View>

            <ThemedText style={styles.helperText}>Intensity: {Math.round(makeupIntensity * 100)}%</ThemedText>
            <Slider
              style={styles.nativeSlider}
              minimumValue={0}
              maximumValue={1}
              step={0.01}
              value={makeupIntensity}
              onValueChange={setMakeupIntensity}
              minimumTrackTintColor={colors.tint}
              maximumTrackTintColor="rgba(120,120,120,0.25)"
              thumbTintColor={colors.tint}
              disabled={!preprocessedB64}
            />

            <Pressable
              style={[styles.cvButton, { backgroundColor: Colors[colorScheme].tint, opacity: preprocessedB64 ? 1 : 0.5 }]}
              onPress={applyMakeup}
              disabled={!preprocessedB64 || makeupLoading}>
              <ThemedText style={[styles.cvButtonText, { color: tintTextColor }]}>Uygula</ThemedText>
            </Pressable>

            {makeupLoading ? <ActivityIndicator /> : null}
            {makeupError ? <Text style={styles.errorText}>{makeupError}</Text> : null}

            {/* Makeup layers display */}
            {makeupLayers.length > 0 && (
              <View style={styles.makeupLayersContainer}>
                <ThemedText style={styles.makeupLayersTitle}>Makyaj Katmanları</ThemedText>
                {makeupLayers.map((layer, idx) => {
                  const label = MANUAL_MAKEUP_LABELS[Object.keys(MANUAL_MAKEUP_LABELS).find((key) => MANUAL_MAKEUP_LABELS[key as MakeupUiTarget] === layer.region) as MakeupUiTarget] || layer.region;
                  return (
                    <View key={layer.id} style={styles.makeupLayerRow}>
                      <Switch
                        value={layer.locked}
                        onValueChange={(newVal) => {
                          const updated = [...makeupLayers];
                          updated[idx].locked = newVal;
                          setMakeupLayers(updated);
                          if (newVal) {
                            setMakeupResultB64(layer.resultB64);
                            setMakeupPreviewKind('makeup');
                          } else {
                            // Rebuild from previous locked layer
                            const lastLockedIdx = updated.slice(0, idx).findLastIndex((l) => l.locked);
                            if (lastLockedIdx >= 0) {
                              setMakeupResultB64(updated[lastLockedIdx].resultB64);
                              setMakeupPreviewKind('makeup');
                            } else {
                              setMakeupResultB64(null);
                              setMakeupPreviewKind((current) => current === 'makeup' ? null : current);
                            }
                          }
                        }}
                      />
                      <View style={[styles.makeupLayerColor, { backgroundColor: layer.color }]} />
                      <ThemedText style={styles.makeupLayerLabel}>{label} {Math.round(layer.intensity * 100)}%</ThemedText>
                      <Pressable
                        onPress={() => {
                          void rebuildMakeupLayers(makeupLayers.filter((_, layerIdx) => layerIdx !== idx));
                        }}
                        style={styles.makeupLayerDeleteBtn}>
                        <Ionicons name="close-circle" size={20} color={colors.tint} />
                      </Pressable>
                    </View>
                  );
                })}
              </View>
            )}
              </>
            )}

            {evalMetrics ? (
              <View style={styles.metricsCard}>
                <View style={styles.metricsCardHeader}>
                  <ThemedText type="defaultSemiBold" style={styles.metricsCardTitle}>Quantitative Evaluation</ThemedText>
                  {evalSourceLabel ? (
                    <View style={styles.metricsSourcePill}>
                      <Text style={styles.metricsSourceText}>{evalSourceLabel}</Text>
                    </View>
                  ) : null}
                </View>

                <View style={styles.metricTableHeaderRow}>
                  <Text style={[styles.metricHeaderCell, styles.metricHeaderMetric]}>Metric</Text>
                  <Text style={[styles.metricHeaderCell, styles.metricHeaderValue]}>Value</Text>
                  <Text style={[styles.metricHeaderCell, styles.metricHeaderPurpose]}>Acceptable Range</Text>
                </View>

                {metricTableRows.map((row) => (
                  <View key={row.metric} style={styles.metricTableDataRow}>
                    <Text style={[styles.metricDataCell, styles.metricDataMetric]}>{row.metric}</Text>
                    <View style={[styles.metricDataValue, styles.metricValueCell]}>
                      <View style={[styles.metricStatusDot, { backgroundColor: METRIC_STATUS_COLOR[row.status] }]} />
                      <Text style={[styles.metricValueText, { color: METRIC_STATUS_COLOR[row.status] }]}>{row.value}</Text>
                    </View>
                    <Text style={[styles.metricDataCell, styles.metricDataPurpose]}>{row.purposeRange}</Text>
                  </View>
                ))}

                {proMetrics?.hf_lf_ratio_before != null ? (
                  <View style={styles.hfLfRow}>
                    <View style={styles.hfLfStatBox}>
                      <Text style={styles.hfLfLabel}>Before</Text>
                      <Text style={styles.hfLfValue}>{proMetrics.hf_lf_ratio_before.toFixed(4)}</Text>
                    </View>
                    {proMetrics.hf_lf_ratio_after != null ? (
                      <View style={styles.hfLfStatBox}>
                        <Text style={styles.hfLfLabel}>After</Text>
                        <Text style={styles.hfLfValue}>{proMetrics.hf_lf_ratio_after.toFixed(4)}</Text>
                      </View>
                    ) : null}
                    {proMetrics.hf_lf_ratio_delta != null ? (
                      <View style={[styles.hfLfStatBox, styles.hfLfDeltaBox]}>
                        <Text style={styles.hfLfLabel}>HF/LF Delta</Text>
                        <Text style={[styles.hfLfValue, {
                          color: proMetrics.hf_lf_ratio_delta < 0 ? '#22C55E' : proMetrics.hf_lf_ratio_delta > 0 ? '#F59E0B' : '#6B7280',
                        }]}>
                          {proMetrics.hf_lf_ratio_delta > 0 ? '+' : ''}{proMetrics.hf_lf_ratio_delta.toFixed(4)}
                        </Text>
                      </View>
                    ) : null}
                  </View>
                ) : null}
              </View>
            ) : null}

            {spectrumGrayB64 && spectrumBlueB64 && spectrumRedB64 ? (
              <View style={styles.sideBySide}>
                <View style={styles.sideBox}>
                  <ThemedText style={styles.sideLabel}>Spectrum (Gray)</ThemedText>
                  <Pressable onPress={() => setLightboxUri(`data:image/png;base64,${spectrumGrayB64}`)}>
                    <Image source={{ uri: `data:image/png;base64,${spectrumGrayB64}` }} style={styles.sideImage} contentFit="contain" />
                  </Pressable>
                </View>
                <View style={styles.sideBox}>
                  <ThemedText style={styles.sideLabel}>Spectrum (Blue)</ThemedText>
                  <Pressable onPress={() => setLightboxUri(`data:image/png;base64,${spectrumBlueB64}`)}>
                    <Image source={{ uri: `data:image/png;base64,${spectrumBlueB64}` }} style={styles.sideImage} contentFit="contain" />
                  </Pressable>
                </View>
                <View style={styles.sideBox}>
                  <ThemedText style={styles.sideLabel}>Spectrum (Red)</ThemedText>
                  <Pressable onPress={() => setLightboxUri(`data:image/png;base64,${spectrumRedB64}`)}>
                    <Image source={{ uri: `data:image/png;base64,${spectrumRedB64}` }} style={styles.sideImage} contentFit="contain" />
                  </Pressable>
                </View>
              </View>
            ) : null}

            <View style={styles.agingRow}>
              <Pressable
                style={[styles.cvButton, { flex: 1, backgroundColor: Colors[colorScheme].tint, opacity: evalMetrics ? 1 : 0.5 }]}
                onPress={exportCsv}
                disabled={!evalMetrics}>
                <ThemedText style={[styles.cvButtonText, { color: tintTextColor }]}>CSV Export</ThemedText>
              </Pressable>
              <Pressable
                style={[styles.cvButton, { flex: 1, backgroundColor: colors.text, opacity: evalMetrics ? 1 : 0.5 }]}
                onPress={exportPdf}
                disabled={!evalMetrics}>
                <ThemedText style={[styles.cvButtonText, { color: colorScheme === 'dark' ? '#000' : '#fff' }]}>PDF Export</ThemedText>
              </Pressable>
            </View>
            </ScrollView>
            </View>
          </View>
        </View>
        )}
      </View>
      </View>

      <Modal visible={!!lightboxUri} animationType="fade" transparent onRequestClose={() => { setLightboxUri(null); setLightboxCompareUri(null); }}>
        <View style={styles.lightboxBackdrop}>
          {lightboxUri ? (
            <View style={[styles.lightboxViewport, lightboxCompareUri ? { flexDirection: 'row', gap: 10, padding: 20 } : {}]} {...lightboxPanResponder.panHandlers}>
              
              {lightboxCompareUri ? (
                <View style={{ flex: 1, alignItems: 'center', justifyContent: 'center', width: '50%', height: '100%', position: 'relative' }}>
                  <View style={{ position: 'absolute', top: 30, backgroundColor: 'rgba(0,0,0,0.6)', paddingHorizontal: 12, paddingVertical: 6, borderRadius: 20, zIndex: 10 }}>
                    <Text style={{ color: '#fff', fontSize: 14, fontWeight: '700' }}>Önce (Orijinal)</Text>
                  </View>
                  <Image
                    source={{ uri: lightboxCompareUri }}
                    style={[
                      styles.lightboxImage,
                      {
                        transform: [
                          { translateX: lightboxOffset.x },
                          { translateY: lightboxOffset.y },
                          { scale: lightboxZoom },
                        ],
                      },
                    ]}
                    contentFit="contain"
                  />
                </View>
              ) : null}

              <View style={{ flex: 1, alignItems: 'center', justifyContent: 'center', width: lightboxCompareUri ? '50%' : '100%', height: '100%', position: 'relative' }}>
                {lightboxCompareUri ? (
                  <View style={{ position: 'absolute', top: 30, backgroundColor: 'rgba(160,32,240,0.8)', paddingHorizontal: 12, paddingVertical: 6, borderRadius: 20, zIndex: 10 }}>
                    <Text style={{ color: '#fff', fontSize: 14, fontWeight: '700' }}>Sonra (Efektli)</Text>
                  </View>
                ) : null}
                <Image
                  source={{ uri: lightboxUri }}
                  style={[
                    styles.lightboxImage,
                    {
                      transform: [
                        { translateX: lightboxOffset.x },
                        { translateY: lightboxOffset.y },
                        { scale: lightboxZoom },
                      ],
                    },
                  ]}
                  contentFit="contain"
                />
              </View>
              
            </View>
          ) : null}
          <View style={styles.lightboxControls}>
            <Pressable style={styles.lightboxToolButton} onPress={() => setLightboxZoomLevel(lightboxZoom - 0.25)}>
              <Ionicons name="remove" size={20} color="#fff" />
            </Pressable>
            <View style={styles.lightboxZoomPill}>
              <ThemedText style={styles.lightboxZoomText}>{Math.round(lightboxZoom * 100)}%</ThemedText>
            </View>
            <Pressable style={styles.lightboxToolButton} onPress={() => setLightboxZoomLevel(lightboxZoom + 0.25)}>
              <Ionicons name="add" size={20} color="#fff" />
            </Pressable>
            <Pressable
              style={styles.lightboxToolButton}
              onPress={() => {
                setLightboxZoom(1);
                setLightboxOffset({ x: 0, y: 0 });
              }}>
              <Ionicons name="scan-outline" size={18} color="#fff" />
            </Pressable>
          </View>
          <Pressable style={styles.lightboxClose} onPress={() => { setLightboxUri(null); setLightboxCompareUri(null); }}>
            <Ionicons name="close" size={24} color="#fff" />
          </Pressable>
        </View>
      </Modal>

      <Modal visible={cropEditorVisible} animationType="slide" transparent onRequestClose={closeCropEditor}>
        <View style={styles.cropModalBackdrop}>
          <View style={styles.cropModalScroll}>
            <View
              style={[
                styles.cropModalCard,
                {
                  backgroundColor: previewBackground,
                  borderColor: panelBorder,
                },
              ]}>
              <View style={styles.cropModalHeader}>
                <Pressable onPress={closeCropEditor} style={styles.cropModalHeaderButton}>
                  <Ionicons name="close" size={22} color="#FFFFFF" />
                </Pressable>

                <View style={styles.cropModalTitleWrap}>
                  <ThemedText type="subtitle" style={styles.cropModalTitle}>Kırpma Alanı</ThemedText>
                  <ThemedText style={styles.cropModalSubtitle}>Alanı sürükle, kenar veya köşelerden boyutlandır.</ThemedText>
                </View>

                <View style={styles.cropHeaderActions}>
                  <Pressable onPress={resetCropSelection} style={styles.cropToolbarButton}>
                    <Ionicons name="refresh" size={16} color="#FFFFFF" />
                    <ThemedText style={styles.cropToolbarText}>Sıfırla</ThemedText>
                  </Pressable>
                  <Pressable
                    onPress={applyCrop}
                    style={[styles.cropApplyButton, { opacity: cropBox ? 1 : 0.5 }]}
                    disabled={!cropBox}>
                    <ThemedText style={styles.cropApplyButtonText}>Uygula</ThemedText>
                  </Pressable>
                </View>
              </View>

              <View style={styles.cropPresetBar}>
                <Pressable style={styles.cropPresetButton} onPress={() => setCropToAspect('full')}>
                  <ThemedText style={styles.cropPresetText}>Tamamı</ThemedText>
                </Pressable>
                <Pressable style={styles.cropPresetButton} onPress={() => setCropToAspect(1)}>
                  <ThemedText style={styles.cropPresetText}>1:1</ThemedText>
                </Pressable>
                <Pressable style={styles.cropPresetButton} onPress={() => setCropToAspect(4 / 5)}>
                  <ThemedText style={styles.cropPresetText}>4:5</ThemedText>
                </Pressable>
                <Pressable style={styles.cropPresetButton} onPress={() => setCropToAspect(16 / 9)}>
                  <ThemedText style={styles.cropPresetText}>16:9</ThemedText>
                </Pressable>
                <Pressable style={styles.cropPresetButton} onPress={centerCropBox}>
                  <ThemedText style={styles.cropPresetText}>Ortala</ThemedText>
                </Pressable>
              </View>

              <View
                style={[styles.cropStage, { height: cropStageHeight }, WEB_NO_SELECT_STYLE]}
                onLayout={(event) => {
                  const { width: stageWidth, height: stageHeight } = event.nativeEvent.layout;
                  const nextLayout = { width: stageWidth, height: stageHeight };
                  setCropStageLayout(nextLayout);
                  if (!cropBoxRef.current && selectedImageSize) {
                    updateCropBox(createInitialCropBox(nextLayout, selectedImageSize));
                  }
                }}>
                {selectedImageUri ? (
                  <Image
                    pointerEvents="none"
                    source={{ uri: selectedImageUri }}
                    style={[styles.cropStageImage, WEB_NO_SELECT_STYLE]}
                    contentFit="contain"
                  />
                ) : null}

                {cropBox && cropStageLayout && selectedImageSize ? (
                  <View pointerEvents="box-none" style={StyleSheet.absoluteFill}>
                    <View pointerEvents="none" style={[styles.cropShadeOverlay, { left: 0, right: 0, top: 0, height: cropBox.y }]} />
                    <View pointerEvents="none" style={[styles.cropShadeOverlay, { left: 0, top: cropBox.y, width: cropBox.x, height: cropBox.height }]} />
                    <View
                      pointerEvents="none"
                      style={[
                        styles.cropShadeOverlay,
                        {
                          left: cropBox.x + cropBox.width,
                          right: 0,
                          top: cropBox.y,
                          height: cropBox.height,
                        },
                      ]}
                    />
                    <View pointerEvents="none" style={[styles.cropShadeOverlay, { left: 0, right: 0, top: cropBox.y + cropBox.height, bottom: 0 }]} />
                    <View
                      style={[
                        styles.cropBoxFrame,
                        {
                          left: cropBox.x,
                          top: cropBox.y,
                          width: cropBox.width,
                          height: cropBox.height,
                        },
                      ]}>
                      <View style={[styles.cropMoveSurface, WEB_NO_SELECT_STYLE]} {...moveCropResponder.panHandlers}>
                        <View pointerEvents="none" style={styles.cropBoxInnerBorder} />
                        <View pointerEvents="none" style={[styles.cropGridLine, styles.cropGridVerticalOne]} />
                        <View pointerEvents="none" style={[styles.cropGridLine, styles.cropGridVerticalTwo]} />
                        <View pointerEvents="none" style={[styles.cropGridLine, styles.cropGridHorizontalOne]} />
                        <View pointerEvents="none" style={[styles.cropGridLine, styles.cropGridHorizontalTwo]} />
                      </View>

                      <View style={[styles.cropEdgeHandle, styles.cropTopHandle, WEB_NO_SELECT_STYLE]} {...resizeTopResponder.panHandlers} />
                      <View style={[styles.cropEdgeHandle, styles.cropRightHandle, WEB_NO_SELECT_STYLE]} {...resizeRightResponder.panHandlers} />
                      <View style={[styles.cropEdgeHandle, styles.cropBottomHandle, WEB_NO_SELECT_STYLE]} {...resizeBottomResponder.panHandlers} />
                      <View style={[styles.cropEdgeHandle, styles.cropLeftHandle, WEB_NO_SELECT_STYLE]} {...resizeLeftResponder.panHandlers} />

                      <View style={[styles.cropCornerHandle, styles.cropTopLeftHandle, WEB_NO_SELECT_STYLE]} {...resizeTopLeftResponder.panHandlers} />
                      <View style={[styles.cropCornerHandle, styles.cropTopRightHandle, WEB_NO_SELECT_STYLE]} {...resizeTopRightResponder.panHandlers} />
                      <View style={[styles.cropCornerHandle, styles.cropBottomLeftHandle, WEB_NO_SELECT_STYLE]} {...resizeBottomLeftResponder.panHandlers} />
                      <View style={[styles.cropCornerHandle, styles.cropBottomRightHandle, WEB_NO_SELECT_STYLE]} {...resizeBottomRightResponder.panHandlers} />
                    </View>
                  </View>
                ) : null}
              </View>

              <View style={styles.cropModalFooter}>
                <ThemedText style={styles.cropFooterText}>İpucu: Kutunun içinden sürükleyerek taşıyabilir, kenarlardan yalnızca tek ekseni, köşelerden iki ekseni birlikte değiştirebilirsin.</ThemedText>
              </View>
            </View>
          </View>
        </View>
      </Modal>

      {Platform.OS === 'web' && activeTab === 'accessory2' && (
        <canvas
          ref={canvasRef2}
          width={preprocessMeta?.processedSize?.width ?? 512}
          height={preprocessMeta?.processedSize?.height ?? 512}
          style={{
            position: 'absolute',
            left: -9999,
            top: -9999,
            width: preprocessMeta?.processedSize?.width ?? 512,
            height: preprocessMeta?.processedSize?.height ?? 512,
          }}
        />
      )}
    </StudioScreen>
  );
}

const styles = StyleSheet.create({
  screen: {
    flex: 1,
    flexDirection: 'row',
  },
  mainContent: {
    flex: 1,
    height: '100%',
    overflow: 'hidden',
  },
  backButton: {
    width: 42,
    height: 42,
    borderRadius: 21,
    alignItems: 'center',
    justifyContent: 'center',
    backgroundColor: 'rgba(255,255,255,0.04)',
  },
  header: {
    paddingTop: 54,
    paddingHorizontal: 28,
    paddingBottom: 22,
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'space-between',
    gap: 16,
  },
  headerTitleGroup: {
    flex: 1,
    alignItems: 'flex-start',
    gap: 4,
  },
  headerTitle: {
    fontSize: 34,
    fontWeight: '900',
    letterSpacing: 0,
  },
  headerSubtitle: {
    fontSize: 13,
    fontWeight: '600',
  },
  headerActions: {
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'flex-end',
    gap: 8,
  },
  topActionButton: {
    minHeight: 34,
    borderRadius: 999,
    borderWidth: 1,
    paddingHorizontal: 12,
    flexDirection: 'row',
    alignItems: 'center',
    gap: 7,
  },
  topActionPrimary: {
    backgroundColor: '#F0F0F2',
    borderColor: '#F0F0F2',
  },
  topActionText: {
    color: '#FFFFFF',
    fontSize: 14,
    fontWeight: '900',
  },
  topActionPrimaryText: {
    color: '#000000',
    fontSize: 14,
    fontWeight: '900',
  },
  content: {
    flex: 1,
    paddingHorizontal: 28,
    paddingBottom: 28,
    paddingTop: 0,
    overflow: 'hidden',
  },
  workspace: {
    flex: 1,
    gap: 18,
    width: '100%',
    alignSelf: 'center',
    maxWidth: '100%',
    overflow: 'hidden',
  },
  liveWrap: {
    flex: 1,
    width: '100%',
    alignSelf: 'center',
    maxWidth: '100%',
    overflow: 'hidden',
  },
  modeToggle: {
    flexDirection: 'row',
    alignSelf: 'flex-start',
    marginTop: 10,
    padding: 4,
    borderRadius: 999,
    borderWidth: 1,
    gap: 4,
  },
  modeOption: {
    flexDirection: 'row',
    alignItems: 'center',
    gap: 6,
    paddingHorizontal: 12,
    paddingVertical: 7,
    borderRadius: 999,
  },
  modeOptionText: {
    fontSize: 12,
    fontWeight: '700',
  },
  modeOptionDot: {
    width: 6,
    height: 6,
    borderRadius: 3,
    backgroundColor: '#ef4444',
    marginLeft: 2,
  },
  workspaceWide: {
    flexDirection: 'row',
    alignItems: 'stretch',
    justifyContent: 'center',
  },
  panel: {
    borderWidth: 1,
    borderRadius: 28,
    padding: 18,
    gap: 16,
    shadowColor: '#000',
    shadowOpacity: 0.18,
    shadowRadius: 28,
    shadowOffset: { width: 0, height: 18 },
  },
  uploadPanel: {
    flex: 0.92,
    minWidth: 280,
  },
  previewPanel: {
    flex: 1.22,
    minWidth: 360,
    alignItems: 'stretch',
  },
  featurePanel: {
    flex: 1.04,
    minWidth: 300,
    height: '100%',
    maxHeight: '100%' as any,
    overflow: 'hidden',
  },
  featurePanelScroller: {
    flex: 1,
    ...(Platform.OS === 'web' ? ({ overflowY: 'auto' } as any) : null),
  },
  featurePanelContent: {
    gap: 16,
    paddingRight: 4,
  },
  panelTitleRow: {
    flexDirection: 'row',
    alignItems: 'center',
    gap: 9,
  },
  panelTitleDot: {
    width: 7,
    height: 7,
    borderRadius: 4,
  },
  panelTitle: {
    fontWeight: '900',
    letterSpacing: 0,
  },
  helperText: {
    opacity: 0.72,
    lineHeight: 21,
  },
  uploadDropzone: {
    width: '100%',
    alignSelf: 'stretch',
    minHeight: 176,
    borderRadius: 24,
    borderWidth: 1,
    borderStyle: 'dashed',
    alignItems: 'center',
    justifyContent: 'center',
    gap: 10,
    padding: 18,
  },
  uploadDropzoneSmall: {
    minHeight: 44,
    borderRadius: 12,
    borderWidth: 1,
    borderStyle: 'solid',
    alignItems: 'center',
    justifyContent: 'center',
    gap: 6,
    paddingHorizontal: 12,
    paddingVertical: 8,
    flexDirection: 'row',
  },
  uploadIconBubble: {
    width: 58,
    height: 58,
    borderRadius: 18,
    alignItems: 'center',
    justifyContent: 'center',
  },
  uploadDropzoneTitle: {
    fontSize: 15,
    fontWeight: '900',
  },
  uploadDropzoneTitleSmall: {
    fontSize: 12,
    fontWeight: '700',
  },
  uploadDropzoneHint: {
    fontSize: 12,
    fontWeight: '600',
  },
  uploadButton: {
    height: 44,
    borderRadius: 22,
    paddingHorizontal: 18,
    alignSelf: 'flex-start',
    flexDirection: 'row',
    alignItems: 'center',
    gap: 8,
  },
  uploadButtonText: {
    color: '#FFFFFF',
    fontWeight: '700',
  },
  statusRow: {
    flexDirection: 'row',
    alignItems: 'flex-start',
    gap: 8,
  },
  statusText: {
    flex: 1,
    flexWrap: 'wrap',
    opacity: 0.9,
  },
  fileCard: {
    borderRadius: 14,
    padding: 14,
    backgroundColor: 'rgba(120,120,120,0.08)',
    gap: 6,
  },
  fileText: {
    opacity: 0.8,
  },
  previewBox: {
    width: '100%',
    maxWidth: 520,
    aspectRatio: 1,
    borderRadius: 24,
    overflow: 'hidden',
    borderWidth: 1,
    borderColor: 'rgba(120,120,120,0.18)',
    alignSelf: 'center',
    position: 'relative',
  },
  previewImage: {
    width: '100%',
    height: '100%',
  },
  previewBadge: {
    position: 'absolute',
    left: 12,
    top: 12,
    backgroundColor: 'rgba(80,82,84,0.72)',
    borderRadius: 8,
    paddingHorizontal: 11,
    paddingVertical: 8,
  },
  previewBadgeText: {
    color: '#FFFFFF',
    fontSize: 12,
    fontWeight: '700',
  },
  landmarkOverlayDot: {
    position: 'absolute',
    width: 4,
    height: 4,
    marginLeft: -2,
    marginTop: -2,
    borderRadius: 2,
    backgroundColor: '#00E676',
    borderWidth: 1,
    borderColor: 'rgba(0,96,48,0.42)',
  },
  emptyPreview: {
    width: '100%',
    maxWidth: 520,
    aspectRatio: 1,
    borderRadius: 24,
    borderWidth: 1,
    borderColor: 'rgba(120,120,120,0.18)',
    alignItems: 'center',
    justifyContent: 'center',
    gap: 10,
  },
  previewActions: {
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'space-between',
    gap: 10,
    width: '100%',
  },
  canvasHeader: {
    width: '100%',
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'space-between',
    gap: 12,
  },
  liveBadge: {
    borderRadius: 999,
    paddingHorizontal: 12,
    paddingVertical: 7,
  },
  liveBadgeText: {
    fontSize: 10,
    fontWeight: '900',
    textTransform: 'uppercase',
  },
  iconActionButton: {
    minHeight: 40,
    borderWidth: 1,
    borderRadius: 999,
    paddingHorizontal: 16,
    flexDirection: 'row',
    alignItems: 'center',
    gap: 8,
  },
  iconActionText: {
    fontSize: 13,
    fontWeight: '700',
  },
  cropHintPill: {
    flexShrink: 1,
    paddingHorizontal: 12,
    paddingVertical: 8,
    borderRadius: 999,
    backgroundColor: 'rgba(120,120,120,0.12)',
  },
  cropHintText: {
    fontSize: 12,
    fontWeight: '600',
  },
  emptyFeatureSpace: {
    flex: 1,
    minHeight: 180,
    borderRadius: 14,
    borderWidth: 1,
    borderStyle: 'dashed',
    borderColor: 'rgba(120,120,120,0.24)',
    alignItems: 'center',
    justifyContent: 'center',
    padding: 16,
  },
  featureHeader: {
    marginHorizontal: -18,
    marginTop: -18,
    paddingHorizontal: 18,
    paddingVertical: 18,
    borderTopLeftRadius: 28,
    borderTopRightRadius: 28,
    borderBottomWidth: 1,
    borderBottomColor: 'rgba(255,255,255,0.07)',
    backgroundColor: 'rgba(139,92,246,0.10)',
    gap: 3,
  },
  featureHeaderSub: {
    fontSize: 11,
    fontWeight: '900',
    textTransform: 'uppercase',
  },
  sectionHeader: {
    flexDirection: 'row',
    alignItems: 'center',
    gap: 10,
  },
  sectionDivider: {
    height: 1,
    backgroundColor: 'rgba(255,255,255,0.07)',
    marginTop: 6,
  },
  hairSwatchRow: {
    flexDirection: 'row',
    flexWrap: 'wrap',
    gap: 8,
    marginBottom: 4,
  },
  hairSwatch: {
    width: 30,
    height: 30,
    borderRadius: 15,
    borderWidth: 2,
    borderColor: 'transparent',
  },
  hairSwatchActive: {
    borderColor: '#fff',
    transform: [{ scale: 1.15 }],
  },
  hairHexRow: {
    flexDirection: 'row',
    alignItems: 'center',
    gap: 10,
    marginTop: 10,
    marginBottom: 4,
  },
  hairHexPreview: {
    width: 34,
    height: 34,
    borderRadius: 17,
    borderWidth: 1,
    borderColor: 'rgba(255,255,255,0.2)',
  },
  stepBadge: {
    minWidth: 30,
    height: 30,
    borderRadius: 10,
    paddingHorizontal: 6,
    alignItems: 'center',
    justifyContent: 'center',
  },
  stepBadgeText: {
    color: '#FFFFFF',
    fontSize: 11,
    fontWeight: '900',
  },
  cropModalBackdrop: {
    flex: 1,
    alignItems: 'center',
    justifyContent: 'center',
    padding: 24,
    backgroundColor: 'rgba(0,0,0,0.72)',
  },
  cropModalScroll: {
    width: '100%',
    alignItems: 'center',
    justifyContent: 'center',
  },
  cropModalCard: {
    width: '100%',
    maxWidth: 980,
    borderWidth: 1,
    borderRadius: 28,
    overflow: 'hidden',
    alignSelf: 'center',
    maxHeight: '96%',
  },
  cropModalHeader: {
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'space-between',
    gap: 12,
    paddingHorizontal: 18,
    paddingVertical: 14,
    borderBottomWidth: 1,
    borderBottomColor: 'rgba(255,255,255,0.10)',
  },
  cropModalHeaderButton: {
    width: 38,
    height: 38,
    borderRadius: 19,
    alignItems: 'center',
    justifyContent: 'center',
    backgroundColor: 'rgba(255,255,255,0.08)',
  },
  cropModalTitleWrap: {
    flex: 1,
    alignItems: 'center',
    gap: 2,
  },
  cropModalTitle: {
    color: '#FFFFFF',
    fontWeight: '900',
  },
  cropModalSubtitle: {
    color: 'rgba(255,255,255,0.58)',
    fontSize: 12,
    fontWeight: '600',
    textAlign: 'center',
  },
  cropHeaderActions: {
    flexDirection: 'row',
    alignItems: 'center',
    gap: 8,
  },
  cropToolbarButton: {
    height: 38,
    borderRadius: 19,
    paddingHorizontal: 14,
    borderWidth: 1,
    borderColor: 'rgba(255,255,255,0.12)',
    backgroundColor: 'rgba(255,255,255,0.08)',
    flexDirection: 'row',
    alignItems: 'center',
    gap: 7,
  },
  cropToolbarText: {
    color: '#FFFFFF',
    fontSize: 12,
    fontWeight: '800',
  },
  cropApplyButton: {
    minWidth: 104,
    height: 38,
    borderRadius: 19,
    alignItems: 'center',
    justifyContent: 'center',
    paddingHorizontal: 18,
    backgroundColor: '#A020F0',
  },
  cropApplyButtonText: {
    color: '#FFFFFF',
    fontWeight: '900',
    fontSize: 13,
  },
  cropPresetBar: {
    flexDirection: 'row',
    flexWrap: 'wrap',
    gap: 8,
    paddingHorizontal: 18,
    paddingVertical: 12,
    borderBottomWidth: 1,
    borderBottomColor: 'rgba(255,255,255,0.08)',
  },
  cropPresetButton: {
    height: 34,
    borderRadius: 17,
    paddingHorizontal: 14,
    alignItems: 'center',
    justifyContent: 'center',
    backgroundColor: 'rgba(255,255,255,0.08)',
    borderWidth: 1,
    borderColor: 'rgba(255,255,255,0.08)',
  },
  cropPresetText: {
    color: '#FFFFFF',
    fontSize: 12,
    fontWeight: '800',
  },
  cropStage: {
    width: '100%',
    height: 560,
    maxHeight: 560,
    backgroundColor: '#0B0D0F',
    position: 'relative',
    overflow: 'hidden',
  },
  cropStageImage: {
    ...StyleSheet.absoluteFillObject,
  },
  cropShadeOverlay: {
    position: 'absolute',
    backgroundColor: 'rgba(0,0,0,0.58)',
  },
  cropBoxFrame: {
    position: 'absolute',
    borderRadius: 16,
    borderWidth: 2,
    borderColor: '#28D4FF',
    backgroundColor: 'transparent',
    zIndex: 2,
  },
  cropMoveSurface: {
    ...StyleSheet.absoluteFillObject,
    borderRadius: 14,
    zIndex: 1,
  },
  cropBoxInnerBorder: {
    ...StyleSheet.absoluteFillObject,
    borderRadius: 14,
    borderWidth: 1,
    borderColor: 'rgba(255,255,255,0.9)',
  },
  cropGridLine: {
    position: 'absolute',
    backgroundColor: 'rgba(255,255,255,0.44)',
  },
  cropGridVerticalOne: {
    left: '33.333%',
    top: 0,
    bottom: 0,
    width: 1,
  },
  cropGridVerticalTwo: {
    left: '66.666%',
    top: 0,
    bottom: 0,
    width: 1,
  },
  cropGridHorizontalOne: {
    top: '33.333%',
    left: 0,
    right: 0,
    height: 1,
  },
  cropGridHorizontalTwo: {
    top: '66.666%',
    left: 0,
    right: 0,
    height: 1,
  },
  cropEdgeHandle: {
    position: 'absolute',
    backgroundColor: '#0891B2',
    borderWidth: 2,
    borderColor: '#FFFFFF',
    zIndex: 4,
  },
  cropTopHandle: {
    top: -13,
    left: '50%',
    width: 96,
    height: 26,
    marginLeft: -48,
    borderRadius: 13,
  },
  cropRightHandle: {
    right: -13,
    top: '50%',
    width: 26,
    height: 96,
    marginTop: -48,
    borderRadius: 13,
  },
  cropBottomHandle: {
    bottom: -13,
    left: '50%',
    width: 96,
    height: 26,
    marginLeft: -48,
    borderRadius: 13,
  },
  cropLeftHandle: {
    left: -13,
    top: '50%',
    width: 26,
    height: 96,
    marginTop: -48,
    borderRadius: 13,
  },
  cropCornerHandle: {
    position: 'absolute',
    width: 44,
    height: 44,
    borderRadius: 22,
    borderWidth: 3,
    borderColor: '#FFFFFF',
    backgroundColor: '#0891B2',
    zIndex: 5,
  },
  cropTopLeftHandle: {
    left: -21,
    top: -21,
  },
  cropTopRightHandle: {
    right: -21,
    top: -21,
  },
  cropBottomLeftHandle: {
    left: -21,
    bottom: -21,
  },
  cropBottomRightHandle: {
    right: -21,
    bottom: -21,
  },
  cropModalFooter: {
    paddingHorizontal: 18,
    paddingVertical: 14,
    borderTopWidth: 1,
    borderTopColor: 'rgba(255,255,255,0.08)',
  },
  cropFooterText: {
    color: 'rgba(255,255,255,0.70)',
    fontSize: 13,
    lineHeight: 18,
  },
  cvButton: {
    height: 46,
    borderRadius: 23,
    alignItems: 'center',
    justifyContent: 'center',
    paddingHorizontal: 16,
    shadowColor: '#000',
    shadowOpacity: 0.12,
    shadowRadius: 12,
    shadowOffset: { width: 0, height: 8 },
  },
  cvButtonText: {
    color: '#fff',
    fontWeight: '700',
    fontSize: 13,
  },
  errorText: {
    color: '#E53E3E',
    fontSize: 12,
    marginTop: 2,
  },
  resultImage: {
    width: '100%',
    height: 160,
    borderRadius: 10,
    backgroundColor: 'rgba(120,120,120,0.1)',
  },
  landmarkRow: {
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'space-between',
    gap: 10,
  },
  landmarkToggleWrap: {
    flexDirection: 'row',
    alignItems: 'center',
    gap: 8,
  },
  landmarkToggleLabel: {
    fontSize: 12,
    fontWeight: '700',
  },
  landmarkToggleTrack: {
    width: 44,
    height: 26,
    borderRadius: 999,
    borderWidth: 1,
    padding: 2,
    justifyContent: 'center',
  },
  landmarkToggleThumb: {
    width: 20,
    height: 20,
    borderRadius: 10,
    shadowColor: '#000',
    shadowOpacity: 0.16,
    shadowRadius: 5,
    shadowOffset: { width: 0, height: 2 },
  },
  warpGrid: {
    flexDirection: 'row',
    flexWrap: 'wrap',
    gap: 8,
  },
  warpOpButton: {
    flex: 1,
    minWidth: '40%',
    minHeight: 58,
    paddingVertical: 10,
    paddingHorizontal: 36,
    borderRadius: 12,
    alignItems: 'center',
    justifyContent: 'center',
    position: 'relative',
  },
  warpOpText: {
    fontSize: 12,
    lineHeight: 14,
    fontWeight: '600',
  },
  labOperationLabelWrap: {
    alignItems: 'center',
    justifyContent: 'center',
    gap: 3,
    minHeight: 34,
  },
  labOperationValue: {
    fontSize: 11,
    fontWeight: '800',
    lineHeight: 13,
    opacity: 0.9,
  },
  labIntensityControl: {
    position: 'absolute',
    top: '50%',
    marginTop: -13,
    width: 26,
    height: 26,
    borderRadius: 13,
    backgroundColor: 'rgba(0,0,0,0.18)',
    alignItems: 'center',
    justifyContent: 'center',
    zIndex: 2,
  },
  labIntensityControlLeft: {
    left: 8,
  },
  labIntensityControlRight: {
    right: 8,
  },
  labIntensityControlText: {
    fontSize: 17,
    lineHeight: 20,
    fontWeight: '900',
  },
  sliderRow: {
    flexDirection: 'row',
    alignItems: 'center',
    gap: 8,
  },
  sliderBtn: {
    width: 34,
    height: 34,
    borderRadius: 17,
    backgroundColor: 'rgba(120,120,120,0.18)',
    alignItems: 'center',
    justifyContent: 'center',
  },
  sliderTrack: {
    flex: 1,
    height: 6,
    borderRadius: 3,
    backgroundColor: 'rgba(120,120,120,0.2)',
    overflow: 'hidden',
  },
  sliderFill: {
    height: '100%',
    borderRadius: 3,
  },
  nativeSlider: {
    width: '100%',
    height: 38,
  },
  accessoryLiveStatus: {
    flexDirection: 'row',
    alignItems: 'center',
    gap: 12,
    borderRadius: 14,
    paddingHorizontal: 12,
    paddingVertical: 11,
    backgroundColor: 'rgba(120,120,120,0.10)',
  },
  accessoryLiveTitle: {
    fontSize: 13,
    fontWeight: '800',
  },
  accessoryModelGrid: {
    flexDirection: 'row',
    flexWrap: 'wrap',
    gap: 10,
  },
  accessoryModelCard: {
    width: '31.5%',
    minWidth: 104,
    minHeight: 116,
    borderRadius: 14,
    borderWidth: 1.5,
    padding: 8,
    gap: 7,
    position: 'relative',
  },
  accessoryThumbWrap: {
    width: '100%',
    height: 64,
    borderRadius: 10,
    backgroundColor: 'rgba(255,255,255,0.08)',
    overflow: 'hidden',
    alignItems: 'center',
    justifyContent: 'center',
  },
  accessoryThumb: {
    width: '100%',
    height: '100%',
  },
  accessoryModelLabel: {
    fontSize: 11,
    fontWeight: '800',
    textAlign: 'center',
  },
  accessorySelectedDot: {
    position: 'absolute',
    right: 8,
    top: 8,
    width: 10,
    height: 10,
    borderRadius: 5,
  },
  makeupColorRow: {
    flexDirection: 'row',
    alignItems: 'center',
    gap: 10,
  },
  makeupColorPreview: {
    width: 48,
    height: 48,
    borderRadius: 14,
    borderWidth: 1,
    borderColor: 'rgba(120,120,120,0.16)',
    alignItems: 'center',
    justifyContent: 'center',
    backgroundColor: 'rgba(120,120,120,0.08)',
  },
  makeupColorSwatch: {
    width: 30,
    height: 30,
    borderRadius: 10,
    borderWidth: 1,
    borderColor: 'rgba(255,255,255,0.45)',
  },
  makeupHexInput: {
    flex: 1,
    height: 48,
    borderRadius: 14,
    borderWidth: 1.5,
    paddingHorizontal: 14,
    fontSize: 14,
    fontWeight: '700',
    letterSpacing: 0.8,
    backgroundColor: 'rgba(120,120,120,0.08)',
  },
  makeupSwatchGrid: {
    flexDirection: 'row',
    flexWrap: 'wrap',
    gap: 8,
  },
  makeupSwatch: {
    minWidth: 86,
    minHeight: 38,
    borderRadius: 12,
    borderWidth: 2,
    paddingHorizontal: 10,
    alignItems: 'center',
    justifyContent: 'center',
  },
  makeupSwatchLabel: {
    fontSize: 11,
    fontWeight: '800',
  },
  sideBySide: {
    flexDirection: 'row',
    gap: 8,
  },
  compareCard: {
    width: '100%',
    gap: 8,
  },
  compareHeaderRow: {
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'space-between',
    gap: 8,
  },
  compareHintPill: {
    paddingHorizontal: 10,
    paddingVertical: 6,
    borderRadius: 999,
    backgroundColor: 'rgba(120,120,120,0.12)',
  },
  compareHintText: {
    fontSize: 11,
    fontWeight: '700',
    opacity: 0.8,
  },
  compareStage: {
    width: '100%',
    height: 176,
    borderRadius: 12,
    overflow: 'hidden',
    backgroundColor: 'rgba(120,120,120,0.1)',
    position: 'relative',
  },
  compareImage: {
    width: '100%',
    height: '100%',
  },
  compareOverlay: {
    ...StyleSheet.absoluteFillObject,
  },
  originalTag: {
    position: 'absolute',
    left: 10,
    top: 10,
    paddingHorizontal: 10,
    paddingVertical: 6,
    borderRadius: 999,
    backgroundColor: 'rgba(0,0,0,0.55)',
  },
  originalTagText: {
    color: '#FFFFFF',
    fontSize: 11,
    fontWeight: '700',
  },
  compareFab: {
    position: 'absolute',
    right: 10,
    bottom: 10,
    width: 44,
    height: 44,
    borderRadius: 22,
    backgroundColor: 'rgba(0,0,0,0.68)',
    alignItems: 'center',
    justifyContent: 'center',
    borderWidth: 1,
    borderColor: 'rgba(255,255,255,0.28)',
  },
  sideBox: {
    flex: 1,
    gap: 4,
  },
  sideLabel: {
    fontSize: 11,
    opacity: 0.7,
    textAlign: 'center',
  },
  sideImage: {
    width: '100%',
    height: 120,
    borderRadius: 10,
    backgroundColor: 'rgba(120,120,120,0.1)',
  },
  sideImagePlaceholder: {
    width: '100%',
    height: 120,
    borderRadius: 10,
    backgroundColor: 'rgba(120,120,120,0.08)',
    alignItems: 'center',
    justifyContent: 'center',
    paddingHorizontal: 10,
  },
  agingRow: {
    flexDirection: 'row',
    gap: 8,
  },
  winnerCard: {
    borderRadius: 14,
    borderWidth: 1,
    borderColor: 'rgba(250,204,21,0.35)',
    backgroundColor: 'rgba(250,204,21,0.07)',
    padding: 14,
    marginTop: 10,
    gap: 10,
  },
  winnerHeader: {
    alignItems: 'center',
  },
  winnerTitle: {
    fontSize: 15,
    fontWeight: '700',
    textAlign: 'center',
  },
  winnerMetrics: {
    flexDirection: 'row',
    justifyContent: 'space-around',
  },
  winnerMetricItem: {
    alignItems: 'center',
    gap: 2,
  },
  winnerMetricLabel: {
    fontSize: 11,
    opacity: 0.6,
  },
  winnerMetricValue: {
    fontSize: 13,
    fontWeight: '600',
  },
  metricsCard: {
    borderRadius: 16,
    borderWidth: 1,
    borderColor: 'rgba(120,120,120,0.18)',
    backgroundColor: 'rgba(120,120,120,0.06)',
    padding: 14,
    gap: 10,
  },
  metricsCardHeader: {
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'space-between',
    flexWrap: 'wrap',
    gap: 8,
  },
  metricsCardTitle: {
    fontSize: 14,
  },
  metricsSourcePill: {
    paddingHorizontal: 9,
    paddingVertical: 3,
    borderRadius: 999,
    backgroundColor: 'rgba(120,120,120,0.16)',
  },
  metricsSourceText: {
    fontSize: 10,
    fontWeight: '600',
    color: '#CBD5E1',
  },
  metricTableHeaderRow: {
    flexDirection: 'row',
    borderWidth: 1,
    borderColor: 'rgba(120,120,120,0.22)',
    backgroundColor: 'rgba(0,0,0,0.10)',
    borderRadius: 8,
    paddingVertical: 7,
    paddingHorizontal: 8,
  },
  metricTableDataRow: {
    flexDirection: 'row',
    alignItems: 'center',
    borderWidth: 1,
    borderColor: 'rgba(120,120,120,0.13)',
    borderRadius: 8,
    paddingVertical: 8,
    paddingHorizontal: 8,
    backgroundColor: 'rgba(255,255,255,0.03)',
  },
  metricHeaderCell: {
    fontSize: 11,
    fontWeight: '700',
    color: '#CBD5E1',
  },
  metricDataCell: {
    fontSize: 11,
    color: '#E2E8F0',
  },
  metricHeaderMetric: {
    flex: 0.85,
  },
  metricHeaderValue: {
    flex: 1.1,
  },
  metricHeaderPurpose: {
    flex: 2.55,
  },
  metricDataMetric: {
    flex: 0.85,
    fontWeight: '700',
  },
  metricDataValue: {
    flex: 1.1,
  },
  metricValueCell: {
    flexDirection: 'row',
    alignItems: 'center',
    gap: 5,
  },
  metricStatusDot: {
    width: 7,
    height: 7,
    borderRadius: 3.5,
  },
  metricValueText: {
    fontSize: 11,
    fontWeight: '700',
  },
  metricDataPurpose: {
    flex: 2.55,
    opacity: 0.85,
    fontSize: 10.5,
  },
  hfLfRow: {
    flexDirection: 'row',
    gap: 6,
    marginTop: 2,
  },
  hfLfStatBox: {
    flex: 1,
    backgroundColor: 'rgba(120,120,120,0.10)',
    borderRadius: 10,
    paddingVertical: 8,
    paddingHorizontal: 10,
    alignItems: 'center',
    gap: 3,
  },
  hfLfDeltaBox: {
    backgroundColor: 'rgba(120,120,120,0.14)',
  },
  hfLfLabel: {
    fontSize: 9,
    fontWeight: '600',
    color: '#CBD5E1',
    textTransform: 'uppercase',
    letterSpacing: 0.5,
  },
  hfLfValue: {
    fontSize: 13,
    fontWeight: '700',
    color: '#F1F5F9',
  },
  aiAnalysisCard: {
    borderRadius: 14,
    borderWidth: 1,
    borderColor: 'rgba(120,120,120,0.22)',
    backgroundColor: 'rgba(120,120,120,0.09)',
    padding: 12,
    gap: 10,
  },
  aiAnalysisHeaderRow: {
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'space-between',
    gap: 10,
  },
  aiAnalysisPill: {
    paddingHorizontal: 10,
    paddingVertical: 6,
    borderRadius: 999,
    backgroundColor: 'rgba(120,120,120,0.14)',
  },
  aiAnalysisPillText: {
    fontSize: 11,
    fontWeight: '700',
    opacity: 0.85,
  },
  ageCompareRow: {
    flexDirection: 'row',
    gap: 8,
  },
  ageStatCard: {
    flex: 1,
    borderRadius: 12,
    paddingVertical: 10,
    paddingHorizontal: 12,
    backgroundColor: 'rgba(255,255,255,0.22)',
    gap: 4,
  },
  ageStatLabel: {
    fontSize: 11,
    textTransform: 'uppercase',
    letterSpacing: 0.4,
    opacity: 0.68,
  },
  ageStatValue: {
    fontSize: 22,
    fontWeight: '800',
  },
  lightboxBackdrop: {
    flex: 1,
    backgroundColor: 'rgba(0,0,0,0.92)',
    alignItems: 'center',
    justifyContent: 'center',
  },
  lightboxViewport: {
    width: '100%',
    height: '100%',
    alignItems: 'center',
    justifyContent: 'center',
    overflow: 'hidden',
  },
  lightboxImage: {
    width: '100%',
    height: '100%',
  },
  lightboxControls: {
    position: 'absolute',
    bottom: 28,
    left: 20,
    right: 20,
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'center',
    gap: 10,
  },
  lightboxToolButton: {
    width: 42,
    height: 42,
    borderRadius: 21,
    backgroundColor: 'rgba(255,255,255,0.15)',
    alignItems: 'center',
    justifyContent: 'center',
  },
  lightboxZoomPill: {
    minWidth: 72,
    height: 42,
    borderRadius: 21,
    paddingHorizontal: 14,
    backgroundColor: 'rgba(255,255,255,0.15)',
    alignItems: 'center',
    justifyContent: 'center',
  },
  lightboxZoomText: {
    color: '#FFFFFF',
    fontSize: 13,
    fontWeight: '800',
  },
  lightboxClose: {
    position: 'absolute',
    top: 48,
    right: 20,
    width: 40,
    height: 40,
    borderRadius: 20,
    backgroundColor: 'rgba(255,255,255,0.15)',
    alignItems: 'center',
    justifyContent: 'center',
  },
  makeupLayersContainer: {
    marginTop: 10,
    paddingHorizontal: 10,
    paddingVertical: 10,
    borderRadius: 12,
    backgroundColor: 'rgba(120,120,120,0.08)',
    gap: 10,
  },
  makeupLayersTitle: {
    fontSize: 12,
    fontWeight: '700',
    opacity: 0.7,
    textTransform: 'uppercase',
    letterSpacing: 0.5,
  },
  makeupLayerRow: {
    flexDirection: 'row',
    alignItems: 'center',
    paddingHorizontal: 10,
    paddingVertical: 8,
    borderRadius: 8,
    backgroundColor: 'rgba(255,255,255,0.05)',
    gap: 8,
  },
  makeupLayerColor: {
    width: 24,
    height: 24,
    borderRadius: 6,
    borderWidth: 1,
    borderColor: 'rgba(255,255,255,0.2)',
  },
  makeupLayerLabel: {
    flex: 1,
    fontSize: 12,
    fontWeight: '600',
  },
  makeupLayerDeleteBtn: {
    width: 24,
    height: 24,
    alignItems: 'center',
    justifyContent: 'center',
  },
  tabBarContainer: {
    marginBottom: 16,
  },
  tabBarScroll: {
    gap: 8,
    paddingVertical: 2,
  },
  tabButton: {
    paddingHorizontal: 16,
    paddingVertical: 8,
    borderRadius: 20,
    borderWidth: 1,
    borderColor: 'rgba(120,120,120,0.3)',
    backgroundColor: 'rgba(120,120,120,0.1)',
  },
  tabButtonText: {
    fontSize: 13,
    fontWeight: '600',
    opacity: 0.9,
  },
  variantsGrid: {
    flexDirection: 'row',
    flexWrap: 'wrap',
    gap: 8,
    paddingBottom: 8,
  },
  variantButton: {
    width: '31%',
    flexDirection: 'column',
    alignItems: 'center',
    justifyContent: 'center',
    gap: 6,
    paddingVertical: 10,
    paddingHorizontal: 4,
    borderRadius: 12,
    borderWidth: 1.5,
  },
  variantImage: {
    width: 44,
    height: 44,
    borderRadius: 8,
    backgroundColor: 'rgba(255,255,255,0.05)',
  },
  variantImagePlaceholder: {
    width: 44,
    height: 44,
    borderRadius: 8,
    backgroundColor: 'rgba(255,255,255,0.05)',
  },
  variantLabel: {
    fontSize: 11,
    fontWeight: '700',
    textAlign: 'center',
  },
});
