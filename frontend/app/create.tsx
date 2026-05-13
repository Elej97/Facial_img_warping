import Slider from '@react-native-community/slider';
import * as FileSystem from 'expo-file-system';
import { Image } from 'expo-image';
import * as ImageManipulator from 'expo-image-manipulator';
import * as ImagePicker from 'expo-image-picker';
import { useRouter } from 'expo-router';
import * as Sharing from 'expo-sharing';
import { useEffect, useMemo, useRef, useState } from 'react';
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

import { SideNav } from '@/components/side-nav';
import { ThemedText } from '@/components/themed-text';
import { ThemedView } from '@/components/themed-view';
import { Colors } from '@/constants/theme';
import { useColorScheme } from '@/hooks/use-color-scheme';
import {
    applyMakeupFromBase64,
    estimateAgeFromBase64,
    estimateAgeFromUri,
    exportEvaluationReportFromBase64,
    frequencyFromBase64,
    frequencyProFromBase64,
    landmarksFromBase64,
    preprocessFromUri,
    transferExpressionFromBase64,
    warpFromBase64,
    warpProFromBase64,
    type ProMetrics,
    type ProWarpOperation
} from '@/services/facial-api';
import { Ionicons } from '@expo/vector-icons';

const MIN_WIDTH = 512;
const MIN_HEIGHT = 512;
const MIN_CROP_SIZE = 24;
const BASE64_ALPHABET = 'ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/';

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

type ProOperation = ProWarpOperation | 'aging' | 'deaging';
type ProPreset = 'natural' | 'balanced' | 'strong';
type MakeupUiTarget = 'lip' | 'cheek' | 'bronzer' | 'lash' | 'brow';
type MakeupBackendRegion = 'lip' | 'cheek' | 'brow' | 'lash';

type MakeupPreset = {
  key: MakeupUiTarget;
  label: string;
  backendRegion: MakeupBackendRegion;
  defaultColor: string;
};

type EvalMetricRow = {
  metric: 'MSE' | 'PSNR' | 'SSIM';
  value: string;
  purposeRange: string;
};

const PRO_OPERATIONS: ProOperation[] = [
  'smile_enhancement',
  'brow_lift',
  'lip_plump',
  'slim_face',
  'aging',
  'deaging',
];

const PRO_LABEL: Record<ProOperation, string> = {
  smile_enhancement: 'Pro Smile',
  brow_lift: 'Pro Brow Lift',
  lip_plump: 'Pro Lip Plump',
  slim_face: 'Pro Slim Face',
  aging: 'Pro Aging',
  deaging: 'Pro De-Aging',
};

const PRO_PRESET_VALUES: Record<ProPreset, { intensity: number; rbfSmooth: number }> = {
  natural: { intensity: 0.3, rbfSmooth: 4.0 },
  balanced: { intensity: 0.6, rbfSmooth: 3.0 },
  strong: { intensity: 0.85, rbfSmooth: 2.0 },
};

const PRO_PRESET_LABEL: Record<ProPreset, string> = {
  natural: 'Natural',
  balanced: 'Balanced',
  strong: 'Strong',
};

const MANUAL_MAKEUP_PRESETS: MakeupPreset[] = [
  { key: 'lip', label: 'Ruj', backendRegion: 'lip', defaultColor: '#D45A73' },
  { key: 'cheek', label: 'Allık', backendRegion: 'cheek', defaultColor: '#F29AAF' },
  { key: 'bronzer', label: 'Bronzer', backendRegion: 'cheek', defaultColor: '#B97A4C' },
  { key: 'lash', label: 'Kirpik', backendRegion: 'lash', defaultColor: '#1D1D1F' },
  { key: 'brow', label: 'Kaş', backendRegion: 'brow', defaultColor: '#5E4735' },
];

const MANUAL_MAKEUP_SWATCHES: Record<MakeupUiTarget, string[]> = {
  lip: ['#D45A73', '#A83253', '#F18FA7', '#BE4369', '#FF7AA2'],
  cheek: ['#F29AAF', '#F2B2A6', '#E88E7A', '#DB6F93', '#F7C1C8'],
  bronzer: ['#B97A4C', '#9F6642', '#D09A6B', '#7F5337', '#C88557'],
  lash: ['#1D1D1F', '#2E2E33', '#505057'],
  brow: ['#5E4735', '#463427', '#7A5B43', '#2D221A'],
};

const MANUAL_MAKEUP_LABELS: Record<MakeupUiTarget, string> = {
  lip: 'Ruj',
  cheek: 'Allık',
  bronzer: 'Bronzer',
  lash: 'Kirpik',
  brow: 'Kaş',
};

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

export default function CreateScreen() {
  const router = useRouter();
  const colorScheme = useColorScheme() ?? 'light';
  const colors = Colors[colorScheme];
  // In dark mode tint is #fff — text on tint buttons must be dark to be visible
  const tintTextColor = colorScheme === 'dark' ? '#11181C' : '#FFFFFF';
  const { width } = useWindowDimensions();
  const isWide = width >= 960;
  const [selectedImageName, setSelectedImageName] = useState<string | null>(null);
  const [selectedImageUri, setSelectedImageUri] = useState<string | null>(null);
  const [selectedImageSize, setSelectedImageSize] = useState<{ width: number; height: number } | null>(null);
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

  const [landmarkLoading, setLandmarkLoading] = useState(false);
  const [landmarkError, setLandmarkError] = useState<string | null>(null);
  const [landmarkB64, setLandmarkB64] = useState<string | null>(null);
  const [landmarkCount, setLandmarkCount] = useState<number | null>(null);
  const [showLandmarks, setShowLandmarks] = useState(false);

  const [warpOp, setWarpOp] = useState<'smile' | 'raise_eyebrows' | 'widen_lips' | 'slim_face'>('smile');
  const [warpIntensity, setWarpIntensity] = useState(0.8);
  const [warpLoading, setWarpLoading] = useState(false);
  const [warpError, setWarpError] = useState<string | null>(null);
  const [warpResultB64, setWarpResultB64] = useState<string | null>(null);

  const [referenceExpressionName, setReferenceExpressionName] = useState<string | null>(null);
  const [referenceExpressionUri, setReferenceExpressionUri] = useState<string | null>(null);
  const [referenceExpressionSize, setReferenceExpressionSize] = useState<{ width: number; height: number } | null>(null);
  const [expressionTransferIntensity, setExpressionTransferIntensity] = useState(0.75);
  const [expressionTransferLoading, setExpressionTransferLoading] = useState(false);
  const [expressionTransferError, setExpressionTransferError] = useState<string | null>(null);
  const [expressionTransferResultB64, setExpressionTransferResultB64] = useState<string | null>(null);
  const [manualLipWarpResultB64, setManualLipWarpResultB64] = useState<string | null>(null);
  const [manualLipWarpError, setManualLipWarpError] = useState<string | null>(null);

  const [agingIntensity, setAgingIntensity] = useState(0.8);
  const [agingLoading, setAgingLoading] = useState(false);
  const [agingError, setAgingError] = useState<string | null>(null);
  const [agingResultB64, setAgingResultB64] = useState<string | null>(null);
  const [agingMode, setAgingMode] = useState<'aging' | 'deaging'>('aging');
  const [proOperation, setProOperation] = useState<ProOperation>('smile_enhancement');
  const [proPreset, setProPreset] = useState<ProPreset>('balanced');
  const [proIntensity, setProIntensity] = useState(0.65);
  const [proRbfSmooth, setProRbfSmooth] = useState(2.8);
  const [proLoading, setProLoading] = useState(false);
  const [proError, setProError] = useState<string | null>(null);
  const [proResultB64, setProResultB64] = useState<string | null>(null);
  const [proMetrics, setProMetrics] = useState<ProMetrics | null>(null);
  const [makeupTarget, setMakeupTarget] = useState<MakeupUiTarget>('lip');
  const [makeupHexColor, setMakeupHexColor] = useState(MANUAL_MAKEUP_PRESETS[0].defaultColor);
  const [makeupIntensity, setMakeupIntensity] = useState(0.48);
  const [makeupLoading, setMakeupLoading] = useState(false);
  const [makeupError, setMakeupError] = useState<string | null>(null);
  const [makeupResultB64, setMakeupResultB64] = useState<string | null>(null);
  const [evalMetrics, setEvalMetrics] = useState<ProMetrics | null>(null);
  const [evalSourceLabel, setEvalSourceLabel] = useState<string | null>(null);
  const [evalResultB64, setEvalResultB64] = useState<string | null>(null);
  const [spectrumBeforeB64, setSpectrumBeforeB64] = useState<string | null>(null);
  const [spectrumAfterB64, setSpectrumAfterB64] = useState<string | null>(null);
  const [ageBefore, setAgeBefore] = useState<number | null>(null);
  const [ageAfter, setAgeAfter] = useState<number | null>(null);
  const [ageLoading, setAgeLoading] = useState(false);
  const [ageError, setAgeError] = useState<string | null>(null);
  const proDebounceRef = useRef<ReturnType<typeof setTimeout> | null>(null);
  const ageRequestRef = useRef(0);
  const [proCompareHeld, setProCompareHeld] = useState(false);
  const proCompareOpacity = useRef(new Animated.Value(0)).current;
  const [lightboxUri, setLightboxUri] = useState<string | null>(null);

  const metricTableRows = useMemo<EvalMetricRow[]>(() => {
    if (!evalMetrics) {
      return [];
    }

    return [
      {
        metric: 'MSE',
        value: Number.isFinite(evalMetrics.mse) ? evalMetrics.mse.toFixed(6) : 'N/A',
        purposeRange: 'Pixel-level difference. Lower is better, ideal 0.',
      },
      {
        metric: 'PSNR',
        value: Number.isFinite(evalMetrics.psnr) ? `${evalMetrics.psnr.toFixed(4)} dB` : 'Infinity',
        purposeRange: 'Signal quality. Higher is better, generally > 30 dB.',
      },
      {
        metric: 'SSIM',
        value: Number.isFinite(evalMetrics.ssim) ? evalMetrics.ssim.toFixed(6) : 'N/A',
        purposeRange: 'Perceptual similarity. Closer to 1 is better, generally >= 0.80.',
      },
    ];
  }, [evalMetrics]);

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

  useEffect(() => {
    cropBoxRef.current = cropBox;
  }, [cropBox]);

  useEffect(() => {
    if (cropEditorVisible && cropStageLayout && selectedImageSize && !cropBoxRef.current) {
      ensureCropBox(cropStageLayout);
    }
  }, [cropEditorVisible, cropStageLayout, selectedImageSize]);

  useEffect(() => {
    Animated.timing(proCompareOpacity, {
      toValue: proCompareHeld ? 1 : 0,
      duration: 200,
      useNativeDriver: true,
    }).start();
  }, [proCompareHeld, proCompareOpacity]);

  const updateCropBox = (nextBox: CropBox | null) => {
    cropBoxRef.current = nextBox;
    setCropBox(nextBox);
  };

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

  const runAgeAnalysis = async (source: string, target: AgeTarget, kind: 'uri' | 'base64') => {
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
  };

  const closeCropEditor = () => {
    setCropEditorVisible(false);
    setCropStageLayout(null);
    updateCropBox(null);
  };

  const ensureCropBox = (stage: StageLayout) => {
    if (!selectedImageSize) {
      return;
    }

    updateCropBox(createInitialCropBox(stage, selectedImageSize));
  };

  const openCropEditor = () => {
    if (!selectedImageUri || !selectedImageSize) {
      setProcessState('error');
      setStatusMessage('Kırpma için önce bir görsel seçmelisin.');
      return;
    }

    setCropEditorVisible(true);
    setStatusMessage('Kırpma alanını sürükle ya da köşeden büyüt/küçült.');
  };

  const moveCropResponder = useMemo(
    () =>
      PanResponder.create({
        onStartShouldSetPanResponder: () => false,
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
      }),
    [cropStageLayout, selectedImageSize]
  );

  const createResizeResponder = useMemo(
    () => (mode: 'x' | 'y' | 'xy') =>
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
          const maxWidth = contain.offsetX + contain.renderWidth - startBox.x;
          const maxHeight = contain.offsetY + contain.renderHeight - startBox.y;

          const nextWidth =
            mode === 'y' ? startBox.width : clamp(startBox.width + gestureState.dx, MIN_CROP_SIZE, maxWidth);
          const nextHeight =
            mode === 'x' ? startBox.height : clamp(startBox.height + gestureState.dy, MIN_CROP_SIZE, maxHeight);

          updateCropBox(
            clampCropBox(
              {
                x: startBox.x,
                y: startBox.y,
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
      }),
    [cropStageLayout, selectedImageSize]
  );

  const resizeRightResponder = useMemo(() => createResizeResponder('x'), [createResizeResponder]);
  const resizeBottomResponder = useMemo(() => createResizeResponder('y'), [createResizeResponder]);
  const resizeCornerResponder = useMemo(() => createResizeResponder('xy'), [createResizeResponder]);

  const pickImage = async () => {
    const result = await ImagePicker.launchImageLibraryAsync({
      mediaTypes: ['images'],
      quality: 0.4,  // Daha düşük çözünürlük (çok hızlı işleme)
      allowsEditing: false,
      selectionLimit: 1,
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
      setProcessState('error');
      setStatusMessage('Lütfen geçerli bir görsel dosyası seçin.');
      return;
    }

    if ((asset.width ?? 0) < MIN_WIDTH || (asset.height ?? 0) < MIN_HEIGHT) {
      setSelectedImageName(null);
      setSelectedImageUri(null);
      setSelectedImageSize(null);
      setProcessState('error');
      setStatusMessage(`Minimum çözünürlük ${MIN_WIDTH}x${MIN_HEIGHT} olmalı. Seçilen: ${asset.width}x${asset.height}.`);
      return;
    }

    setSelectedImageName(rawName);
    setSelectedImageUri(asset.uri);
    setSelectedImageSize({ width: asset.width ?? MIN_WIDTH, height: asset.height ?? MIN_HEIGHT });
    setProcessState('selected');
    setStatusMessage(`Seçilen görsel hazır: ${asset.width}x${asset.height}.`);
    setCropApplied(false);
    resetExpressionTransferState();
    resetAgeAnalysis();
    setMakeupResultB64(null);
    setMakeupError(null);
    closeCropEditor();
    void runAgeAnalysis(asset.uri, 'before', 'uri');
  };

  const pickReferenceExpressionImage = async () => {
    const result = await ImagePicker.launchImageLibraryAsync({
      mediaTypes: ['images'],
      quality: 0.4,
      allowsEditing: false,
      selectionLimit: 1,
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
    setReferenceExpressionUri(asset.uri);
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
        { compress: 1, format: ImageManipulator.SaveFormat.PNG }
      );

      setSelectedImageUri(result.uri);
      setSelectedImageSize({ width: result.width ?? Math.round(cropWidth), height: result.height ?? Math.round(cropHeight) });
      setCropApplied(true);
      setProcessState('selected');
      setMakeupResultB64(null);
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

  const handlePreprocess = async () => {
    if (!selectedImageUri) return;
    setPreprocessLoading(true);
    setPreprocessError(null);
    setPreprocessedB64(null);
    setLandmarkB64(null);
    setLandmarkCount(null);
    setWarpResultB64(null);
    setMakeupResultB64(null);
    setMakeupError(null);
    resetExpressionTransferState();
    setAgingResultB64(null);
    setProResultB64(null);
    setProMetrics(null);
    setEvalMetrics(null);
    setEvalSourceLabel(null);
    setEvalResultB64(null);
    setSpectrumBeforeB64(null);
    setSpectrumAfterB64(null);
    setAgeAfter(null);
    setProCompareHeld(false);
    proCompareOpacity.setValue(0);
    try {
      const data = await preprocessFromUri(selectedImageUri);
      if (!data.success) throw new Error(data.message ?? 'Preprocess failed');
      setPreprocessedB64(data.processed_image_b64);
    } catch (e: any) {
      setPreprocessError(e?.message ?? 'Unknown error');
    } finally {
      setPreprocessLoading(false);
    }
  };

  const handleLandmarks = async () => {
    if (!preprocessedB64) return;
    setLandmarkLoading(true);
    setLandmarkError(null);
    try {
      const data = await landmarksFromBase64(preprocessedB64);
      if (!data.success) throw new Error(data.message ?? 'Landmark detection failed');
      setLandmarkB64(data.landmark_image_b64);
      setLandmarkCount(data.landmark_count);
    } catch (e: any) {
      setLandmarkError(e?.message ?? 'Unknown error');
    } finally {
      setLandmarkLoading(false);
    }
  };

  const handleWarp = async () => {
    if (!preprocessedB64 || !landmarkCount) return;
    setWarpLoading(true);
    setWarpError(null);
    setWarpResultB64(null);
    try {
      const data = await warpFromBase64(preprocessedB64, warpOp, warpIntensity);
      if (!data.success) throw new Error(data.message ?? 'Warp failed');
      setWarpResultB64(data.result_image_b64);
      setEvalMetrics(data.metrics ?? null);
      setEvalSourceLabel(`Warp / ${warpOp}`);
      setEvalResultB64(data.result_image_b64 ?? null);
      setAgeAfter(null);
      void runAgeAnalysis(data.result_image_b64, 'after', 'base64');
    } catch (e: any) {
      setWarpError(e?.message ?? 'Unknown error');
    } finally {
      setWarpLoading(false);
    }
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
      const transferData = await transferExpressionFromBase64(preprocessedB64, referenceExpressionUri, expressionTransferIntensity, {
        landmarkBackend: 'hybrid',
      });

      if (!transferData.success) {
        throw new Error(transferData.message ?? 'Expression transfer failed');
      }

      setExpressionTransferResultB64(transferData.result_image_b64);
      setAgeAfter(null);
      void runAgeAnalysis(transferData.result_image_b64, 'after', 'base64');

      try {
        const baselineData = await warpFromBase64(preprocessedB64, 'widen_lips', expressionTransferIntensity);
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

  const handleAging = async (mode: 'aging' | 'deaging') => {
    if (!preprocessedB64) return;
    setAgingMode(mode);
    setAgingLoading(true);
    setAgingError(null);
    setAgingResultB64(null);
    try {
      const data = await frequencyFromBase64(preprocessedB64, mode, agingIntensity);
      if (!data.success) throw new Error(data.message ?? 'Frequency effect failed');
      setAgingResultB64(data.result_image_b64);
      setEvalMetrics(data.metrics ?? null);
      setEvalSourceLabel(mode === 'aging' ? 'Frequency / Aging' : 'Frequency / De-Aging');
      setEvalResultB64(data.result_image_b64 ?? null);
      setAgeAfter(null);
      void runAgeAnalysis(data.result_image_b64, 'after', 'base64');
    } catch (e: any) {
      setAgingError(e?.message ?? 'Unknown error');
    } finally {
      setAgingLoading(false);
    }
  };

  const runProOperation = async (override?: { intensity?: number; rbfSmooth?: number; operation?: ProOperation }) => {
    if (!preprocessedB64) return;

    const effectiveOperation = override?.operation ?? proOperation;
    const effectiveIntensity = override?.intensity ?? proIntensity;
    const effectiveRbfSmooth = override?.rbfSmooth ?? proRbfSmooth;

    setProCompareHeld(false);

    setProLoading(true);
    setProError(null);
    try {
      if (effectiveOperation === 'aging' || effectiveOperation === 'deaging') {
        const data = await frequencyProFromBase64(preprocessedB64, effectiveOperation, effectiveIntensity, {
          landmarkBackend: 'hybrid',
          temporalSmoothing: true,
          emaAlpha: 0.62,
          streamId: 'pro-ui',
        });
        if (!data.success) throw new Error(data.message ?? 'Pro frequency failed');
        setProResultB64(data.result_image_b64);
        setProMetrics(data.metrics ?? null);
        setEvalMetrics(data.metrics ?? null);
        setEvalSourceLabel(data.mode === 'aging' ? 'Pro Frequency / Aging' : 'Pro Frequency / De-Aging');
        setEvalResultB64(data.result_image_b64 ?? null);
        setSpectrumBeforeB64(data.spectrum_before_b64 ?? null);
        setSpectrumAfterB64(data.spectrum_after_b64 ?? null);
        setAgeAfter(null);
        void runAgeAnalysis(data.result_image_b64, 'after', 'base64');
      } else {
        const data = await warpProFromBase64(preprocessedB64, effectiveOperation, effectiveIntensity, effectiveRbfSmooth, {
          landmarkBackend: 'hybrid',
          temporalSmoothing: true,
          emaAlpha: 0.62,
          streamId: 'pro-ui',
        });
        if (!data.success) throw new Error(data.message ?? 'Pro warp failed');
        setProResultB64(data.result_image_b64);
        setProMetrics(data.metrics ?? null);
        setEvalMetrics(data.metrics ?? null);
        setEvalSourceLabel(`Pro Warp / ${PRO_LABEL[effectiveOperation]}`);
        setEvalResultB64(data.result_image_b64 ?? null);
        setSpectrumBeforeB64(null);
        setSpectrumAfterB64(null);
        setAgeAfter(null);
        void runAgeAnalysis(data.result_image_b64, 'after', 'base64');
      }
    } catch (e: any) {
      setProError(e?.message ?? 'Unknown pro error');
    } finally {
      setProLoading(false);
    }
  };

  const applyMakeup = async () => {
    if (!preprocessedB64) return;

    const preset = MANUAL_MAKEUP_PRESETS.find((item) => item.key === makeupTarget) ?? MANUAL_MAKEUP_PRESETS[0];
    const hexColor = normalizeHexColor(makeupHexColor, preset.defaultColor);

    setMakeupLoading(true);
    setMakeupError(null);

    try {
      const data = await applyMakeupFromBase64(preprocessedB64, preset.backendRegion, hexColor, makeupIntensity, {
        landmarkBackend: 'hybrid',
        temporalSmoothing: true,
        emaAlpha: 0.62,
        streamId: 'makeup-ui',
      });

      if (!data.success) {
        throw new Error(data.message ?? 'Makeup effect failed');
      }

      setMakeupResultB64(data.result_image_b64);
      setAgeAfter(null);
      void runAgeAnalysis(data.result_image_b64, 'after', 'base64');
    } catch (error: any) {
      setMakeupError(error?.message ?? 'Unknown makeup error');
    } finally {
      setMakeupLoading(false);
    }
  };

  const applyProPreset = (preset: ProPreset) => {
    const values = PRO_PRESET_VALUES[preset];
    setProPreset(preset);
    setProIntensity(values.intensity);
    setProRbfSmooth(values.rbfSmooth);

    if (!preprocessedB64) return;
    if (proDebounceRef.current) {
      clearTimeout(proDebounceRef.current);
    }
    runProOperation({ intensity: values.intensity, rbfSmooth: values.rbfSmooth });
  };

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
  }, [preprocessedB64, proOperation, proIntensity, proRbfSmooth]);

  const exportCsv = async () => {
    if (!preprocessedB64 || !evalResultB64 || !evalMetrics) {
      Alert.alert('Rapor Hazır Değil', 'Önce bir warp veya frequency işlemi çalıştırmalısın.');
      return;
    }

    try {
      const data = await exportEvaluationReportFromBase64(
        'csv',
        evalSourceLabel ?? 'Unknown Operation',
        preprocessedB64,
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
    if (!preprocessedB64 || !evalResultB64 || !evalMetrics) {
      Alert.alert('Rapor Hazır Değil', 'Önce bir warp veya frequency işlemi çalıştırmalısın.');
      return;
    }

    try {
      const data = await exportEvaluationReportFromBase64(
        'pdf',
        evalSourceLabel ?? 'Unknown Operation',
        preprocessedB64,
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
          <ThemedText style={styles.helperText}>Reference expression to target face</ThemedText>
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

  return (
    <ThemedView style={styles.screen}>
      <SideNav />

      <View style={styles.mainContent}>
      <View style={styles.header}>
        <Pressable onPress={() => router.back()}>
          <Ionicons name="chevron-back" size={28} color={colors.text} />
        </Pressable>
        <ThemedText type="title">Oluştur</ThemedText>
        <View style={{ width: 28 }} />
      </View>

      <ScrollView showsVerticalScrollIndicator={false} contentContainerStyle={styles.content}>
        <View style={[styles.workspace, isWide && styles.workspaceWide]}>
          <View
            style={[
              styles.panel,
              styles.uploadPanel,
              {
                backgroundColor: colorScheme === 'dark' ? '#202426' : '#F3F7FC',
                borderColor: colorScheme === 'dark' ? '#32383B' : '#D7E0EA',
              },
            ]}>
            <ThemedText type="subtitle">Fotoğraf Yükle</ThemedText>
            <ThemedText style={styles.helperText}>
              Fotoğrafını seçip merkez önizlemede kırpma işlemini yapabilirsin.
            </ThemedText>

            <Pressable style={[styles.uploadButton, { backgroundColor: Colors[colorScheme].tint }]} onPress={pickImage}>
              <Ionicons name="image-outline" size={18} color={tintTextColor} />
              <ThemedText style={[styles.uploadButtonText, { color: tintTextColor }]}>Görsel Seç</ThemedText>
            </Pressable>

            <View style={styles.statusRow}>
              <Ionicons
                name={processState === 'error' ? 'alert-circle-outline' : 'information-circle-outline'}
                size={18}
                color={colors.text}
              />
              <ThemedText style={styles.statusText}>{statusMessage}</ThemedText>
            </View>

            {selectedImageName ? (
              <View style={styles.fileCard}>
                <ThemedText type="defaultSemiBold">Seçilen Dosya</ThemedText>
                <ThemedText style={styles.fileText}>{selectedImageName}</ThemedText>
                <ThemedText style={styles.fileText}>
                  {selectedImageSize ? `${selectedImageSize.width} x ${selectedImageSize.height}` : 'Boyut bilinmiyor'}
                </ThemedText>
              </View>
            ) : null}
          </View>

          <View
            style={[
              styles.panel,
              styles.previewPanel,
              {
                backgroundColor: colorScheme === 'dark' ? '#181C1E' : '#FFFFFF',
                borderColor: colorScheme === 'dark' ? '#32383B' : '#D7E0EA',
              },
            ]}>
            <ThemedText type="subtitle">Orta Önizleme</ThemedText>
            {selectedImageUri ? (
              <View style={styles.previewBox}>
                <Image source={{ uri: selectedImageUri }} style={styles.previewImage} contentFit="cover" />
                <View style={styles.previewBadge}>
                  <ThemedText style={styles.previewBadgeText}>Fotoğraf Ortada</ThemedText>
                </View>
              </View>
            ) : (
              <View style={styles.emptyPreview}>
                <Ionicons name="image-outline" size={40} color={colors.text} />
                <ThemedText style={styles.helperText}>Fotoğraf seçince burada ortada görünecek.</ThemedText>
              </View>
            )}

            <View style={styles.previewActions}>
              <Pressable
                style={[
                  styles.iconActionButton,
                  { backgroundColor: colorScheme === 'dark' ? '#202426' : '#F3F7FC', borderColor: colors.tint },
                ]}
                onPress={cropImage}
                disabled={!selectedImageUri}>
                <Ionicons name="crop-outline" size={18} color={colors.tint} />
                <ThemedText style={[styles.iconActionText, { color: colors.tint }]}>Kırp</ThemedText>
              </Pressable>

              <View style={styles.cropHintPill}>
                <ThemedText style={styles.cropHintText}>{cropApplied ? 'Kırpma uygulandı' : 'Kırpma bekliyor'}</ThemedText>
              </View>
            </View>
          </View>

          <View
            style={[
              styles.panel,
              styles.featurePanel,
              {
                backgroundColor: colorScheme === 'dark' ? '#202426' : '#F3F7FC',
                borderColor: colorScheme === 'dark' ? '#32383B' : '#D7E0EA',
              },
            ]}>
            <ThemedText type="subtitle">Özellikler</ThemedText>

            {/* Section 1: Preprocessing */}
            <ThemedText type="defaultSemiBold">1. Yüz Tespiti</ThemedText>
            <Pressable
              style={[styles.cvButton, { backgroundColor: Colors[colorScheme].tint, opacity: selectedImageUri ? 1 : 0.5 }]}
              onPress={handlePreprocess}
              disabled={!selectedImageUri || preprocessLoading}>
              {preprocessLoading
                ? <ActivityIndicator color={tintTextColor} />
                : <ThemedText style={[styles.cvButtonText, { color: tintTextColor }]}>Yüzü Tespit Et</ThemedText>}
            </Pressable>
            {preprocessError ? <Text style={styles.errorText}>{preprocessError}</Text> : null}
            {preprocessedB64 ? (
              <Pressable onPress={() => setLightboxUri(`data:image/png;base64,${preprocessedB64}`)}>
                <Image
                  source={{ uri: `data:image/png;base64,${preprocessedB64}` }}
                  style={styles.resultImage}
                  contentFit="contain"
                />
              </Pressable>
            ) : null}

            {/* Section 2: Landmarks */}
            <ThemedText type="defaultSemiBold">2. Yüz Noktaları</ThemedText>
            <Pressable
              style={[styles.cvButton, { backgroundColor: Colors[colorScheme].tint, opacity: preprocessedB64 ? 1 : 0.4 }]}
              onPress={handleLandmarks}
              disabled={!preprocessedB64 || landmarkLoading}>
              {landmarkLoading
                ? <ActivityIndicator color={tintTextColor} />
                : <ThemedText style={[styles.cvButtonText, { color: tintTextColor }]}>Noktaları Tespit Et</ThemedText>}
            </Pressable>
            {landmarkError ? <Text style={styles.errorText}>{landmarkError}</Text> : null}
            {landmarkCount != null ? (
              <View style={styles.landmarkRow}>
                <View style={{ flex: 1 }}>
                  <ThemedText style={styles.helperText}>{landmarkCount} nokta bulundu</ThemedText>
                  <ThemedText style={[styles.helperText, { fontSize: 11, opacity: 0.5 }]}>
                    Gözler, kaşlar, ağız, burun, çene ve alın noktaları
                  </ThemedText>
                </View>
                <Switch value={showLandmarks} onValueChange={setShowLandmarks} />
              </View>
            ) : null}
            {landmarkB64 && showLandmarks ? (
              <Pressable onPress={() => setLightboxUri(`data:image/png;base64,${landmarkB64}`)}>
                <Image
                  source={{ uri: `data:image/png;base64,${landmarkB64}` }}
                  style={styles.resultImage}
                  contentFit="contain"
                />
              </Pressable>
            ) : preprocessedB64 && landmarkCount != null && !showLandmarks ? (
              <Pressable onPress={() => setLightboxUri(`data:image/png;base64,${preprocessedB64}`)}>
                <Image
                  source={{ uri: `data:image/png;base64,${preprocessedB64}` }}
                  style={styles.resultImage}
                  contentFit="contain"
                />
              </Pressable>
            ) : null}

            {/* Section 3.1: Expression Transfer */}
            <ThemedText type="defaultSemiBold">3.1 İfade Transferi</ThemedText>
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

            {/* Section 3: Expression Warp */}
            <ThemedText type="defaultSemiBold">3. Yüz Deforme</ThemedText>
            <View style={styles.warpGrid}>
              {(['smile', 'raise_eyebrows', 'widen_lips', 'slim_face'] as const).map((op) => (
                <Pressable
                  key={op}
                  style={[
                    styles.warpOpButton,
                    {
                      backgroundColor: warpOp === op ? Colors[colorScheme].tint : 'rgba(120,120,120,0.15)',
                      opacity: landmarkCount ? 1 : 0.4,
                    },
                  ]}
                  onPress={() => setWarpOp(op)}
                  disabled={!landmarkCount}>
                  <ThemedText style={[styles.warpOpText, { color: warpOp === op ? tintTextColor : colors.text }]}>
                    {op === 'smile' ? 'Gülümse' : op === 'raise_eyebrows' ? 'Kaşları Kaldır' : op === 'widen_lips' ? 'Dudakları Genişlet' : 'İnce Yüz'}
                  </ThemedText>
                </Pressable>
              ))}
            </View>
            <ThemedText style={styles.helperText}>Yoğunluk: {warpIntensity.toFixed(1)}</ThemedText>
            <View style={styles.sliderRow}>
              <Pressable onPress={() => setWarpIntensity(Math.max(0, +(warpIntensity - 0.1).toFixed(1)))} style={styles.sliderBtn} disabled={!landmarkCount}>
                <ThemedText>−</ThemedText>
              </Pressable>
              <View style={styles.sliderTrack}>
                <View style={[styles.sliderFill, { width: `${warpIntensity * 100}%`, backgroundColor: Colors[colorScheme].tint }]} />
              </View>
              <Pressable onPress={() => setWarpIntensity(Math.min(1, +(warpIntensity + 0.1).toFixed(1)))} style={styles.sliderBtn} disabled={!landmarkCount}>
                <ThemedText>+</ThemedText>
              </Pressable>
            </View>
            <Pressable
              style={[styles.cvButton, { backgroundColor: Colors[colorScheme].tint, opacity: landmarkCount ? 1 : 0.4 }]}
              onPress={handleWarp}
              disabled={!landmarkCount || warpLoading}>
              {warpLoading
                ? <ActivityIndicator color={tintTextColor} />
                : <ThemedText style={[styles.cvButtonText, { color: tintTextColor }]}>Deforme Uygula</ThemedText>}
            </Pressable>
            {warpError ? <Text style={styles.errorText}>{warpError}</Text> : null}
            {warpResultB64 && preprocessedB64 ? (
              <View style={styles.sideBySide}>
                <View style={styles.sideBox}>
                  <ThemedText style={styles.sideLabel}>Orijinal</ThemedText>
                  <Pressable onPress={() => setLightboxUri(`data:image/png;base64,${preprocessedB64}`)}>
                    <Image source={{ uri: `data:image/png;base64,${preprocessedB64}` }} style={styles.sideImage} contentFit="contain" />
                  </Pressable>
                </View>
                <View style={styles.sideBox}>
                  <ThemedText style={styles.sideLabel}>Sonuç</ThemedText>
                  <Pressable onPress={() => setLightboxUri(`data:image/png;base64,${warpResultB64}`)}>
                    <Image source={{ uri: `data:image/png;base64,${warpResultB64}` }} style={styles.sideImage} contentFit="contain" />
                  </Pressable>
                </View>
              </View>
            ) : null}
            {warpResultB64 && preprocessedB64 ? renderAgeAnalysisCard() : null}

            {/* Section 4: Aging Simulation */}
            <ThemedText type="defaultSemiBold">4. Yaşlandırma / Gençleştirme</ThemedText>
            <ThemedText style={styles.helperText}>Yoğunluk: {agingIntensity.toFixed(1)}</ThemedText>
            <View style={styles.sliderRow}>
              <Pressable onPress={() => setAgingIntensity(Math.max(0, +(agingIntensity - 0.1).toFixed(1)))} style={styles.sliderBtn} disabled={!selectedImageUri}>
                <ThemedText>−</ThemedText>
              </Pressable>
              <View style={styles.sliderTrack}>
                <View style={[styles.sliderFill, { width: `${agingIntensity * 100}%`, backgroundColor: Colors[colorScheme].tint }]} />
              </View>
              <Pressable onPress={() => setAgingIntensity(Math.min(1, +(agingIntensity + 0.1).toFixed(1)))} style={styles.sliderBtn} disabled={!selectedImageUri}>
                <ThemedText>+</ThemedText>
              </Pressable>
            </View>
            <View style={styles.agingRow}>
              <Pressable
                style={[styles.cvButton, { flex: 1, backgroundColor: Colors[colorScheme].tint, opacity: preprocessedB64 ? 1 : 0.4 }]}
                onPress={() => handleAging('aging')}
                disabled={!preprocessedB64 || agingLoading}>
                {agingLoading && agingMode === 'aging'
                  ? <ActivityIndicator color={tintTextColor} />
                  : <ThemedText style={[styles.cvButtonText, { color: tintTextColor }]}>Yaşlandır</ThemedText>}
              </Pressable>
              <Pressable
                style={[styles.cvButton, { flex: 1, backgroundColor: colors.text, opacity: preprocessedB64 ? 1 : 0.4 }]}
                onPress={() => handleAging('deaging')}
                disabled={!preprocessedB64 || agingLoading}>
                {agingLoading && agingMode === 'deaging'
                  ? <ActivityIndicator color={colorScheme === 'dark' ? '#000' : '#fff'} />
                  : <ThemedText style={[styles.cvButtonText, { color: colorScheme === 'dark' ? '#000' : '#fff' }]}>Gençleştir</ThemedText>}
              </Pressable>
            </View>
            {agingError ? <Text style={styles.errorText}>{agingError}</Text> : null}
            {agingResultB64 && preprocessedB64 ? (
              <View style={styles.sideBySide}>
                <View style={styles.sideBox}>
                  <ThemedText style={styles.sideLabel}>Orijinal</ThemedText>
                  <Pressable onPress={() => setLightboxUri(`data:image/png;base64,${preprocessedB64}`)}>
                    <Image source={{ uri: `data:image/png;base64,${preprocessedB64}` }} style={styles.sideImage} contentFit="contain" />
                  </Pressable>
                </View>
                <View style={styles.sideBox}>
                  <ThemedText style={styles.sideLabel}>Sonuç</ThemedText>
                  <Pressable onPress={() => setLightboxUri(`data:image/png;base64,${agingResultB64}`)}>
                    <Image source={{ uri: `data:image/png;base64,${agingResultB64}` }} style={styles.sideImage} contentFit="contain" />
                  </Pressable>
                </View>
              </View>
            ) : null}
            {agingResultB64 && preprocessedB64 ? renderAgeAnalysisCard() : null}

            {/* Section 5: Pro Lab */}
            <ThemedText type="defaultSemiBold">5. Pro Lab (Canlı)</ThemedText>
            <ThemedText style={styles.helperText}>Operasyon: {PRO_LABEL[proOperation]}</ThemedText>
            <View style={styles.warpGrid}>
              {PRO_OPERATIONS.map((op) => (
                <Pressable
                  key={op}
                  style={[
                    styles.warpOpButton,
                    {
                      backgroundColor: proOperation === op ? Colors[colorScheme].tint : 'rgba(120,120,120,0.15)',
                      opacity: preprocessedB64 ? 1 : 0.4,
                    },
                  ]}
                  onPress={() => setProOperation(op)}
                  disabled={!preprocessedB64}>
                  <ThemedText style={[styles.warpOpText, { color: proOperation === op ? tintTextColor : colors.text }]}>
                    {PRO_LABEL[op]}
                  </ThemedText>
                </Pressable>
              ))}
            </View>

            <View style={styles.presetRow}>
              {(['natural', 'balanced', 'strong'] as ProPreset[]).map((preset) => (
                <Pressable
                  key={preset}
                  style={[
                    styles.presetButton,
                    {
                      backgroundColor: proPreset === preset ? Colors[colorScheme].tint : 'rgba(120,120,120,0.15)',
                      opacity: preprocessedB64 ? 1 : 0.45,
                    },
                  ]}
                  onPress={() => applyProPreset(preset)}
                  disabled={!preprocessedB64}>
                  <ThemedText style={[styles.warpOpText, { color: proPreset === preset ? tintTextColor : colors.text }]}>
                    {PRO_PRESET_LABEL[preset]}
                  </ThemedText>
                </Pressable>
              ))}
            </View>

            <ThemedText style={styles.helperText}>Intensity: {proIntensity.toFixed(2)}</ThemedText>
            <Slider
              style={styles.nativeSlider}
              minimumValue={0}
              maximumValue={1}
              step={0.01}
              value={proIntensity}
              onValueChange={setProIntensity}
              minimumTrackTintColor={colors.tint}
              maximumTrackTintColor="rgba(120,120,120,0.25)"
              thumbTintColor={colors.tint}
              disabled={!preprocessedB64}
            />

            <ThemedText style={styles.helperText}>RBF Smooth: {proRbfSmooth.toFixed(1)}</ThemedText>
            <Slider
              style={styles.nativeSlider}
              minimumValue={0.8}
              maximumValue={10}
              step={0.1}
              value={proRbfSmooth}
              onValueChange={setProRbfSmooth}
              minimumTrackTintColor={colors.tint}
              maximumTrackTintColor="rgba(120,120,120,0.25)"
              thumbTintColor={colors.tint}
              disabled={!preprocessedB64}
            />

            {proLoading ? <ActivityIndicator /> : null}
            {proError ? <Text style={styles.errorText}>{proError}</Text> : null}

            {proResultB64 && preprocessedB64 ? (
              <View style={styles.compareCard}>
                <View style={styles.compareHeaderRow}>
                  <ThemedText style={styles.sideLabel}>Pro Sonuç</ThemedText>
                  <View style={styles.compareHintPill}>
                    <ThemedText style={styles.compareHintText}>Basılı tut: Orijinal</ThemedText>
                  </View>
                </View>

                <Pressable
                  style={styles.compareStage}
                  onPress={() => setLightboxUri(`data:image/png;base64,${proResultB64}`)}>
                  <Image source={{ uri: `data:image/png;base64,${proResultB64}` }} style={styles.compareImage} contentFit="contain" />

                  {preprocessedB64 ? (
                    <Animated.View
                      pointerEvents="none"
                      style={[
                        styles.compareOverlay,
                        {
                          opacity: proCompareOpacity,
                        },
                      ]}>
                      <Image source={{ uri: `data:image/png;base64,${preprocessedB64}` }} style={styles.compareImage} contentFit="contain" />
                      <View style={styles.originalTag}>
                        <ThemedText style={styles.originalTagText}>Original</ThemedText>
                      </View>
                    </Animated.View>
                  ) : null}

                  <Pressable
                    style={styles.compareFab}
                    onPressIn={() => setProCompareHeld(true)}
                    onPressOut={() => setProCompareHeld(false)}
                    onPress={() => null}>
                    <Ionicons name="swap-horizontal" size={18} color="#fff" />
                  </Pressable>
                </Pressable>
              </View>
            ) : null}

            {proResultB64 && preprocessedB64 ? renderAgeAnalysisCard() : null}

            {/* Section 6: Manual Makeup */}
            <ThemedText type="defaultSemiBold">6. Manual Makeup</ThemedText>
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

            {makeupResultB64 && preprocessedB64 ? (
              <View style={styles.compareCard}>
                <View style={styles.compareHeaderRow}>
                  <ThemedText style={styles.sideLabel}>Makeup Sonuç</ThemedText>
                  <View style={styles.compareHintPill}>
                    <ThemedText style={styles.compareHintText}>Basılı tut: Orijinal</ThemedText>
                  </View>
                </View>

                <Pressable style={styles.compareStage} onPress={() => setLightboxUri(`data:image/png;base64,${makeupResultB64}`)}>
                  <Image source={{ uri: `data:image/png;base64,${makeupResultB64}` }} style={styles.compareImage} contentFit="contain" />

                  {preprocessedB64 ? (
                    <Animated.View
                      pointerEvents="none"
                      style={[
                        styles.compareOverlay,
                        {
                          opacity: proCompareOpacity,
                        },
                      ]}>
                      <Image source={{ uri: `data:image/png;base64,${preprocessedB64}` }} style={styles.compareImage} contentFit="contain" />
                      <View style={styles.originalTag}>
                        <ThemedText style={styles.originalTagText}>Original</ThemedText>
                      </View>
                    </Animated.View>
                  ) : null}

                  <Pressable
                    style={styles.compareFab}
                    onPressIn={() => setProCompareHeld(true)}
                    onPressOut={() => setProCompareHeld(false)}
                    onPress={() => null}>
                    <Ionicons name="swap-horizontal" size={18} color="#fff" />
                  </Pressable>
                </Pressable>
              </View>
            ) : null}

            {evalMetrics ? (
              <View style={styles.metricsCard}>
                <ThemedText type="defaultSemiBold">Quantitative Evaluation</ThemedText>
                {evalSourceLabel ? <ThemedText style={styles.helperText}>Source: {evalSourceLabel}</ThemedText> : null}

                <View style={styles.metricTableHeaderRow}>
                  <Text style={[styles.metricHeaderCell, styles.metricHeaderMetric]}>Metric</Text>
                  <Text style={[styles.metricHeaderCell, styles.metricHeaderValue]}>Value</Text>
                  <Text style={[styles.metricHeaderCell, styles.metricHeaderPurpose]}>Purpose/Acceptable Range</Text>
                </View>

                {metricTableRows.map((row) => (
                  <View key={row.metric} style={styles.metricTableDataRow}>
                    <Text style={[styles.metricDataCell, styles.metricDataMetric]}>{row.metric}</Text>
                    <Text style={[styles.metricDataCell, styles.metricDataValue]}>{row.value}</Text>
                    <Text style={[styles.metricDataCell, styles.metricDataPurpose]}>{row.purposeRange}</Text>
                  </View>
                ))}

                {proMetrics?.hf_lf_ratio_before != null ? (
                  <ThemedText style={styles.helperText}>HF/LF Before: {proMetrics.hf_lf_ratio_before.toFixed(4)}</ThemedText>
                ) : null}
                {proMetrics?.hf_lf_ratio_after != null ? (
                  <ThemedText style={styles.helperText}>HF/LF After: {proMetrics.hf_lf_ratio_after.toFixed(4)}</ThemedText>
                ) : null}
                {proMetrics?.hf_lf_ratio_delta != null ? (
                  <ThemedText style={styles.helperText}>HF/LF Delta: {proMetrics.hf_lf_ratio_delta.toFixed(4)}</ThemedText>
                ) : null}
              </View>
            ) : null}

            {spectrumBeforeB64 && spectrumAfterB64 ? (
              <View style={styles.sideBySide}>
                <View style={styles.sideBox}>
                  <ThemedText style={styles.sideLabel}>Spectrum Before</ThemedText>
                  <Pressable onPress={() => setLightboxUri(`data:image/png;base64,${spectrumBeforeB64}`)}>
                    <Image source={{ uri: `data:image/png;base64,${spectrumBeforeB64}` }} style={styles.sideImage} contentFit="contain" />
                  </Pressable>
                </View>
                <View style={styles.sideBox}>
                  <ThemedText style={styles.sideLabel}>Spectrum After</ThemedText>
                  <Pressable onPress={() => setLightboxUri(`data:image/png;base64,${spectrumAfterB64}`)}>
                    <Image source={{ uri: `data:image/png;base64,${spectrumAfterB64}` }} style={styles.sideImage} contentFit="contain" />
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
          </View>
        </View>
      </ScrollView>
      </View>

      <Modal visible={!!lightboxUri} animationType="fade" transparent onRequestClose={() => setLightboxUri(null)}>
        <Pressable style={styles.lightboxBackdrop} onPress={() => setLightboxUri(null)}>
          {lightboxUri ? (
            <Image source={{ uri: lightboxUri }} style={styles.lightboxImage} contentFit="contain" />
          ) : null}
          <Pressable style={styles.lightboxClose} onPress={() => setLightboxUri(null)}>
            <Ionicons name="close" size={24} color="#fff" />
          </Pressable>
        </Pressable>
      </Modal>

      <Modal visible={cropEditorVisible} animationType="slide" transparent onRequestClose={closeCropEditor}>
        <View style={styles.cropModalBackdrop}>
          <ScrollView
            style={styles.cropModalScroll}
            contentContainerStyle={styles.cropModalScrollContent}
            showsVerticalScrollIndicator={false}
            bounces={false}>
            <View
              style={[
                styles.cropModalCard,
                {
                  backgroundColor: colorScheme === 'dark' ? '#181C1E' : '#FFFFFF',
                  borderColor: colorScheme === 'dark' ? '#32383B' : '#D7E0EA',
                },
              ]}>
              <View style={styles.cropModalHeader}>
                <Pressable onPress={closeCropEditor} style={styles.cropModalHeaderButton}>
                  <Ionicons name="close" size={22} color={colors.text} />
                </Pressable>

                <ThemedText type="subtitle">Kırpma Alanı</ThemedText>

                <Pressable
                  onPress={applyCrop}
                  style={[styles.cropApplyButton, { backgroundColor: Colors[colorScheme].tint }]}
                  disabled={!cropBox}>
                  <ThemedText style={styles.cropApplyButtonText}>Uygula</ThemedText>
                </Pressable>
              </View>

              <View
                style={styles.cropStage}
                onLayout={(event) => {
                  const { width: stageWidth, height: stageHeight } = event.nativeEvent.layout;
                  const nextLayout = { width: stageWidth, height: stageHeight };
                  setCropStageLayout(nextLayout);
                  if (!cropBoxRef.current && selectedImageSize) {
                    updateCropBox(createInitialCropBox(nextLayout, selectedImageSize));
                  }
                }}>
                {selectedImageUri ? <Image source={{ uri: selectedImageUri }} style={styles.cropStageImage} contentFit="contain" /> : null}

                {cropBox && cropStageLayout && selectedImageSize ? (
                  <View style={[StyleSheet.absoluteFill, { pointerEvents: 'box-none' }]}>
                    <View
                      style={[
                        styles.cropBoxFrame,
                        {
                          left: cropBox.x,
                          top: cropBox.y,
                          width: cropBox.width,
                          height: cropBox.height,
                        },
                      ]}
                      {...moveCropResponder.panHandlers}>
                      <View style={styles.cropBoxShade} />
                      <View style={styles.cropBoxInnerBorder} />
                    <View style={styles.cropRightHandle} {...resizeRightResponder.panHandlers} />
                    <View style={styles.cropBottomHandle} {...resizeBottomResponder.panHandlers} />
                    <View style={styles.cropCornerHandle} {...resizeCornerResponder.panHandlers} />
                    </View>
                  </View>
                ) : null}
              </View>

              <View style={styles.cropModalFooter}>
                <ThemedText style={styles.helperText}>
                  Alanı sürükleyerek konumlandırabilir, sağ tutamactan genişliği, alt tutamactan yüksekliği değiştirebilirsin.
                </ThemedText>
                <ThemedText style={styles.helperText}>
                  Köşe tutamaci ile her iki boyutu birden ayarlayabilirsin; alan kare olmak zorunda degil.
                </ThemedText>
              </View>
            </View>
          </ScrollView>
        </View>
      </Modal>
    </ThemedView>
  );
}

const styles = StyleSheet.create({
  screen: {
    flex: 1,
    flexDirection: 'row',
  },
  mainContent: {
    flex: 1,
  },
  header: {
    paddingTop: 56,
    paddingHorizontal: 16,
    paddingBottom: 16,
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'space-between',
  },
  content: {
    flexGrow: 1,
    paddingHorizontal: 16,
    paddingBottom: 24,
    paddingTop: 8,
  },
  workspace: {
    gap: 14,
    width: '100%',
    alignSelf: 'center',
    maxWidth: 1240,
  },
  workspaceWide: {
    flexDirection: 'row',
    alignItems: 'stretch',
    justifyContent: 'center',
  },
  panel: {
    borderWidth: 1,
    borderRadius: 18,
    padding: 16,
    gap: 12,
  },
  uploadPanel: {
    flex: 1,
    minWidth: 240,
  },
  previewPanel: {
    flex: 1.3,
    minWidth: 320,
    alignItems: 'center',
  },
  featurePanel: {
    flex: 1,
    minWidth: 240,
  },
  helperText: {
    opacity: 0.8,
    lineHeight: 21,
  },
  uploadButton: {
    height: 44,
    borderRadius: 22,
    paddingHorizontal: 16,
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
    padding: 12,
    backgroundColor: 'rgba(120,120,120,0.08)',
    gap: 4,
  },
  fileText: {
    opacity: 0.8,
  },
  previewBox: {
    width: '100%',
    maxWidth: 560,
    aspectRatio: 1,
    borderRadius: 16,
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
    backgroundColor: 'rgba(0,0,0,0.45)',
    borderRadius: 10,
    paddingHorizontal: 10,
    paddingVertical: 6,
  },
  previewBadgeText: {
    color: '#FFFFFF',
    fontSize: 12,
    fontWeight: '700',
  },
  emptyPreview: {
    width: '100%',
    maxWidth: 560,
    aspectRatio: 1,
    borderRadius: 16,
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
  iconActionButton: {
    minHeight: 40,
    borderWidth: 1,
    borderRadius: 999,
    paddingHorizontal: 14,
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
  cropModalBackdrop: {
    flex: 1,
    backgroundColor: 'rgba(0,0,0,0.65)',
  },
  cropModalScroll: {
    flex: 1,
  },
  cropModalScrollContent: {
    padding: 16,
    minHeight: '100%',
    justifyContent: 'center',
  },
  cropModalCard: {
    width: '100%',
    maxWidth: 760,
    borderWidth: 1,
    borderRadius: 24,
    overflow: 'hidden',
    alignSelf: 'center',
    maxHeight: '96%',
  },
  cropModalHeader: {
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'space-between',
    gap: 12,
    paddingHorizontal: 16,
    paddingVertical: 14,
    borderBottomWidth: 1,
    borderBottomColor: 'rgba(120,120,120,0.18)',
  },
  cropModalHeaderButton: {
    width: 36,
    height: 36,
    borderRadius: 18,
    alignItems: 'center',
    justifyContent: 'center',
    backgroundColor: 'rgba(120,120,120,0.12)',
  },
  cropApplyButton: {
    minWidth: 88,
    height: 36,
    borderRadius: 18,
    alignItems: 'center',
    justifyContent: 'center',
    paddingHorizontal: 14,
  },
  cropApplyButtonText: {
    color: '#FFFFFF',
    fontWeight: '700',
    fontSize: 13,
  },
  cropStage: {
    width: '100%',
    aspectRatio: 1,
    backgroundColor: '#101214',
    position: 'relative',
  },
  cropStageImage: {
    ...StyleSheet.absoluteFillObject,
  },
  cropBoxFrame: {
    position: 'absolute',
    borderRadius: 18,
    borderWidth: 2,
    borderColor: Colors.light.tint,
    backgroundColor: 'rgba(122, 162, 255, 0.08)',
  },
  cropBoxShade: {
    ...StyleSheet.absoluteFillObject,
    borderRadius: 16,
  },
  cropBoxInnerBorder: {
    ...StyleSheet.absoluteFillObject,
    borderRadius: 16,
    borderWidth: 1,
    borderColor: 'rgba(255,255,255,0.8)',
  },
  cropCornerHandle: {
    position: 'absolute',
    right: -6,
    bottom: -6,
    width: 42,
    height: 42,
    borderRadius: 21,
    borderWidth: 2,
    borderColor: '#FFFFFF',
    backgroundColor: Colors.light.tint,
  },
  cropRightHandle: {
    position: 'absolute',
    right: -8,
    top: '28%',
    width: 24,
    height: 74,
    borderRadius: 12,
    borderWidth: 2,
    borderColor: '#FFFFFF',
    backgroundColor: Colors.light.tint,
  },
  cropBottomHandle: {
    position: 'absolute',
    left: '28%',
    bottom: -8,
    width: 74,
    height: 24,
    borderRadius: 12,
    borderWidth: 2,
    borderColor: '#FFFFFF',
    backgroundColor: Colors.light.tint,
  },
  cropModalFooter: {
    paddingHorizontal: 16,
    paddingVertical: 14,
    gap: 6,
  },
  cvButton: {
    height: 42,
    borderRadius: 21,
    alignItems: 'center',
    justifyContent: 'center',
    paddingHorizontal: 16,
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
    borderRadius: 12,
    backgroundColor: 'rgba(120,120,120,0.1)',
  },
  landmarkRow: {
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'space-between',
  },
  warpGrid: {
    flexDirection: 'row',
    flexWrap: 'wrap',
    gap: 8,
  },
  warpOpButton: {
    flex: 1,
    minWidth: '40%',
    paddingVertical: 10,
    borderRadius: 10,
    alignItems: 'center',
  },
  warpOpText: {
    fontSize: 12,
    fontWeight: '600',
  },
  presetRow: {
    flexDirection: 'row',
    gap: 8,
  },
  presetButton: {
    flex: 1,
    minHeight: 38,
    borderRadius: 10,
    alignItems: 'center',
    justifyContent: 'center',
    paddingHorizontal: 8,
  },
  sliderRow: {
    flexDirection: 'row',
    alignItems: 'center',
    gap: 8,
  },
  sliderBtn: {
    width: 32,
    height: 32,
    borderRadius: 16,
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
  metricsCard: {
    borderRadius: 12,
    borderWidth: 1,
    borderColor: 'rgba(120,120,120,0.2)',
    backgroundColor: 'rgba(120,120,120,0.08)',
    padding: 10,
    gap: 8,
  },
  metricTableHeaderRow: {
    flexDirection: 'row',
    borderWidth: 1,
    borderColor: 'rgba(120,120,120,0.24)',
    backgroundColor: 'rgba(0,0,0,0.08)',
    borderRadius: 8,
    paddingVertical: 8,
    paddingHorizontal: 6,
  },
  metricTableDataRow: {
    flexDirection: 'row',
    borderWidth: 1,
    borderColor: 'rgba(120,120,120,0.16)',
    borderRadius: 8,
    paddingVertical: 8,
    paddingHorizontal: 6,
    backgroundColor: 'rgba(255,255,255,0.04)',
  },
  metricHeaderCell: {
    fontSize: 11,
    fontWeight: '700',
  },
  metricDataCell: {
    fontSize: 11,
    opacity: 0.92,
  },
  metricHeaderMetric: {
    flex: 0.9,
  },
  metricHeaderValue: {
    flex: 1.05,
  },
  metricHeaderPurpose: {
    flex: 2.55,
  },
  metricDataMetric: {
    flex: 0.9,
    fontWeight: '700',
  },
  metricDataValue: {
    flex: 1.05,
    fontWeight: '600',
  },
  metricDataPurpose: {
    flex: 2.55,
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
  lightboxImage: {
    width: '100%',
    height: '100%',
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
});
