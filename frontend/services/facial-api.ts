import { Platform } from 'react-native';

const API_BASE = (process.env.EXPO_PUBLIC_API_URL ?? 'http://localhost:3000').replace(/\/$/, '');

const BASE64_ALPHABET = 'ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/';
const BASE64_INVALID = 255;
const BASE64_LOOKUP = (() => {
  const lookup = new Uint8Array(256);
  lookup.fill(BASE64_INVALID);
  for (let i = 0; i < BASE64_ALPHABET.length; i++) {
    lookup[BASE64_ALPHABET.charCodeAt(i)] = i;
  }
  return lookup;
})();

function decodeBase64ToBytes(input: string): Uint8Array {
  const normalized = input.replace(/^data:.*;base64,/, '').replace(/\s+/g, '');
  if (normalized.length === 0) {
    return new Uint8Array(0);
  }
  if (normalized.length % 4 !== 0) {
    throw new Error('Invalid base64 string length.');
  }

  if (typeof globalThis.atob === 'function') {
    const binary = globalThis.atob(normalized);
    const bytes = new Uint8Array(binary.length);
    for (let i = 0; i < binary.length; i++) {
      bytes[i] = binary.charCodeAt(i);
    }
    return bytes;
  }

  const padding = normalized.endsWith('==') ? 2 : normalized.endsWith('=') ? 1 : 0;
  const outputLength = (normalized.length / 4) * 3 - padding;
  const bytes = new Uint8Array(outputLength);

  let out = 0;
  for (let i = 0; i < normalized.length; i += 4) {
    const c1 = normalized.charCodeAt(i);
    const c2 = normalized.charCodeAt(i + 1);
    const c3 = normalized.charCodeAt(i + 2);
    const c4 = normalized.charCodeAt(i + 3);

    const v1 = BASE64_LOOKUP[c1];
    const v2 = BASE64_LOOKUP[c2];
    const v3 = c3 === 61 ? 0 : BASE64_LOOKUP[c3];
    const v4 = c4 === 61 ? 0 : BASE64_LOOKUP[c4];

    if (
      v1 === BASE64_INVALID ||
      v2 === BASE64_INVALID ||
      (c3 !== 61 && v3 === BASE64_INVALID) ||
      (c4 !== 61 && v4 === BASE64_INVALID)
    ) {
      throw new Error('Invalid base64 character encountered.');
    }

    const triple = (v1 << 18) | (v2 << 12) | (v3 << 6) | v4;

    if (out < outputLength) bytes[out++] = (triple >> 16) & 0xff;
    if (out < outputLength) bytes[out++] = (triple >> 8) & 0xff;
    if (out < outputLength) bytes[out++] = triple & 0xff;
  }

  return bytes;
}

function base64ToBlob(b64: string, mimeType = 'image/png'): Blob {
  const bytes = decodeBase64ToBytes(b64);
  const buffer = new ArrayBuffer(bytes.byteLength);
  new Uint8Array(buffer).set(bytes);
  return new Blob([buffer], { type: mimeType });
}

async function requestJson(endpoint: string, body: FormData): Promise<any> {
  const cleanEndpoint = endpoint.replace(/^\/api/, '');
  const response = await fetch(`${API_BASE}${cleanEndpoint}`, { method: 'POST', body });

  if (!response.ok) {
    const errorText = await response.text();
    throw new Error(errorText || `HTTP ${response.status}`);
  }

  const contentType = response.headers.get('content-type') ?? '';
  if (!contentType.includes('application/json')) {
    const responseText = await response.text();
    throw new Error(responseText || 'Expected JSON response');
  }

  return response.json();
}

function parseFilename(contentDisposition: string | null, fallback: string): string {
  if (!contentDisposition) {
    return fallback;
  }

  const match = /filename\*=UTF-8''([^;]+)|filename="?([^";]+)"?/i.exec(contentDisposition);
  const encodedName = match?.[1] || match?.[2];
  if (!encodedName) {
    return fallback;
  }

  try {
    return decodeURIComponent(encodedName);
  } catch {
    return encodedName;
  }
}

async function requestDownload(endpoint: string, body: FormData, fallbackName: string): Promise<{
  bytes: ArrayBuffer;
  filename: string;
  mimeType: string;
}> {
  const cleanEndpoint = endpoint.replace(/^\/api/, '');
  const response = await fetch(`${API_BASE}${cleanEndpoint}`, { method: 'POST', body });

  if (!response.ok) {
    const errorText = await response.text();
    throw new Error(errorText || `HTTP ${response.status}`);
  }

  return {
    bytes: await response.arrayBuffer(),
    filename: parseFilename(response.headers.get('content-disposition'), fallbackName),
    mimeType: response.headers.get('content-type') ?? 'application/octet-stream',
  };
}

async function toImageFilePart(uri: string): Promise<Blob | { uri: string; name: string; type: string }> {
  if (Platform.OS === 'web') {
    try {
      // Try to fetch the URI (works for blob: and data: URLs)
      const blobRes = await fetch(uri);
      if (!blobRes.ok) {
        throw new Error(`HTTP ${blobRes.status}`);
      }
      return blobRes.blob();
    } catch (error) {
      // If fetch fails, provide a more helpful error message
      console.error('Failed to fetch image from URI:', uri, error);
      throw new Error(
        `Could not load image. If using file input, make sure the file is a valid image. URI: ${uri.substring(0, 50)}...`
      );
    }
  }

  const filename = uri.split('/').pop()?.split('?')[0] ?? 'image.jpg';
  const ext = filename.split('.').pop()?.toLowerCase() ?? 'jpg';
  const mimeType = ext === 'png' ? 'image/png' : 'image/jpeg';
  return { uri, name: filename, type: mimeType };
}

async function toBase64ImagePart(imageBase64: string): Promise<Blob> {
  return base64ToBlob(imageBase64);
}

export async function preprocessFromUri(uri: string): Promise<any> {
  const formData = new FormData();
  const imagePart = await toImageFilePart(uri);

  if (Platform.OS === 'web') {
    const blob = imagePart as Blob;
    const ext = blob.type === 'image/png' ? 'png' : 'jpg';
    formData.append('image', blob, `image.${ext}`);
  } else {
    formData.append('image', imagePart as any);
  }

  return requestJson('/api/preprocess', formData);
}

export async function landmarksFromBase64(
  imageBase64: string,
  options?: { landmarkBackend?: 'mediapipe' | 'dlib' | 'hybrid' }
): Promise<any> {
  const formData = new FormData();
  formData.append('image', base64ToBlob(imageBase64), 'image.png');
  formData.append('landmark_backend', options?.landmarkBackend ?? 'hybrid');
  return requestJson('/api/landmarks', formData);
}

export async function warpFromBase64(imageBase64: string, operation: string, intensity: number): Promise<any> {
  const formData = new FormData();
  formData.append('image', base64ToBlob(imageBase64), 'image.png');
  formData.append('operation', operation);
  formData.append('intensity', String(intensity));
  return requestJson('/api/warp', formData);
}

export async function frequencyFromBase64(imageBase64: string, mode: 'aging' | 'deaging', intensity: number): Promise<any> {
  const formData = new FormData();
  formData.append('image', base64ToBlob(imageBase64), 'image.png');
  formData.append('mode', mode);
  formData.append('intensity', String(intensity));
  return requestJson('/api/frequency', formData);
}

export async function estimateAgeFromUri(uri: string): Promise<any> {
  const formData = new FormData();
  const imagePart = await toImageFilePart(uri);

  if (Platform.OS === 'web') {
    formData.append('image', imagePart as Blob, 'image.png');
  } else {
    formData.append('image', imagePart as any);
  }

  return requestJson('/api/estimate-age', formData);
}

export async function estimateAgeFromBase64(imageBase64: string): Promise<any> {
  const formData = new FormData();
  formData.append('image', await toBase64ImagePart(imageBase64), 'image.png');
  return requestJson('/api/estimate-age', formData);
}

export async function transferExpressionFromBase64(
  imageBase64: string,
  referenceImageUri: string,
  intensity: number,
  options?: {
    landmarkBackend?: 'mediapipe' | 'dlib' | 'hybrid';
  }
): Promise<any> {
  const formData = new FormData();
  const referencePart = await toImageFilePart(referenceImageUri);

  formData.append('image', base64ToBlob(imageBase64), 'image.png');

  if (Platform.OS === 'web') {
    const referenceBlob = referencePart as Blob;
    const referenceExt = referenceBlob.type === 'image/png' ? 'png' : 'jpg';
    formData.append('reference_image', referenceBlob, `reference.${referenceExt}`);
  } else {
    formData.append('reference_image', referencePart as any);
  }

  formData.append('intensity', String(intensity));
  formData.append('landmark_backend', options?.landmarkBackend ?? 'hybrid');
  return requestJson('/api/expression/transfer', formData);
}

export type ProMetrics = {
  mse: number;
  psnr: number;
  ssim: number;
  total_spectral_energy_before?: number;
  total_spectral_energy_after?: number;
  total_spectral_energy_delta?: number;
  low_frequency_energy_before?: number;
  low_frequency_energy_after?: number;
  high_frequency_energy_before?: number;
  high_frequency_energy_after?: number;
  hf_lf_ratio_before?: number;
  hf_lf_ratio_after?: number;
  hf_lf_ratio_delta?: number;
};

export type ProWarpOperation = 'smile_enhancement' | 'brow_lift' | 'lip_plump' | 'slim_face';

export async function exportEvaluationReportFromBase64(
  format: 'csv' | 'pdf',
  operation: string,
  originalImageBase64: string,
  transformedImageBase64: string,
  metrics: Pick<ProMetrics, 'mse' | 'psnr' | 'ssim'>
): Promise<any> {
  const formData = new FormData();
  formData.append('format', format);
  formData.append('operation', operation);
  formData.append('original_image_b64', originalImageBase64);
  formData.append('transformed_image_b64', transformedImageBase64);
  formData.append('metrics_json', JSON.stringify(metrics));
  return requestJson('/api/report/export', formData);
}

export async function warpProFromBase64(
  imageBase64: string,
  operation: ProWarpOperation,
  intensity: number,
  rbfSmooth: number,
  options?: {
    landmarkBackend?: 'mediapipe' | 'dlib' | 'hybrid';
    temporalSmoothing?: boolean;
    emaAlpha?: number;
    streamId?: string;
  }
): Promise<any> {
  const formData = new FormData();
  formData.append('image', base64ToBlob(imageBase64), 'image.png');
  formData.append('operation', operation);
  formData.append('intensity', String(intensity));
  formData.append('rbf_smooth', String(rbfSmooth));
  formData.append('landmark_backend', options?.landmarkBackend ?? 'hybrid');
  formData.append('temporal_smoothing', String(options?.temporalSmoothing ?? false));
  formData.append('ema_alpha', String(options?.emaAlpha ?? 0.62));
  formData.append('stream_id', options?.streamId ?? 'default');
  return requestJson('/api/warp/pro', formData);
}

export async function frequencyProFromBase64(
  imageBase64: string,
  mode: 'aging' | 'deaging',
  intensity: number,
  options?: {
    landmarkBackend?: 'mediapipe' | 'dlib' | 'hybrid';
    temporalSmoothing?: boolean;
    emaAlpha?: number;
    streamId?: string;
  }
): Promise<any> {
  const formData = new FormData();
  formData.append('image', base64ToBlob(imageBase64), 'image.png');
  formData.append('mode', mode);
  formData.append('intensity', String(intensity));
  formData.append('landmark_backend', options?.landmarkBackend ?? 'hybrid');
  formData.append('temporal_smoothing', String(options?.temporalSmoothing ?? false));
  formData.append('ema_alpha', String(options?.emaAlpha ?? 0.62));
  formData.append('stream_id', options?.streamId ?? 'default');
  return requestJson('/api/frequency/pro', formData);
}

export type AgingCompareResult = {
  success: boolean;
  message?: string;
  mode: 'aging' | 'deaging';
  intensity: number;
  age_estimation: {
    before: number;
    detected: boolean;
    after_frequency: number;
    after_ai: number | null;
  };
  frequency_based: {
    result_image_b64: string;
    metrics: { mse: number; psnr: number; ssim: number };
  };
  ai_guided: {
    success: boolean;
    landmark_info: unknown | null;
    result_image_b64: string | null;
    metrics: { mse: number; psnr: number; ssim: number };
  };
  comparison: {
    mse_delta: number | null;
    psnr_delta: number | null;
    ssim_delta: number | null;
    winner: 'frequency_based' | 'ai_guided' | 'tie' | null;
  };
};

export async function agingCompareFromBase64(
  imageBase64: string,
  mode: 'aging' | 'deaging',
  intensity: number,
  options?: {
    landmarkBackend?: 'mediapipe' | 'dlib' | 'hybrid';
  }
): Promise<AgingCompareResult> {
  const formData = new FormData();
  formData.append('image', base64ToBlob(imageBase64), 'image.png');
  formData.append('mode', mode);
  formData.append('intensity', String(intensity));
  formData.append('landmark_backend', options?.landmarkBackend ?? 'hybrid');
  return requestJson('/api/aging/compare', formData);
}

export async function aiGuidedAgingFromBase64(
  imageBase64: string,
  mode: 'aging' | 'deaging',
  intensity: number,
  options?: {
    landmarkBackend?: 'mediapipe' | 'dlib' | 'hybrid';
  }
): Promise<any> {
  const formData = new FormData();
  formData.append('image', base64ToBlob(imageBase64), 'image.png');
  formData.append('mode', mode);
  formData.append('intensity', String(intensity));
  formData.append('landmark_backend', options?.landmarkBackend ?? 'hybrid');
  return requestJson('/api/aging/ai', formData);
}

export async function agingSamFromBase64(
  imageBase64: string,
  mode: 'aging' | 'deaging',
  intensity: number,
  options?: {
    landmarkBackend?: 'mediapipe' | 'dlib' | 'hybrid';
    targetAge?: number;   // 0-100 arası; belirtilmezse backend otomatik seçer
  }
): Promise<any> {
  const formData = new FormData();
  formData.append('image', base64ToBlob(imageBase64), 'image.png');
  formData.append('mode', mode);
  formData.append('intensity', String(intensity));
  formData.append('landmark_backend', options?.landmarkBackend ?? 'hybrid');
  if (options?.targetAge !== undefined) {
    formData.append('target_age', String(options.targetAge));
  }
  return requestJson('/api/aging/sam', formData);
}

export type MakeupRegion = 'lip' | 'cheek' | 'brow' | 'lash' | 'eye' | 'teeth';

export async function applyMakeupFromBase64(
  imageBase64: string,
  region: MakeupRegion,
  hexColor: string,
  intensity: number,
  options?: {
    landmarkBackend?: 'mediapipe' | 'dlib' | 'hybrid';
    temporalSmoothing?: boolean;
    emaAlpha?: number;
    streamId?: string;
  }
): Promise<any> {
  const formData = new FormData();
  formData.append('image', base64ToBlob(imageBase64), 'image.png');
  formData.append('region', region);
  formData.append('hex_color', hexColor);
  formData.append('intensity', String(intensity));
  formData.append('landmark_backend', options?.landmarkBackend ?? 'hybrid');
  formData.append('temporal_smoothing', String(options?.temporalSmoothing ?? false));
  formData.append('ema_alpha', String(options?.emaAlpha ?? 0.62));
  formData.append('stream_id', options?.streamId ?? 'default');
  return requestJson('/api/pro/apply-makeup', formData);
}

/**
 * Professional hair recoloring.
 *
 * @param imageBase64  Source image as raw base64 (no data-URI prefix needed).
 * @param hexColor     Target hair color, e.g. "#A020F0".
 * @param intensity    0.0 – 1.0. Default 0.85.
 * @param preserveHighlights  Keep specular shine. Default true.
 */
export async function applyHairColorFromBase64(
  imageBase64: string,
  hexColor: string,
  intensity: number = 0.85,
  preserveHighlights: boolean = true,
): Promise<{
  success: boolean;
  hex_color?: string;
  intensity?: number;
  result_image_b64?: string;
  error?: string;
  details?: string;
}> {
  const formData = new FormData();
  formData.append('image', base64ToBlob(imageBase64), 'image.png');
  formData.append('hex_color', hexColor.startsWith('#') ? hexColor : `#${hexColor}`);
  formData.append('intensity', String(Math.max(0, Math.min(1, intensity))));
  formData.append('preserve_highlights', String(preserveHighlights));
  return requestJson('/api/pro/hair-color', formData);
}

export type AccessoryType = 'glasses' | 'mustache' | 'hat';
export type AccessoryStyle = 'classic' | 'round' | 'aviator' | 'heart' | 'handlebar' | 'chevron' | 'cowboy' | 'cap' | 'asian' | 'newasian' | 'pink';

export function accessoryAssetUrl(path: string): string {
  const normalized = path.replace(/^\/+/, '');
  return `${API_BASE}/assets/accessories/${normalized}`;
}

export async function applyAccessoryFromBase64(
  imageBase64: string,
  accessoryType: AccessoryType,
  style: AccessoryStyle,
  hexColor: string,
  intensity: number,
  scale: number,
  offsetX: number,
  offsetY: number,
  options?: {
    landmarkBackend?: 'mediapipe' | 'dlib' | 'hybrid';
    temporalSmoothing?: boolean;
    emaAlpha?: number;
    streamId?: string;
  }
): Promise<any> {
  const formData = new FormData();
  formData.append('image', base64ToBlob(imageBase64), 'image.png');
  formData.append('accessory_type', accessoryType);
  formData.append('style', style);
  formData.append('hex_color', hexColor);
  formData.append('intensity', String(intensity));
  formData.append('scale', String(scale));
  formData.append('offset_x', String(offsetX));
  formData.append('offset_y', String(offsetY));
  formData.append('landmark_backend', options?.landmarkBackend ?? 'hybrid');
  formData.append('temporal_smoothing', String(options?.temporalSmoothing ?? false));
  formData.append('ema_alpha', String(options?.emaAlpha ?? 0.62));
  formData.append('stream_id', options?.streamId ?? 'default');
  return requestJson('/api/pro/apply-accessory', formData);
}

export async function exportCsvReportFromBase64(params: {
  originalImageBase64: string;
  resultImageBase64: string;
  operation: string;
  intensity: number;
  ageBefore?: string | number | null;
  ageAfter?: string | number | null;
}): Promise<{ bytes: ArrayBuffer; filename: string; mimeType: string }> {
  const formData = new FormData();
  formData.append('original_image', base64ToBlob(params.originalImageBase64), 'original.png');
  formData.append('result_image', base64ToBlob(params.resultImageBase64), 'result.png');
  formData.append('operation', params.operation);
  formData.append('intensity', String(params.intensity));
  formData.append('age_before', params.ageBefore != null ? String(params.ageBefore) : '');
  formData.append('age_after', params.ageAfter != null ? String(params.ageAfter) : '');
  return requestDownload('/api/export/csv', formData, 'facial-report.csv');
}

export async function exportPdfReportFromBase64(params: {
  originalImageBase64: string;
  resultImageBase64: string;
  operation: string;
  intensity: number;
  ageBefore?: string | number | null;
  ageAfter?: string | number | null;
}): Promise<{ bytes: ArrayBuffer; filename: string; mimeType: string }> {
  const formData = new FormData();
  formData.append('original_image', base64ToBlob(params.originalImageBase64), 'original.png');
  formData.append('result_image', base64ToBlob(params.resultImageBase64), 'result.png');
  formData.append('operation', params.operation);
  formData.append('intensity', String(params.intensity));
  formData.append('age_before', params.ageBefore != null ? String(params.ageBefore) : '');
  formData.append('age_after', params.ageAfter != null ? String(params.ageAfter) : '');
  return requestDownload('/api/export/pdf', formData, 'facial-report.pdf');
}
