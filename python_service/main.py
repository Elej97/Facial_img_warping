import base64
import json
import os

import numpy as np
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, Response

from startup_check import warn_missing_ai_dependencies


def _parse_allowed_origins() -> list[str]:
    """
    ALLOWED_ORIGINS env değişkeninden izin verilen origin listesini okur.

    Örnekler (.env dosyasında):
        ALLOWED_ORIGINS=*                                          → geliştirme
        ALLOWED_ORIGINS=http://localhost:8081,http://localhost:19006
        ALLOWED_ORIGINS=https://facemorphapp.com,https://www.facemorphapp.com

    Değişken tanımlı değilse geliştirme modunda çalışır (["*"]).
    """
    raw = os.environ.get("ALLOWED_ORIGINS", "").strip()
    if not raw or raw == "*":
        return ["*"]
    origins = [o.strip() for o in raw.split(",") if o.strip()]
    return origins if origins else ["*"]


_ALLOWED_ORIGINS = _parse_allowed_origins()

# insightface and DeepFace are used ONLY for age estimation (see
# _estimate_age_from_image). DeepFace in particular pulls in TensorFlow (~2 GB
# RAM) at import time. Loading them eagerly made the whole service fail to start
# under memory pressure — even though face detection, landmarks and warping never
# touch them. So both are imported lazily, on first age-estimation request.
# Sentinel values: None = not yet attempted, False = load failed, else the object.
_deepface = None
_insight_app = None


def _get_deepface():
    """Import DeepFace on first use; cache the module (or False on failure)."""
    global _deepface
    if _deepface is None:
        try:
            from deepface import DeepFace as _DF
            _deepface = _DF
        except Exception:
            _deepface = False
    return _deepface or None


def _get_insight_app():
    """Build the insightface age/detection model on first use; cache it."""
    global _insight_app
    if _insight_app is None:
        try:
            from insightface.app import FaceAnalysis
            app_ = FaceAnalysis(
                allowed_modules=["detection", "genderage"],
                providers=["CPUExecutionProvider"],
            )
            app_.prepare(ctx_id=-1, det_size=(320, 320))
            _insight_app = app_
        except Exception:
            _insight_app = False
    return _insight_app or None

from modules.aging import apply_aging, apply_deaging
from modules.accessory_service import apply_accessory
from modules.hair_segmentation import apply_hair_color
from modules.evaluation_metrics import compute_quality_metrics
from modules.landmark import detect_landmarks, draw_landmarks
from modules.landmark_fusion import detect_landmarks_fused
from modules.expression_transfer import transfer_expression
from modules.makeup_service import apply_makeup
from modules.preprocessing import (
    detect_and_crop_face,
    normalize_face,
    validate_image,
)
from modules.evaluation_metrics import evaluate_metrics
from modules.pro_warping import (
    aging_pro,
    brow_lift_pro,
    de_aging_pro,
    plump_lips_pro,
    quality_metrics,
    slim_face_pro,
    smile_enhancement_pro,
)
from modules.reporting import build_report_payload, generate_csv_bytes, generate_csv_report, generate_pdf_bytes, generate_pdf_report
from modules.warping import raise_eyebrows, simulate_smile, slim_face, widen_lips
from utils.image_utils import b64_to_numpy, bytes_to_numpy, numpy_to_b64


warn_missing_ai_dependencies()


_AGE_FALLBACK = 30
def _estimate_age_from_image(img: np.ndarray) -> tuple[float, bool]:
    """Returns (estimated_age, is_estimated). Tries insightface → DeepFace → fallback."""
    insight_app = _get_insight_app()
    if insight_app is not None:
        try:
            faces = insight_app.get(img)
            if faces:
                age = getattr(faces[0], "age", None)
                if age is not None:
                    return float(round(float(age), 1)), True
        except Exception:
            pass

    deepface = _get_deepface()
    if deepface is not None:
        try:
            result = deepface.analyze(
                img_path=img,
                actions=["age"],
                detector_backend="opencv",
                enforce_detection=False,
                align=True,
                silent=True,
            )
            if isinstance(result, list):
                result = result[0] if result else {}
            estimated_age = result.get("age") if isinstance(result, dict) else None
            if estimated_age is not None:
                return float(round(float(estimated_age), 1)), True
        except Exception:
            pass

    return float(_AGE_FALLBACK), False

app = FastAPI(title="Facial CV Service")

app.add_middleware(
    CORSMiddleware,
    allow_origins=_ALLOWED_ORIGINS,
    allow_credentials=_ALLOWED_ORIGINS != ["*"],  # credentials sadece spesifik origin'lerde
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["Content-Type", "Authorization", "X-Requested-With"],
)


@app.exception_handler(Exception)
async def global_exception_handler(request, exc):
    return JSONResponse(status_code=500, content={"success": False, "message": str(exc)})


@app.get("/health")
async def health():
    return {"status": "ok", "service": "python-cv"}


@app.post("/preprocess")
async def preprocess(image: UploadFile = File(...)):
    file_bytes = await image.read()
    valid, err = validate_image(file_bytes, image.filename or "upload.jpg")
    if not valid:
        return {"success": False, "message": err}

    img = bytes_to_numpy(file_bytes)
    orig_h, orig_w = img.shape[:2]

    face, bbox = detect_and_crop_face(img)
    if face is None:
        return {"success": False, "message": "No face detected in image."}

    processed = normalize_face(face)
    return {
        "success": True,
        "original_size": [orig_w, orig_h],
        "face_bbox": list(bbox),
        "processed_size": [processed.shape[1], processed.shape[0]],
        "processed_image_b64": numpy_to_b64(processed),
        "message": "Face detected and preprocessed",
    }


@app.post("/estimate-age")
async def estimate_age(image: UploadFile = File(...)):
    try:
        file_bytes = await image.read()
        valid, err = validate_image(file_bytes, image.filename or "upload.jpg")
        if not valid:
            return {"error": "Face not detected or model error", "details": err}

        img = bytes_to_numpy(file_bytes)
        estimated_age, age_detected = _estimate_age_from_image(img)
    except Exception as exc:
        return {"error": "Face not detected or model error", "details": str(exc)}

    return {
        "success": True,
        "estimated_age": estimated_age,
        "model": "DeepFace" if age_detected else "fallback",
        "message": "Age estimated successfully" if age_detected else f"DeepFace unavailable; using default age {_AGE_FALLBACK}",
    }


@app.post("/aging/ai")
async def ai_guided_aging(
    image: UploadFile = File(...),
    mode: str = Form("aging"),
    intensity: float = Form(0.6),
    landmark_backend: str = Form("hybrid"),
):
    try:
        intensity = float(np.clip(intensity, 0.0, 1.0))
        file_bytes = await image.read()
        valid, err = validate_image(file_bytes, image.filename or "upload.jpg")
        if not valid:
            return {"error": "Face not detected or model error", "details": err}

        img = bytes_to_numpy(file_bytes)
        estimated_age, age_detected = _estimate_age_from_image(img)
        target_delta = 28 if mode == "aging" else -18
        target_age = max(12, int(round(estimated_age + target_delta * intensity)))
        age_gap = abs(target_age - estimated_age)
        guided_intensity = float(np.clip(0.35 + (age_gap / 32.0), 0.35, 1.0))

        lms, landmark_info = detect_landmarks_fused(img, backend=landmark_backend, temporal_smoothing=False)
        if lms is None:
            return {"error": "Face not detected or model error", "details": "No face detected."}

        if mode == "aging":
            out = aging_pro(img, lms, intensity=guided_intensity)
        elif mode == "deaging":
            out = de_aging_pro(img, lms, intensity=guided_intensity)
        else:
            return {"success": False, "message": "Unknown mode. Valid: aging, deaging"}

        estimated_age_after, _ = _estimate_age_from_image(out["result_image"])

        return {
            "success": True,
            "mode": mode,
            "model": "landmark-guided aging" if not age_detected else "DeepFace-guided aging",
            "age_detection": "deepface" if age_detected else "fallback",
            "estimated_age_before": estimated_age,
            "estimated_age_after": estimated_age_after,
            "age_delta": estimated_age_after - estimated_age,
            "target_age": target_age,
            "guided_intensity": guided_intensity,
            "landmark_info": landmark_info,
            "result_image_b64": numpy_to_b64(out["result_image"]),
            "spectrum_before_b64": numpy_to_b64(out["spectrum_before"]),
            "spectrum_after_b64": numpy_to_b64(out["spectrum_after"]),
            "metrics": out["metrics"],
        }
    except Exception as exc:
        return {"error": "Face not detected or model error", "details": str(exc)}


@app.post("/landmarks")
async def landmarks(
    image: UploadFile = File(...),
    landmark_backend: str = Form("hybrid"),
):
    try:
        file_bytes = await image.read()
        img = bytes_to_numpy(file_bytes)

        lms, landmark_info = detect_landmarks_fused(
            img,
            backend=landmark_backend,
            temporal_smoothing=False,
        )
        if lms is None:
            return {"error": "Face not detected or model error", "details": "No face detected for landmark extraction."}

        annotated = draw_landmarks(img, lms)
        return {
            "success": True,
            "landmarks": [[x, y] for x, y in lms],
            "landmark_image_b64": numpy_to_b64(annotated),
            "landmark_count": len(lms),
            "landmark_backend": landmark_backend,
            "landmark_info": landmark_info,
        }
    except Exception as exc:
        return {"error": "Face not detected or model error", "details": str(exc)}


@app.post("/expression/transfer")
async def expression_transfer(
    image: UploadFile = File(...),
    reference_image: UploadFile = File(...),
    intensity: float = Form(0.7),
    landmark_backend: str = Form("hybrid"),
):
    try:
        intensity = float(np.clip(intensity, 0.0, 1.0))

        target_bytes = await image.read()
        reference_bytes = await reference_image.read()

        valid_target, target_err = validate_image(target_bytes, image.filename or "target.jpg")
        if not valid_target:
            return {"error": "Face not detected or model error", "details": target_err}

        valid_reference, reference_err = validate_image(reference_bytes, reference_image.filename or "reference.jpg")
        if not valid_reference:
            return {"error": "Face not detected or model error", "details": reference_err}

        target_img = bytes_to_numpy(target_bytes)
        reference_img = bytes_to_numpy(reference_bytes)

        target_lms, target_info = detect_landmarks_fused(target_img, backend=landmark_backend, temporal_smoothing=False)
        reference_lms, reference_info = detect_landmarks_fused(reference_img, backend=landmark_backend, temporal_smoothing=False)

        if target_lms is None:
            return {"error": "Face not detected or model error", "details": "No face detected in target image."}
        if reference_lms is None:
            return {"error": "Face not detected or model error", "details": "No face detected in reference image."}

        transferred = transfer_expression(target_img, target_lms, reference_lms, intensity=intensity)

        return {
            "success": True,
            "intensity": intensity,
            "landmark_backend": landmark_backend,
            "target_landmark_info": target_info,
            "reference_landmark_info": reference_info,
            "result_image_b64": numpy_to_b64(transferred["result_image"]),
            "destination_landmarks": transferred["destination_landmarks"],
            "aligned_reference_landmarks": transferred["aligned_reference_landmarks"],
        }
    except Exception as exc:
        return {"error": "Face not detected or model error", "details": str(exc)}


@app.post("/warp")
async def warp(
    image: UploadFile = File(...),
    operation: str = Form("smile"),
    intensity: float = Form(0.5),
):
    try:
        requested_intensity = float(np.clip(intensity, 0.0, 1.0))
        file_bytes = await image.read()
        img = bytes_to_numpy(file_bytes)

        lms = detect_landmarks(img)
        if lms is None:
            return {"error": "Face not detected or model error", "details": "No face detected."}

        ops = {
            "smile": simulate_smile,
            "raise_eyebrows": raise_eyebrows,
            "widen_lips": widen_lips,
            "slim_face": slim_face,
        }

        if operation not in ops:
            return {"error": "Face not detected or model error", "details": f"Unknown operation '{operation}'. Valid: {list(ops.keys())}"}

        # Operation-specific caps reduce common artifacts at high intensity values.
        intensity_caps = {
            "smile": 1.0,
            "raise_eyebrows": 0.72,
            "widen_lips": 0.78,
            "slim_face": 0.68,
        }
        applied_intensity = float(min(requested_intensity, intensity_caps.get(operation, 1.0)))

        result_img = ops[operation](img, lms, applied_intensity)
        lms_after = detect_landmarks(result_img) or lms
        metrics = evaluate_metrics(img, result_img)

        return {
            "success": True,
            "operation": operation,
            "requested_intensity": requested_intensity,
            "applied_intensity": applied_intensity,
            "result_image_b64": numpy_to_b64(result_img),
            "metrics": metrics,
            "landmarks_before": [[x, y] for x, y in lms],
            "landmarks_after": [[x, y] for x, y in lms_after],
        }
    except Exception as exc:
        return {"error": "Face not detected or model error", "details": str(exc)}


@app.post("/warp/pro")
async def warp_pro(
    image: UploadFile = File(...),
    operation: str = Form("lip_plump"),
    intensity: float = Form(0.6),
    landmark_backend: str = Form("hybrid"),
    rbf_smooth: float = Form(2.8),
    temporal_smoothing: bool = Form(False),
    ema_alpha: float = Form(0.62),
    stream_id: str = Form("default"),
):
    try:
        intensity = float(np.clip(intensity, 0.0, 1.0))
        file_bytes = await image.read()
        img = bytes_to_numpy(file_bytes)

        lms, landmark_info = detect_landmarks_fused(
            img,
            backend=landmark_backend,
            temporal_smoothing=temporal_smoothing,
            ema_alpha=ema_alpha,
            stream_id=stream_id,
        )
        if lms is None:
            return {"error": "Face not detected or model error", "details": "No face detected."}

        rbf_smooth = float(np.clip(rbf_smooth, 0.8, 10.0))

        ops = {
            "lip_plump": lambda im, lm, inten: plump_lips_pro(im, lm, inten, smooth=rbf_smooth),
            "slim_face": lambda im, lm, inten: slim_face_pro(im, lm, inten, smooth=rbf_smooth),
            "smile_enhancement": lambda im, lm, inten: smile_enhancement_pro(im, lm, inten, smooth=rbf_smooth),
            "brow_lift": lambda im, lm, inten: brow_lift_pro(im, lm, inten, smooth=rbf_smooth),
        }
        if operation not in ops:
            return {
                "error": "Face not detected or model error",
                "details": f"Unknown operation '{operation}'. Valid: {list(ops.keys())}",
            }

        result_img = ops[operation](img, lms, intensity)
        metrics = quality_metrics(img, result_img)

        return {
            "success": True,
            "operation": operation,
            "intensity": intensity,
            "rbf_smooth": rbf_smooth,
            "landmark_info": landmark_info,
            "metrics": metrics,
            "result_image_b64": numpy_to_b64(result_img),
        }
    except Exception as exc:
        return {"error": "Face not detected or model error", "details": str(exc)}


@app.post("/frequency")
async def frequency(
    image: UploadFile = File(...),
    mode: str = Form("aging"),
    intensity: float = Form(0.5),
):
    intensity = float(np.clip(intensity, 0.0, 1.0))
    file_bytes = await image.read()
    img = bytes_to_numpy(file_bytes)

    if mode == "aging":
        result_img = apply_aging(img, intensity)
    elif mode == "deaging":
        result_img = apply_deaging(img, intensity)
    else:
        return {"success": False, "message": f"Unknown mode '{mode}'. Valid: aging, deaging"}

    metrics = evaluate_metrics(img, result_img)

    return {
        "success": True,
        "mode": mode,
        "result_image_b64": numpy_to_b64(result_img),
        "intensity": intensity,
        "metrics": metrics,
    }


@app.post("/report/export")
async def report_export(
    format: str = Form("csv"),
    operation: str = Form("unknown"),
    original_image_b64: str = Form(...),
    transformed_image_b64: str = Form(...),
    metrics_json: str = Form(...),
):
    fmt = (format or "csv").strip().lower()
    if fmt not in {"csv", "pdf"}:
        return {"success": False, "message": "Unsupported format. Use csv or pdf."}

    try:
        metrics = json.loads(metrics_json)
        core_metrics = {
            "mse": float(metrics["mse"]),
            "psnr": float(metrics["psnr"]),
            "ssim": float(metrics["ssim"]),
        }
    except Exception:
        return {"success": False, "message": "Invalid metrics payload. Expected mse, psnr, ssim."}

    try:
        original_np = b64_to_numpy(original_image_b64)
        transformed_np = b64_to_numpy(transformed_image_b64)
    except Exception as exc:
        return {"success": False, "message": f"Image decode failed: {exc}"}

    safe_op = "_".join(operation.lower().split()) or "unknown"

    if fmt == "csv":
        csv_text = generate_csv_report(core_metrics, operation=operation)
        file_name = f"evaluation-{safe_op}.csv"
        return {
            "success": True,
            "format": "csv",
            "file_name": file_name,
            "mime_type": "text/csv",
            "file_b64": base64.b64encode(csv_text.encode("utf-8")).decode("utf-8"),
        }

    pdf_bytes = generate_pdf_report(
        original_image=original_np,
        transformed_image=transformed_np,
        metrics=core_metrics,
        operation=operation,
    )
    file_name = f"evaluation-{safe_op}.pdf"
    return {
        "success": True,
        "format": "pdf",
        "file_name": file_name,
        "mime_type": "application/pdf",
        "file_b64": base64.b64encode(pdf_bytes).decode("utf-8"),
    }


@app.post("/aging/compare")
async def aging_compare(
    image: UploadFile = File(...),
    mode: str = Form("aging"),
    intensity: float = Form(0.6),
    landmark_backend: str = Form("hybrid"),
):
    """
    Runs both frequency-based and AI-guided aging on the same image and returns
    both results with metrics so the caller can show a side-by-side comparison.
    Satisfies FR-4.6 and FR-7.4.
    """
    try:
        intensity = float(np.clip(intensity, 0.0, 1.0))
        if mode not in ("aging", "deaging"):
            return {"success": False, "message": "Unknown mode. Valid: aging, deaging"}

        file_bytes = await image.read()
        valid, err = validate_image(file_bytes, image.filename or "upload.jpg")
        if not valid:
            return {"success": False, "message": err}

        img = bytes_to_numpy(file_bytes)

        # --- Frequency-based branch ---
        if mode == "aging":
            freq_result_img = apply_aging(img, intensity)
        else:
            freq_result_img = apply_deaging(img, intensity)
        freq_metrics = evaluate_metrics(img, freq_result_img)

        # --- AI-guided branch ---
        lms, landmark_info = detect_landmarks_fused(img, backend=landmark_backend, temporal_smoothing=False)
        ai_success = lms is not None
        if ai_success:
            if mode == "aging":
                ai_out = aging_pro(img, lms, intensity=intensity)
            else:
                ai_out = de_aging_pro(img, lms, intensity=intensity)
            ai_result_img = ai_out["result_image"]
            # ai_out["metrics"] uses raw 0-255 scale; keep it for the detailed response
            # but also compute normalized metrics for an apples-to-apples comparison.
            ai_metrics = ai_out["metrics"]
            ai_metrics_normalized = evaluate_metrics(img, ai_result_img)
        else:
            ai_result_img = img
            ai_metrics = {"mse": 0.0, "psnr": 0.0, "ssim": 1.0}
            ai_metrics_normalized = ai_metrics

        # --- Age estimation (best-effort) ---
        estimated_age_before, age_detected = _estimate_age_from_image(img)
        estimated_age_after_freq, _ = _estimate_age_from_image(freq_result_img)
        estimated_age_after_ai = None
        if ai_success:
            estimated_age_after_ai, _ = _estimate_age_from_image(ai_result_img)

        # --- Comparison deltas (freq vs ai, both on normalized 0-1 scale) ---
        def _safe_delta(a, b):
            try:
                return round(float(a) - float(b), 6)
            except Exception:
                return None

        comparison = {
            "note": "Deltas use normalized [0-1] MSE/PSNR scale for fair comparison",
            "mse_delta":  _safe_delta(ai_metrics_normalized.get("mse"),  freq_metrics.get("mse")),
            "psnr_delta": _safe_delta(ai_metrics_normalized.get("psnr"), freq_metrics.get("psnr")),
            "ssim_delta": _safe_delta(ai_metrics_normalized.get("ssim"), freq_metrics.get("ssim")),
            "winner": None,
        }
        try:
            freq_ssim = float(freq_metrics.get("ssim", 0))
            ai_ssim   = float(ai_metrics_normalized.get("ssim", 0))
            if abs(freq_ssim - ai_ssim) < 0.005:
                comparison["winner"] = "tie"
            elif ai_ssim > freq_ssim:
                comparison["winner"] = "ai_guided"
            else:
                comparison["winner"] = "frequency_based"
        except Exception:
            pass

        return {
            "success": True,
            "mode": mode,
            "intensity": intensity,
            "age_estimation": {
                "before": estimated_age_before,
                "detected": age_detected,
                "after_frequency": estimated_age_after_freq,
                "after_ai": estimated_age_after_ai,
            },
            "frequency_based": {
                "result_image_b64": numpy_to_b64(freq_result_img),
                "metrics": freq_metrics,
            },
            "ai_guided": {
                "success": ai_success,
                "landmark_info": landmark_info if ai_success else None,
                "result_image_b64": numpy_to_b64(ai_result_img) if ai_success else None,
                "metrics": ai_metrics,
            },
            "comparison": comparison,
        }
    except Exception as exc:
        return {"success": False, "message": str(exc)}


@app.post("/frequency/pro")
async def frequency_pro(
    image: UploadFile = File(...),
    mode: str = Form("aging"),
    intensity: float = Form(0.6),
    landmark_backend: str = Form("hybrid"),
    temporal_smoothing: bool = Form(False),
    ema_alpha: float = Form(0.62),
    stream_id: str = Form("default"),
):
    try:
        intensity = float(np.clip(intensity, 0.0, 1.0))
        file_bytes = await image.read()
        img = bytes_to_numpy(file_bytes)

        lms, landmark_info = detect_landmarks_fused(
            img,
            backend=landmark_backend,
            temporal_smoothing=temporal_smoothing,
            ema_alpha=ema_alpha,
            stream_id=stream_id,
        )

        if lms is None:
            return {"error": "Face not detected or model error", "details": "No face detected."}

        age_before, age_detected = _estimate_age_from_image(img)

        if mode == "aging":
            out = aging_pro(img, lms, intensity=intensity)
        elif mode == "deaging":
            out = de_aging_pro(img, lms, intensity=intensity)
        else:
            return {
                "error": "Face not detected or model error",
                "details": f"Unknown mode '{mode}'. Valid: aging, deaging",
            }

        age_after, _ = _estimate_age_from_image(out["result_image"])

        return {
            "success": True,
            "mode": mode,
            "intensity": intensity,
            "landmark_info": landmark_info,
            "estimated_age_before": age_before,
            "estimated_age_after": age_after,
            "age_delta": age_after - age_before,
            "age_detection": "deepface" if age_detected else "fallback",
            "result_image_b64": numpy_to_b64(out["result_image"]),
            "spectrum_gray_b64": numpy_to_b64(out["spectrum_gray"]),
            "spectrum_blue_b64": numpy_to_b64(out["spectrum_blue"]),
            "spectrum_red_b64": numpy_to_b64(out["spectrum_red"]),
            "metrics": out["metrics"],
        }
    except Exception as exc:
        return {"error": "Face not detected or model error", "details": str(exc)}


@app.post("/pro/apply-makeup")
async def pro_apply_makeup(
    image: UploadFile = File(...),
    region: str = Form(...),
    hex_color: str = Form(...),
    intensity: float = Form(0.6),
    landmark_backend: str = Form("hybrid"),
    temporal_smoothing: bool = Form(False),
    ema_alpha: float = Form(0.62),
    stream_id: str = Form("default"),
):
    try:
        intensity = float(np.clip(intensity, 0.0, 1.0))
        file_bytes = await image.read()
        img = bytes_to_numpy(file_bytes)

        lms, landmark_info = detect_landmarks_fused(
            img,
            backend=landmark_backend,
            temporal_smoothing=temporal_smoothing,
            ema_alpha=ema_alpha,
            stream_id=stream_id,
        )

        if lms is None:
            return {"error": "Face not detected or model error", "details": "No face detected."}

        out = apply_makeup(img, lms, region=region, hex_color=hex_color, intensity=intensity)

        return {
            "success": True,
            "region": out["region"],
            "hex_color": out["hex_color"],
            "intensity": out["intensity"],
            "landmark_info": landmark_info,
            "result_image_b64": numpy_to_b64(out["result_image"]),
        }
    except Exception as exc:
        return {"error": "Face not detected or model error", "details": str(exc)}


@app.post("/pro/apply-accessory")
async def pro_apply_accessory(
    image: UploadFile = File(...),
    accessory_type: str = Form(...),
    style: str = Form("classic"),
    hex_color: str = Form("#1D1D1F"),
    intensity: float = Form(0.7),
    scale: float = Form(1.0),
    offset_x: float = Form(0.0),
    offset_y: float = Form(0.0),
    landmark_backend: str = Form("hybrid"),
    temporal_smoothing: bool = Form(False),
    ema_alpha: float = Form(0.62),
    stream_id: str = Form("default"),
):
    try:
        intensity = float(np.clip(intensity, 0.0, 1.0))
        scale = float(np.clip(scale, 0.6, 1.6))
        offset_x = float(np.clip(offset_x, -120.0, 120.0))
        offset_y = float(np.clip(offset_y, -120.0, 120.0))
        file_bytes = await image.read()
        img = bytes_to_numpy(file_bytes)

        lms, landmark_info = detect_landmarks_fused(
            img,
            backend=landmark_backend,
            temporal_smoothing=temporal_smoothing,
            ema_alpha=ema_alpha,
            stream_id=stream_id,
        )

        if lms is None:
            return {"error": "Face not detected or model error", "details": "No face detected."}

        out = apply_accessory(
            img,
            lms,
            accessory_type=accessory_type,
            style=style,
            hex_color=hex_color,
            intensity=intensity,
            scale=scale,
            offset_x=offset_x,
            offset_y=offset_y,
        )

        return {
            "success": True,
            "accessory_type": out["accessory_type"],
            "style": out["style"],
            "hex_color": out["hex_color"],
            "intensity": out["intensity"],
            "scale": out["scale"],
            "offset_x": out["offset_x"],
            "offset_y": out["offset_y"],
            "landmark_info": landmark_info,
            "result_image_b64": numpy_to_b64(out["result_image"]),
        }
    except Exception as exc:
        return {"error": "Face not detected or model error", "details": str(exc)}


@app.post("/pro/hair-color")
async def pro_hair_color(
    image: UploadFile = File(...),
    hex_color: str = Form(...),
    intensity: float = Form(0.85),
    preserve_highlights: bool = Form(True),
):
    """
    Professional hair recoloring.

    - hex_color: Target color in #RRGGBB format (e.g. "#A020F0").
    - intensity: 0.0 (no change) – 1.0 (full replacement). Default 0.85.
    - preserve_highlights: Keep specular shine areas. Default True.

    Returns: { success, result_image_b64, hex_color, intensity }
    """
    try:
        intensity = float(np.clip(intensity, 0.0, 1.0))
        file_bytes = await image.read()
        valid, err = validate_image(file_bytes, image.filename or "upload.jpg")
        if not valid:
            return {"success": False, "error": err, "details": err}

        img = bytes_to_numpy(file_bytes)

        result_img = apply_hair_color(
            img,
            hex_color=hex_color,
            intensity=intensity,
            preserve_highlights=preserve_highlights,
        )

        return {
            "success": True,
            "hex_color": hex_color.lstrip("#").upper(),
            "intensity": round(intensity, 3),
            "result_image_b64": numpy_to_b64(result_img),
        }
    except ValueError as exc:
        return {"success": False, "error": "Invalid parameter", "details": str(exc)}
    except Exception as exc:
        return {"success": False, "error": "Hair color failed", "details": str(exc)}


def _parse_optional_float(value: str | None) -> float | None:
    if value is None or value == "":
        return None
    return float(value)


def _build_export_response(payload, csv_or_pdf: bytes, filename: str, media_type: str) -> Response:
    return Response(
        content=csv_or_pdf,
        media_type=media_type,
        headers={"Content-Disposition": f'attachment; filename="{filename}"'},
    )


@app.post("/export/csv")
async def export_csv(
    original_image: UploadFile = File(...),
    result_image: UploadFile = File(...),
    operation: str = Form("pro"),
    intensity: float = Form(0.0),
    age_before: str | None = Form(None),
    age_after: str | None = Form(None),
):
    try:
        original_bytes = await original_image.read()
        result_bytes = await result_image.read()
        valid_original, original_err = validate_image(original_bytes, original_image.filename or "original.jpg")
        if not valid_original:
            return {"error": "Face not detected or model error", "details": original_err}

        valid_result, result_err = validate_image(result_bytes, result_image.filename or "result.jpg")
        if not valid_result:
            return {"error": "Face not detected or model error", "details": result_err}

        original_img = bytes_to_numpy(original_bytes)
        result_img = bytes_to_numpy(result_bytes)
        metrics = compute_quality_metrics(original_img, result_img)
        payload = build_report_payload(
            operation=operation,
            intensity=float(intensity),
            age_before=age_before,
            age_after=age_after,
            metrics=metrics,
            original_image_bytes=original_bytes,
            result_image_bytes=result_bytes,
        )
        csv_bytes = generate_csv_bytes(payload)
        return _build_export_response(payload, csv_bytes, "facial-report.csv", "text/csv; charset=utf-8")
    except Exception as exc:
        return {"error": "Face not detected or model error", "details": str(exc)}


@app.post("/export/pdf")
async def export_pdf(
    original_image: UploadFile = File(...),
    result_image: UploadFile = File(...),
    operation: str = Form("pro"),
    intensity: float = Form(0.0),
    age_before: str | None = Form(None),
    age_after: str | None = Form(None),
):
    try:
        original_bytes = await original_image.read()
        result_bytes = await result_image.read()
        valid_original, original_err = validate_image(original_bytes, original_image.filename or "original.jpg")
        if not valid_original:
            return {"error": "Face not detected or model error", "details": original_err}

        valid_result, result_err = validate_image(result_bytes, result_image.filename or "result.jpg")
        if not valid_result:
            return {"error": "Face not detected or model error", "details": result_err}

        original_img = bytes_to_numpy(original_bytes)
        result_img = bytes_to_numpy(result_bytes)
        metrics = compute_quality_metrics(original_img, result_img)
        payload = build_report_payload(
            operation=operation,
            intensity=float(intensity),
            age_before=age_before,
            age_after=age_after,
            metrics=metrics,
            original_image_bytes=original_bytes,
            result_image_bytes=result_bytes,
        )
        pdf_bytes = generate_pdf_bytes(payload)
        return _build_export_response(payload, pdf_bytes, "facial-report.pdf", "application/pdf")
    except Exception as exc:
        return {"error": "Face not detected or model error", "details": str(exc)}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
