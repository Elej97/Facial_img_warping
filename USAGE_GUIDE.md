# Facial Warping - Usage Guide

## 1. Servisleri Baslat

1. Python CV service:
   - `cd python_service`
   - `python -m uvicorn main:app --host 0.0.0.0 --port 8000 --reload`
2. Backend proxy:
   - `cd backend`
   - `node server.js`
3. Frontend:
   - `cd frontend`
   - `npx expo start --web --port 8082`

## 2. Temel Akis (Create Ekrani)

1. `Gorsel Sec` ile fotograf yukle.
2. `Yuzu Tespit Et` ile preprocess olustur.
3. Istersen `Noktalari Tespit Et` ile landmark onizle.

## 3. Klasik Moduller

- Yuz Deforme: smile, raise_eyebrows, widen_lips, slim_face
- Yaslandirma / Genclestirme: klasik frequency endpoint

## 4. Pro Lab (Canli)

Create ekranindaki `Pro Lab (Canli)` bolumunde:

1. Operasyon sec:
   - Pro Smile
   - Pro Brow Lift
   - Pro Lip Plump
   - Pro Slim Face
   - Pro Aging
   - Pro De-Aging
2. `Intensity` slider ile etki siddetini ayarla.
3. `RBF Smooth` slider ile deformasyon yumusakligini ayarla.
4. Slider hareketlerinde sonuc otomatik guncellenir (real-time debounce ile).

## 5. Side-by-Side ve Metrikler

Pro sonucu geldikten sonra ayni panelde:

- Sol: Orijinal
- Sag: Pro Sonuc
- Altta: MSE, PSNR, SSIM (ve varsa HF/LF ratio metrikleri)

## 6. Spectrum Goruntuleme

`Pro Aging` veya `Pro De-Aging` secildiginde:

- `Spectrum Before`
- `Spectrum After`

log-scaled frekans spektrumlari yan yana gorunur.

## 7. Export

Pro sonuc ve metrikler hazirken:

- `CSV Export`: metrik raporunu CSV olarak indirir/paylasir.
- `PDF Export`: metrik raporunu PDF olarak olusturur/paylasir.

## 8. API Ozeti (Pro)

Backend uzerinden kullanilan endpointler:

- `POST /api/warp/pro`
- `POST /api/frequency/pro`

Donen alanlar (ozet):

- `result_image_b64`
- `metrics` (MSE, PSNR, SSIM, HF/LF)
- Frequency/pro icin:
  - `spectrum_before_b64`
  - `spectrum_after_b64`

## 9. Onerilen Baslangic Degerleri

- Pro Smile: intensity `0.55-0.70`, rbf `2.4-3.2`
- Pro Brow Lift: intensity `0.45-0.65`, rbf `3.0-4.2`
- Pro Lip Plump: intensity `0.55-0.75`, rbf `1.8-2.8`
- Pro Slim Face: intensity `0.45-0.70`, rbf `3.0-4.5`
- Pro Aging/De-Aging: intensity `0.45-0.75`
