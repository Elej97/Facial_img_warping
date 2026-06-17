# Facial Image Warping

Yuz tespiti, nokta haritasi, geometrik deforme ve yaslandirma/genclestime efektleri iceren mobil + web uygulamasi.

## Mimari

```
frontend/          Expo + React Native (web & mobil)
backend/           Express proxy (Node.js, port 3000)
python_service/    FastAPI bilgisayarli goru servisi (port 8000)
```

Frontend, dosya yukleme ve kimlik dogrulama icin Express backend uzerinden iletisim kurar. Express, CV isteklerini Python servisine yonlendirir.

## Ozellikler

### Yuz Tespiti ve On Isleme
- MediaPipe Face Detector ile yuz bulma ve kisilma
- Tespit edilen yuz 256x256 piksel olarak normalize edilir

### Yuz Noktalari (Landmark)
- MediaPipe Face Landmarker ile 478 nokta tespiti
- Gozler, kaslar, agiz, burun, cene ve alin bolgeleri
- Nokta gorunumu acip kapatilabilir

### Geometrik Deforme (Warping)
Scipy Delaunay ucgenleme + sinir cubukları (boundary anchors) kullanilarak kesintisiz deforme:

| Operasyon | Aciklama |
|---|---|
| Gulumse | Agiz koseleri disari ve yukari kayar, yanaklar hafifce siser |
| Kaslari Kaldir | Kas noktalarini yukari ceker, ust goz kapagi da hafifce yukselir |
| Dudaklari Genislet | Agiz koseleri yanlara acilir, ic dudak noktaları da olceklenir |
| Ince Yuz | Cene ve yanak noktalari burun ucuna dogru iceridoğru cekilir |

### Yas Efektleri (FFT Tabanli)

**Yaslandirma:**
- Gaussiyan gurultu ile kirisi dokusu
- FFT yuksek frekansi amplifikasyonu (deri purtuksuzlugu artar)
- Deri tonu kararma ve sararmasi
- Unsharp masking ile doku keskinlestirme
- Yogunluga gore %35–85 harmanlama

**Genclestime:**
- Cok gecisli bilateral filtre ile deri yumusatma
- FFT alcak geciren filtre (yuksek frekans bastirma)
- Canny kenar koruma (yuz yapisi keskin kalir)
- Parlaklik artisi ve serin ton kayma
- Yogunluga gore %40–85 harmanlama

### Arayuz
- Oncesi / Sonrasi yan yana karsilastirma
- Gorsele tiklaninca tam ekran lightbox
- Kirpma editoru (serbest boyut, surukle-birak)
- Yogunluk kaydirici (varsayilan 0.8)

## Klasor Yapisi

```
frontend/
  app/
    create.tsx        Ana CV ekrani (tespit, nokta, deforme, yas)
    index.tsx         Hosgeldin ekrani
    register.tsx      Kayit ekrani
    login.tsx         Giris ekrani
  components/         Paylasilan UI bilesenleri
  constants/          Renk paleti ve tema

backend/
  server.js           Express API + SQLite kimlik dogrulama + Python proxy
  db.js               SQLite erisim

python_service/
  main.py             FastAPI endpoint tanimlari
  modules/
    preprocessing.py  Yuz tespiti ve normalize
    landmark.py       478 nokta tespiti ve cizimi
    warping.py        Delaunay deforme motoru
    aging.py          Yaslandirma / genclestime efektleri
  utils/
    image_utils.py    Bayt-NumPy-Base64 donusumleri
    model_utils.py    MediaPipe model yolu yonetimi
  models/             MediaPipe .task model dosyalari
```

## Kurulum

### 1. Frontend & Backend (Node.js)

```bash
npm --prefix frontend install
npm --prefix backend install
```

### 2. Python Servisi

```bash
cd python_service
pip install -r requirements.txt
```

Gereklilikler: `fastapi`, `uvicorn`, `opencv-python-headless`, `mediapipe`, `numpy`, `scipy`, `Pillow`

### 3. SAM Model Dosyası (Yaşlandırma için zorunlu)

`sam_ffhq_aging.pt` dosyası 2.1 GB olduğundan Git'e dahil edilmemiştir.
Projeyi ilk kez kuruyorsan aşağıdaki scripti çalıştır — model otomatik indirilir:

```bash
python python_service/download_models.py
```

> Model kaynağı: [yuval-alaluf/SAM](https://github.com/yuval-alaluf/SAM) (Style-based Age Manipulation)
> İndirme yaklaşık 2.1 GB veri gerektirir, internet bağlantısına göre 5–15 dakika sürebilir.

İndirme tamamlandığında dosya `python_service/models/sam_ffhq_aging.pt` konumunda olmalıdır.

## Calistirma

Her uc servisi ayri terminalde baslatın:

```bash
# Python CV servisi (port 8000)
cd python_service
uvicorn main:app --host 0.0.0.0 --port 8000 --reload

# Node.js backend (port 3000)
cd backend
# 1. Node.js bağımlılıkları
npm --prefix frontend install
npm --prefix backend install

# 2. Python bağımlılıkları (doğru dizinden)
cd python_service
pip install -r requirements.txt
cd ..

# 3. Servisleri ayrı terminallerde çalıştır
# Terminal 1:
cd python_service
uvicorn main:app --host 0.0.0.0 --port 8000 --reload

# Terminal 2:
cd backend
node server.js

# Terminal 3:
cd frontend
npx expo start --web --port 8082

# Expo frontend (web, port 8082)
cd frontend
npx expo start --web --port 8082
```

## API Endpoint'leri

Python servisi (port 8000), Express uzerinden `/api/*` yoluyla erisiliyor:

| Yontem | Yol | Aciklama |
|---|---|---|
| POST | `/api/preprocess` | Yuz tespit + 256x256 normalize |
| POST | `/api/landmarks` | 478 nokta tespiti |
| POST | `/api/warp` | Geometrik deforme (`operation`, `intensity`) |
| POST | `/api/frequency` | Yas efekti (`mode=aging/deaging`, `intensity`) |
| GET  | `/api/health` | Servis saglik kontrolu |

Tum gorsel gonderimler `multipart/form-data` formatinda yapilir; cevaplar `result_image_b64` (PNG base64) icerir.
## Realtime Camera (Advanced Setup)

To run realtime camera (Vision Camera):

You need:

- Android Studio
- Android SDK
- ADB configured

Then run:

npx expo run:android

⚠️ Not required for normal development.
Web version works without this setup.