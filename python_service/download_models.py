"""
SAM (Style-based Age Manipulation) model indirme scripti.

Kullanım:
    python python_service/download_models.py
"""
import os
import sys

MODELS_DIR = os.path.join(os.path.dirname(__file__), "models")
SAM_MODEL  = os.path.join(MODELS_DIR, "sam_ffhq_aging.pt")

# SAM modeli — orijinal repo: https://github.com/yuval-alaluf/SAM
# Google Drive direkt indirme linki (SAM resmi weighti, 2.1GB)
SAM_GDRIVE_ID = "1XyumF6_fdAxFmxpFcmPf-q84LU_22EMC"


def download_from_gdrive(file_id: str, dest: str) -> None:
    try:
        import gdown
    except ImportError:
        print("gdown yükleniyor...")
        os.system(f"{sys.executable} -m pip install gdown -q")
        import gdown

    url = f"https://drive.google.com/uc?id={file_id}"
    print(f"İndiriliyor: {url}")
    gdown.download(url, dest, quiet=False)


def main() -> None:
    os.makedirs(MODELS_DIR, exist_ok=True)

    if os.path.exists(SAM_MODEL):
        size_gb = os.path.getsize(SAM_MODEL) / (1024 ** 3)
        print(f"SAM modeli zaten mevcut ({size_gb:.1f} GB): {SAM_MODEL}")
        return

    print("SAM modeli bulunamadı, indiriliyor (≈2.1 GB)...")
    download_from_gdrive(SAM_GDRIVE_ID, SAM_MODEL)

    if os.path.exists(SAM_MODEL):
        size_gb = os.path.getsize(SAM_MODEL) / (1024 ** 3)
        print(f"İndirildi ({size_gb:.1f} GB): {SAM_MODEL}")
    else:
        print("HATA: İndirme başarısız. Manuel olarak indirin:")
        print(f"  https://drive.google.com/file/d/{SAM_GDRIVE_ID}/view")
        print(f"  Hedef: {SAM_MODEL}")
        sys.exit(1)


if __name__ == "__main__":
    main()
