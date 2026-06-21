"""
SAM (Style-based Age Manipulation) kurulum scripti.
Hem model ağırlıklarını hem de vendor kaynak kodunu indirir.

Kullanım:
    python python_service/download_models.py
"""
import os
import subprocess
import sys

SERVICE_DIR = os.path.dirname(__file__)
MODELS_DIR  = os.path.join(SERVICE_DIR, "models")
VENDOR_DIR  = os.path.join(SERVICE_DIR, "vendor")
SAM_MODEL   = os.path.join(MODELS_DIR, "sam_ffhq_aging.pt")
SAM_VENDOR  = os.path.join(VENDOR_DIR, "SAM")

SAM_REPO_URL  = "https://github.com/yuval-alaluf/SAM.git"
SAM_GDRIVE_ID = "1XyumF6_fdAxFmxpFcmPf-q84LU_22EMC"


def clone_sam_vendor() -> None:
    if os.path.isdir(SAM_VENDOR) and os.listdir(SAM_VENDOR):
        print(f"SAM kaynak kodu zaten mevcut: {SAM_VENDOR}")
        return
    print("SAM kaynak kodu indiriliyor...")
    os.makedirs(VENDOR_DIR, exist_ok=True)
    result = subprocess.run(
        ["git", "clone", "--depth", "1", SAM_REPO_URL, SAM_VENDOR],
        capture_output=True, text=True,
    )
    if result.returncode != 0:
        print(f"HATA: git clone başarısız:\n{result.stderr}")
        print(f"Manuel olarak klonlayın: git clone {SAM_REPO_URL} {SAM_VENDOR}")
        sys.exit(1)
    # İç .git klasörünü kaldır — ana repoda submodule sorunu yaratmaz
    nested_git = os.path.join(SAM_VENDOR, ".git")
    if os.path.isdir(nested_git):
        import shutil
        shutil.rmtree(nested_git)
    print(f"SAM kaynak kodu hazır: {SAM_VENDOR}")


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


def download_sam_model() -> None:
    os.makedirs(MODELS_DIR, exist_ok=True)
    if os.path.exists(SAM_MODEL):
        size_gb = os.path.getsize(SAM_MODEL) / (1024 ** 3)
        print(f"SAM modeli zaten mevcut ({size_gb:.1f} GB): {SAM_MODEL}")
        return
    print("SAM model agirliklari indiriliyor (~2.1 GB)...")
    download_from_gdrive(SAM_GDRIVE_ID, SAM_MODEL)
    if os.path.exists(SAM_MODEL):
        size_gb = os.path.getsize(SAM_MODEL) / (1024 ** 3)
        print(f"İndirildi ({size_gb:.1f} GB): {SAM_MODEL}")
    else:
        print("HATA: İndirme başarısız. Manuel olarak indirin:")
        print(f"  https://drive.google.com/file/d/{SAM_GDRIVE_ID}/view")
        print(f"  Hedef: {SAM_MODEL}")
        sys.exit(1)


def main() -> None:
    clone_sam_vendor()
    download_sam_model()
    print("\nKurulum tamamlandı. Servisi başlatabilirsiniz:")
    print("  uvicorn main:app --host 0.0.0.0 --port 8000")


if __name__ == "__main__":
    main()
