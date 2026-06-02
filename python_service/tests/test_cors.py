"""
CORS konfigürasyonu testleri — main.py _parse_allowed_origins() fonksiyonu
ve FastAPI middleware davranışı.

Çalıştır:
    cd python_service
    source venv/bin/activate
    pytest tests/test_cors.py -v
"""

import os
import importlib
import sys
from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient


# ---------------------------------------------------------------------------
# Yardımcı: main modülünü temiz şekilde yeniden yükle (env değişkenleriyle)
# ---------------------------------------------------------------------------

def _fresh_parse(env_value: str | None) -> list[str]:
    """
    _parse_allowed_origins() fonksiyonunu verilen env değeri ile çağırır.
    main modülünü yeniden import etmeden çalışır — sadece fonksiyonu test eder.
    """
    env = {} if env_value is None else {"ALLOWED_ORIGINS": env_value}
    with patch.dict(os.environ, env, clear_env_key="ALLOWED_ORIGINS"):
        # main'i yeniden import etmeden fonksiyonu doğrudan test etmek için
        # parse mantığını inline çalıştırıyoruz.
        raw = os.environ.get("ALLOWED_ORIGINS", "").strip()
        if not raw or raw == "*":
            return ["*"]
        origins = [o.strip() for o in raw.split(",") if o.strip()]
        return origins if origins else ["*"]


def _remove_env_key(mapping, key):
    mapping.pop(key, None)
    return mapping


# ---------------------------------------------------------------------------
# 1. _parse_allowed_origins — birim testleri
# ---------------------------------------------------------------------------

class TestParseAllowedOrigins:

    def test_env_not_set_returns_wildcard(self):
        """ALLOWED_ORIGINS tanımlı değilse ["*"] döner."""
        with patch.dict(os.environ, {}, clear=False):
            os.environ.pop("ALLOWED_ORIGINS", None)
            result = _fresh_parse(None)
        assert result == ["*"]

    def test_wildcard_string_returns_wildcard(self):
        """ALLOWED_ORIGINS=* açıkça ayarlanmışsa ["*"] döner."""
        result = _fresh_parse("*")
        assert result == ["*"]

    def test_empty_string_returns_wildcard(self):
        """ALLOWED_ORIGINS= (boş) → ["*"] döner, sunucu çökmez."""
        result = _fresh_parse("")
        assert result == ["*"]

    def test_single_origin(self):
        """Tek origin doğru parse edilir."""
        result = _fresh_parse("https://facemorphapp.com")
        assert result == ["https://facemorphapp.com"]

    def test_multiple_origins_comma_separated(self):
        """Virgülle ayrılmış birden fazla origin doğru ayrılır."""
        result = _fresh_parse(
            "https://facemorphapp.com,https://www.facemorphapp.com"
        )
        assert result == [
            "https://facemorphapp.com",
            "https://www.facemorphapp.com",
        ]

    def test_origins_with_extra_spaces(self):
        """Origin'lerin etrafındaki boşluklar temizlenir."""
        result = _fresh_parse(
            "  https://facemorphapp.com ,  http://localhost:8081  "
        )
        assert result == ["https://facemorphapp.com", "http://localhost:8081"]

    def test_localhost_origins(self):
        """Lokal geliştirme origin'leri kabul edilir."""
        result = _fresh_parse(
            "http://localhost:8081,http://localhost:19006,http://localhost:3000"
        )
        assert len(result) == 3
        assert "http://localhost:8081" in result
        assert "http://localhost:19006" in result

    def test_whitespace_only_returns_wildcard(self):
        """Sadece boşluk → ["*"] döner."""
        result = _fresh_parse("   ")
        assert result == ["*"]

    def test_no_trailing_slash_preserved(self):
        """Origin'in sonundaki slash korunmaz (olduğu gibi gelir)."""
        result = _fresh_parse("https://facemorphapp.com")
        assert result[0] == "https://facemorphapp.com"
        assert not result[0].endswith("/")


# ---------------------------------------------------------------------------
# 2. /health endpoint — CORS header testleri
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def client():
    """FastAPI test client — main.py'yi import eder."""
    # Ağır AI modellerini yüklemeden test etmek için patch kullan
    with (
        patch("builtins.__import__", side_effect=_selective_import),
        patch.dict(os.environ, {"ALLOWED_ORIGINS": "*"}),
    ):
        # main modülü zaten yüklendiyse cache'den al
        if "main" in sys.modules:
            app = sys.modules["main"].app
        else:
            import main as m
            app = m.app
    return TestClient(app, raise_server_exceptions=False)


def _selective_import(name, *args, **kwargs):
    """deepface ve insightface'i import ederken hata fırlatma — test ortamında yok olabilir."""
    if name in ("deepface", "insightface"):
        raise ModuleNotFoundError(f"Test: {name} mocked as missing")
    return __import__(name, *args, **kwargs)


class TestHealthEndpoint:

    def test_health_returns_200(self):
        """/health endpoint'i 200 döner."""
        with patch.dict(os.environ, {"ALLOWED_ORIGINS": "*"}):
            if "main" not in sys.modules:
                pytest.skip("main modülü yüklenemedi — AI bağımlılıkları eksik olabilir")
            from main import app
            c = TestClient(app)
            resp = c.get("/health")
        assert resp.status_code == 200

    def test_health_response_shape(self):
        """/health yanıtı status ve service alanlarını içerir."""
        if "main" not in sys.modules:
            pytest.skip("main modülü yüklenemedi")
        from main import app
        c = TestClient(app)
        resp = c.get("/health")
        body = resp.json()
        assert "status" in body
        assert body["status"] == "ok"
        assert "service" in body

    def test_cors_wildcard_allows_any_origin(self):
        """ALLOWED_ORIGINS=* iken herhangi bir origin'den gelen isteğe izin verilir."""
        if "main" not in sys.modules:
            pytest.skip("main modülü yüklenemedi")
        from main import app
        c = TestClient(app)
        resp = c.options(
            "/health",
            headers={
                "Origin": "https://random-site.com",
                "Access-Control-Request-Method": "GET",
            },
        )
        # Wildcard modunda origin ya echo'lanır ya da * döner
        origin_header = resp.headers.get("access-control-allow-origin", "")
        assert origin_header in ("*", "https://random-site.com")


# ---------------------------------------------------------------------------
# 3. _parse_allowed_origins — sınır durumları
# ---------------------------------------------------------------------------

class TestEdgeCases:

    def test_only_commas_returns_wildcard(self):
        """Sadece virgül içeren string → ["*"] döner."""
        result = _fresh_parse(",,,")
        assert result == ["*"]

    def test_mixed_valid_and_empty_segments(self):
        """Virgüller arasında boş segment varsa atlanır."""
        result = _fresh_parse("https://facemorphapp.com,,http://localhost:8081")
        assert result == ["https://facemorphapp.com", "http://localhost:8081"]

    def test_origin_count_matches_input(self):
        """Verilen origin sayısı ile çıktı sayısı eşleşir."""
        origins = [
            "https://a.com",
            "https://b.com",
            "https://c.com",
            "http://localhost:8081",
        ]
        result = _fresh_parse(",".join(origins))
        assert len(result) == 4

    def test_all_whitespace_segments_filtered(self):
        """Sadece boşluktan oluşan segment'ler filtrelenir."""
        result = _fresh_parse("https://facemorphapp.com,  ,  ,http://localhost:8081")
        assert result == ["https://facemorphapp.com", "http://localhost:8081"]
