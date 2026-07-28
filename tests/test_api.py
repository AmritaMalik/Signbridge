"""
Integration tests for the sign-language-recognition FastAPI server.

Tests cover:
  - Valid image prediction (POST /predict)
  - Invalid base64 rejection (POST /predict -> 400)
  - Frontend serving (GET / -> 200, text/html)

A session-scoped conftest fixture (conftest.py) creates dummy model
artefacts before these tests run so no real GPU / trained model is needed.
"""

from __future__ import annotations

import base64
import io

import httpx
import pytest
from PIL import Image
from starlette.testclient import TestClient

# ---------------------------------------------------------------------------
# Import the app AFTER conftest has set up artefacts (session autouse fixture)
# ---------------------------------------------------------------------------
from main import app  # noqa: E402


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_test_image_b64(width: int = 64, height: int = 64) -> str:
    """
    Create a solid red 64×64 JPEG image and return it as a base64 string.
    Simulates a webcam frame captured from the bounding box region.
    """
    img = Image.new("RGB", (width, height), color=(255, 0, 0))
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=80)
    return base64.b64encode(buf.getvalue()).decode("utf-8")


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def client(dummy_model_artifacts):  # depends on session fixture
    """
    ASGI test client for the FastAPI app.

    Uses Starlette's TestClient which correctly handles the ASGI lifespan
    (model loading) in synchronous pytest without needing anyio/asyncio.
    Scoped to module so the lifespan runs once per test module.
    """
    with TestClient(app, raise_server_exceptions=True) as c:
        yield c


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestPredictEndpoint:
    """Tests for POST /predict."""

    def test_predict_valid_image(self, client: httpx.Client):
        """
        Requirement 11.1 — A valid base64 frame should yield 200 with
        'character' (str) and 'confidence' (float) fields.
        """
        b64 = make_test_image_b64()
        response = client.post("/predict", json={"image": b64})

        assert response.status_code == 200, f"Expected 200, got {response.status_code}: {response.text}"

        body = response.json()
        assert "character" in body, f"Missing 'character' field: {body}"
        assert "confidence" in body, f"Missing 'confidence' field: {body}"
        assert isinstance(body["character"], str), "character should be a string"
        assert isinstance(body["confidence"], float), "confidence should be a float"
        assert 0.0 <= body["confidence"] <= 1.0, "confidence must be in [0, 1]"

    def test_predict_invalid_base64(self, client: httpx.Client):
        """
        Requirement 11.2 — An invalid base64 payload should return HTTP 400.
        """
        response = client.post("/predict", json={"image": "!!!not_valid_base64!!!"})

        assert response.status_code == 400, (
            f"Expected 400 for bad base64, got {response.status_code}: {response.text}"
        )

    def test_predict_empty_string(self, client: httpx.Client):
        """Edge case: empty base64 string should also return 400."""
        response = client.post("/predict", json={"image": ""})
        # An empty string decodes to zero bytes — PIL will fail to open it
        assert response.status_code == 400

    def test_predict_confidence_is_rounded(self, client: httpx.Client):
        """Confidence score should be rounded to at most 4 decimal places."""
        b64 = make_test_image_b64()
        response = client.post("/predict", json={"image": b64})
        assert response.status_code == 200
        confidence = response.json()["confidence"]
        # Verify rounding: converting to string should have <= 4 decimal digits
        decimal_str = str(confidence).split(".")[-1] if "." in str(confidence) else ""
        assert len(decimal_str) <= 4, f"Confidence has more than 4 decimal places: {confidence}"


class TestFrontendServing:
    """Tests for GET /."""

    def test_frontend_served(self, client: httpx.Client):
        """
        Requirement 11.3 — GET / must return HTTP 200 with text/html content-type.
        """
        response = client.get("/")

        assert response.status_code == 200, (
            f"Expected 200 for GET /, got {response.status_code}: {response.text}"
        )
        assert "text/html" in response.headers.get("content-type", ""), (
            f"Expected text/html content-type, got: {response.headers.get('content-type')}"
        )

    def test_frontend_contains_html(self, client: httpx.Client):
        """The frontend response body should look like an HTML document."""
        response = client.get("/")
        assert response.status_code == 200
        assert "<!DOCTYPE html>" in response.text or "<html" in response.text.lower()
