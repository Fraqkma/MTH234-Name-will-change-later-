from fastapi.testclient import TestClient

from api.main import app
from ridge_lab.demo import build_demo

client = TestClient(app)


def test_demo_is_deterministic_and_does_not_return_held_out_targets() -> None:
    first = build_demo(degree=5, lam=0.25)
    second = build_demo(degree=5, lam=0.25)

    assert first == second
    assert first["illustrative"] is True
    assert "test" not in first


def test_api_serves_site_and_validated_train_validation_demo() -> None:
    page = client.get("/")
    demo = client.get("/api/demo", params={"degree": 4, "lambda": 0})

    assert page.status_code == 200
    assert "When perfect" in page.text
    assert client.get("/api/health").json() == {"status": "ok"}
    assert demo.status_code == 200
    payload = demo.json()
    assert payload["degree"] == 4
    assert payload["lambda"] == 0
    assert "test" not in payload
    assert set(payload["metrics"]) == {"ols", "ridge"}


def test_api_rejects_out_of_range_controls() -> None:
    assert client.get("/api/demo", params={"degree": 16}).status_code == 422
    assert client.get("/api/demo", params={"lambda": -1}).status_code == 422
    assert client.get("/api/demo", params={"lambda": 1001}).status_code == 422
