from datetime import datetime, timezone

import pytest
from fastapi.testclient import TestClient

import ml_utils
from main import app


@pytest.fixture
def client():
    # entering the context runs the startup event, which trains the model
    with TestClient(app) as c:
        yield c


def test_ping(client):
    response = client.get("/ping")
    assert response.status_code == 200
    assert response.json() == {"ping": "pong"}


@pytest.mark.parametrize(
    "payload, expected",
    [
        ({"sepal_length": 5.1, "sepal_width": 3.5, "petal_length": 1.4, "petal_width": 0.2}, "Iris Setosa"),
        ({"sepal_length": 6.0, "sepal_width": 2.9, "petal_length": 4.5, "petal_width": 1.5}, "Iris Versicolour"),
        ({"sepal_length": 6.9, "sepal_width": 3.1, "petal_length": 5.8, "petal_width": 2.3}, "Iris Virginica"),
    ],
)
def test_predicts_each_class(client, payload, expected):
    response = client.post("/predict_flower", json=payload)
    assert response.status_code == 200
    assert response.json()["flower_class"] == expected


def test_response_has_model_and_recent_timestamp(client):
    payload = {"sepal_length": 5.1, "sepal_width": 3.5, "petal_length": 1.4, "petal_width": 0.2}
    body = client.post("/predict_flower", json=payload).json()
    assert body["model"] in ml_utils.CANDIDATES
    timestamp = datetime.fromisoformat(body["timestamp"])
    assert abs((datetime.now(timezone.utc) - timestamp).total_seconds()) < 60


def test_best_model_is_selected():
    scores = ml_utils.load_model()
    assert ml_utils.model_name == max(scores, key=scores.get)


def test_feedback_keeps_previous_knowledge(client):
    feedback = [{"sepal_length": 6.9, "sepal_width": 3.1, "petal_length": 5.8, "petal_width": 2.3,
                 "flower_class": "Iris Virginica"}]
    response = client.post("/feedback_loop", json=feedback)
    assert response.status_code == 200
    assert response.json()["samples_received"] == 1
    # a single-class feedback batch must not make the model forget the other classes
    setosa = {"sepal_length": 5.1, "sepal_width": 3.5, "petal_length": 1.4, "petal_width": 0.2}
    assert client.post("/predict_flower", json=setosa).json()["flower_class"] == "Iris Setosa"


def test_rejects_invalid_input(client):
    bad = {"sepal_length": -1, "sepal_width": 3.5, "petal_length": 1.4, "petal_width": 0.2}
    assert client.post("/predict_flower", json=bad).status_code == 422
    unknown = {**bad, "sepal_length": 5.1, "flower_class": "Rose"}
    assert client.post("/feedback_loop", json=[unknown]).status_code == 422
