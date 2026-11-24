# Iris Prediction API with Model Selection and CI/CD

A FastAPI service that predicts the species of an Iris flower from four measurements. On startup it trains several classifiers and serves whichever one has the best cross-validated accuracy. It also has a feedback endpoint that retrains the model with labelled samples from users.

Started from the [PGCSEDS-IIITH/mlops-iris](https://github.com/PGCSEDS-IIITH/mlops-iris) course template (IIIT Hyderabad).

## Features
- **Model selection:** Gaussian Naive Bayes and a Decision Tree are both scored with 5-fold cross-validation on the training split. The better one is used, and its accuracy on a held-out test split is logged.
- **Traceable predictions:** each response includes the predicted class, the model that made it, and a UTC timestamp.
- **Feedback loop:** `/feedback_loop` adds labelled samples to the training data and refits. Earlier data is kept, so the model doesn't forget the other classes.
- **Input validation:** measurements must be positive, and feedback labels must be one of the three species.
- **CI/CD:** GitHub Actions runs the tests with a coverage gate (90%) on every push and PR. Pushes also get a zipped release artifact.

## API
| Method | Route | Description |
|---|---|---|
| GET | `/ping` | Health check |
| POST | `/predict_flower` | `{sepal_length, sepal_width, petal_length, petal_width}` → `{flower_class, model, timestamp}` |
| POST | `/feedback_loop` | A list of measurements, each with its correct `flower_class` → retrains the model |

Interactive docs are served at `/`.

## Run it
```bash
pip install -r requirements.txt
python main.py
```
The API is then available at http://localhost:8888.

```bash
curl -s localhost:8888/predict_flower -H 'content-type: application/json' -d '{"sepal_length":5.1,"sepal_width":3.5,"petal_length":1.4,"petal_width":0.2}'
```

## Tests
```bash
pip install -r requirements-dev.txt
pytest --cov=main --cov=ml_utils
```
There are 8 tests: one per species, the timestamp, model selection, the feedback loop and input validation.
