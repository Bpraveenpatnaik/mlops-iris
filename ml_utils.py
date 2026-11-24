from sklearn import datasets
from sklearn.base import clone
from sklearn.metrics import accuracy_score
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier

# candidate classifiers; the one with the best cross-validated accuracy is used
CANDIDATES = {
    "GaussianNB": GaussianNB(),
    "DecisionTree": DecisionTreeClassifier(max_depth=4, random_state=42),
}

# define the class encodings and reverse encodings
classes = {0: "Iris Setosa", 1: "Iris Versicolour", 2: "Iris Virginica"}
r_classes = {y: x for x, y in classes.items()}

FEATURES = ["sepal_length", "sepal_width", "petal_length", "petal_width"]

# the selected model and the data it was trained on
clf = None
model_name = None
X_base, y_base = [], []
feedback_X, feedback_y = [], []


def load_model():
    """Train every candidate, keep the one with the best cross-validated
    accuracy on the training split, and report its accuracy on the test split."""
    global clf, model_name, X_base, y_base
    X, y = datasets.load_iris(return_X_y=True)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )

    scores = {
        name: cross_val_score(model, X_train, y_train, cv=5).mean()
        for name, model in CANDIDATES.items()
    }
    for name, score in scores.items():
        print(f"{name}: cross-validated accuracy {score:.3f}")

    model_name = max(scores, key=scores.get)
    clf = clone(CANDIDATES[model_name]).fit(X_train, y_train)
    test_acc = accuracy_score(y_test, clf.predict(X_test))
    print(f"Selected {model_name}, test accuracy {test_acc:.3f}")

    X_base, y_base = X_train.tolist(), y_train.tolist()
    feedback_X.clear()
    feedback_y.clear()
    return scores


def predict(query_data):
    """Return the predicted class name for one flower."""
    x = [getattr(query_data, f) for f in FEATURES]
    prediction = clf.predict([x])[0]
    return classes[int(prediction)]


def retrain(data):
    """Add the labelled feedback to what the model already knows and refit.

    Refitting on the feedback alone would throw away the original training data
    (and fail outright on a batch that contains a single class).
    """
    for d in data:
        feedback_X.append([getattr(d, f) for f in FEATURES])
        feedback_y.append(r_classes[d.flower_class])
    clf.fit(X_base + feedback_X, y_base + feedback_y)
