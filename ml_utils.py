from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score
from sklearn.tree import DecisionTreeClassifier

# define a Gaussain NB classifier
clf = GaussianNB()

# Create Decision Tree on the training data
clf1 = DecisionTreeClassifier(criterion='gini', splitter='best', max_depth=None, min_samples_split=2)

# define the class encodings and reverse encodings
classes = {0: "Iris Setosa", 1: "Iris Versicolour", 2: "Iris Virginica"}
r_classes = {y: x for x, y in classes.items()}

# function to train and load the model during startup
def load_model():
    # load the dataset from the official sklearn datasets
    X, y = datasets.load_iris(return_X_y=True)

    # do the test-train split and train the model
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    clf.fit(X_train, y_train)
    clf1.fit(X_train, y_train)
    # calculate the print the accuracy score
    acc = accuracy_score(y_test, clf.predict(X_test))
    acc1 = accuracy_score(y_test, clf1.predict(X_test))
    print(f"Gaussian Model trained with accuracy: {round(acc, 3)}")
    print(f"DecisionTree Model trained with accuracy: {round(acc, 3)}")

# function to predict the flower using the model
def predict(query_data):
    x = list(query_data.dict().values())
    prediction = clf.predict([x])[0]
    print(f"Model prediction: {classes[prediction]}")
    return classes[prediction]

# function to retrain the model as part of the feedback loop
def retrain(data):
    # pull out the relevant X and y from the FeedbackIn object
    X = [list(d.dict().values())[:-1] for d in data]
    y = [r_classes[d.flower_class] for d in data]

    # fit the classifier again based on the new data obtained
    clf.fit(X, y)
