import pandas as pd
import mlflow
import mlflow.sklearn

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.ensemble import RandomForestClassifier

import mlflow

import dagshub
dagshub.init(repo_owner='arshpreetsingh-01', repo_name='mlflow-dagshub-demo', mlflow=True)

# Change this from your sqlite URI to the local server
mlflow.set_tracking_uri("https://dagshub.com/arshpreetsingh-01/mlflow-dagshub-demo.mlflow")

import matplotlib.pyplot as plt
import seaborn as sns
import mlflow
mlflow.autolog()
#

# Load the Iris dataset
iris = load_iris()
X = iris.data
y = iris.target

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
X, y, test_size=0.2, random_state=42
)

# Train a Decision Tree classifier
max_depth = 8
n_estimators = 10

mlflow.set_experiment("iris_rf")

# Apply mlflow to train
# if you are writing code in mlflow.start_run you don't need to write mlflow.end run 
with mlflow.start_run(run_name="pk_exp_with_confusion_matrix_log_artifact"):
    mlflow.set_tag("mlflow.user", "arsh")
    mlflow.log_param("max_depth", max_depth)
    mlflow.log_param("n_estimators", n_estimators)

    rf = RandomForestClassifier(max_depth=max_depth, n_estimators=n_estimators)

    rf.fit(X_train,y_train)

    # Evaluate the model
    y_pred = rf.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)

    mlflow.log_metric("accuracy", accuracy)
    print("accuracy:", accuracy)

    # Log confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(6, 6))
    sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=iris.target_names,
    yticklabels=iris.target_names,

    )

    plt.ylabel("Actual")
    plt.xlabel("Predicted")
    plt.title("Confusion Matrix")

    # Save the confusion matrix
    plt.savefig("confusion_matrix.png")
    mlflow.log_artifact("confusion_matrix.png")

    # Log the model
    mlflow.log_artifact(__file__)
    mlflow.sklearn.log_model(rf, "random_forest_model")

    mlflow.set_tag("author", "arsh")
    mlflow.set_tag("project", "iris-classification")
    mlflow.set_tag("algorithm", "random_forest")

