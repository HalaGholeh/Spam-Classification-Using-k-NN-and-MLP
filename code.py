import numpy as np
import csv
import sys
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.neural_network import MLPClassifier

TEST_SIZE = 0.3
K = 3

class NN:
    def __init__(self, trainingFeatures, trainingLabels):
        self.trainingFeatures = trainingFeatures
        self.trainingLabels = trainingLabels


    def predict(self, features, k):
        """
        Given a list of feature vectors of testing examples,
        return the predicted class labels (list of either 0s or 1s)
        using the k-nearest neighbors algorithm.
        """
        predictions = []  # Initialize prediction list
        for row in features:
            distances = [np.linalg.norm(row - trainFeature) for trainFeature in self.trainingFeatures]
            indicesASC = np.argsort(distances)
            KNN = indicesASC[:k]
            KNN = [self.trainingLabels[idx] for idx in KNN]
            predicted_label = max(set(KNN), key=KNN.count)
            predictions.append(predicted_label)
        return predictions


def load_data(filename):
    """
    Load spam data from a CSV file `filename` and convert it into a list of
    feature vectors and a list of target labels. Return a tuple (features, labels).

    Feature vectors should be a list of lists, where each list contains the
    feature values.

    Labels should be the corresponding list of labels, where each label
    is 1 if spam and 0 otherwise.
    """
    features = []
    labelList = []

    with open(filename, 'r') as file:
        csv_reader = csv.reader(file)

        for row in csv_reader:
            feature_vector = [float(value) for value in row[:-1]]
            features.append(feature_vector)
            label = int(row[-1])
            labelList.append(label)

    return features, labelList


def preprocess(features):
    """
    Normalize each feature by subtracting the mean value in each
    feature and dividing by the standard deviation.
    """
    features = np.array(features)
    means = np.mean(features, axis=0)
    standardDiv = np.std(features, axis=0)
    features = (features - means) / standardDiv  # Assign the normalized features back to the variable
    return features


def train_mlp_model(features, labels):
    """
    Given a list of feature vectors and a list of labels, return a
    fitted MLP model trained on the data using the sklearn implementation.
    """
    model = MLPClassifier()
    model.fit(features, labels)
    return model


def evaluate(labels, predictions):
    """
    Given a list of actual labels and a list of predicted labels,
    return (accuracy, precision, recall, f1).

    Assume each label is either a 1 (positive) or 0 (negative).
    """
    accuracy = accuracy_score(labels, predictions)
    precision = precision_score(labels, predictions)
    recall = recall_score(labels, predictions)
    f1 = f1_score(labels, predictions)
    return accuracy, precision, recall, f1


def main():
    # Check command-line arguments
    if len(sys.argv) != 2:
        sys.exit("Usage: python template.py ./spambase.csv")

    # Load data from spreadsheet and split into train and test sets
    features, labels = load_data(sys.argv[1])
    features = preprocess(features)
    X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=TEST_SIZE)

    # Train a k-NN model and make predictions
    model_nn = NN(X_train, y_train)
    predictions = model_nn.predict(X_test, K)
    accuracy, precision, recall, f1 = evaluate(y_test, predictions)


    # Print results
    print("** k-NN Results **")
    print("Accuracy:", accuracy)
    print("Precision:", precision)
    print("Recall:", recall)
    print("F1:", f1)

    error = np.array(predictions) - np.array(y_test)
    print("Error: " ,error)
    print("predictions : " , predictions)
    print("YTst" , y_test)


    # Train an MLP model and make predictions
    model_mlp = train_mlp_model(X_train, y_train)
    predictions = model_mlp.predict(X_test)
    accuracy, precision, recall, f1 = evaluate(y_test, predictions)

    # Print results
    print("** MLP Results **")
    print("Accuracy:", accuracy)
    print("Precision:", precision)
    print("Recall:", recall)
    print("F1:", f1)

    print(predictions)

    error1 = np.array(predictions) - np.array(y_test)
    print("Error: " ,error1)

    print("predictions : ", predictions)
    print("YTst", y_test)




if __name__ == "__main__":
    main()
