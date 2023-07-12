import numpy as np
import csv
import sys
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

TEST_SIZE = 0.3
K = 3

class NN:
    def __init__(self, trainingFeatures, trainingLabels) -> None:
        self.trainingFeatures = trainingFeatures
        self.trainingLabels = trainingLabels

    def predict(self, features, k):
        """
        Given a list of features vectors of testing examples
        return the predicted class labels (list of either 0s or 1s)
        using the k nearest neighbors
        """
        pridiction = [] #initize pridiction list
        for row in features:
            distances = [np.linalg.norm(row - trainFeature) for trainFeature in self.trainingFeatures]  #calulate distance between each test
            indicesASC = np.argsort(distances)  #Sort the indices of examples based on distances ASC
            KNN = indicesASC[:k]   #Select the indices of the k_nearest_indices
            KNN = [self.trainingLabels[idx] for idx in KNN]  #find the labels of 3 nearst indices
            predicted_label = max(set(KNN), key=KNN.count) #predict the label

            pridiction.append(predicted_label)  #add pridiction label to the pridiction list
            return pridiction

        #raise NotImplementedError


def load_data(filename):
    """
    Load spam data from a CSV file `filename` and convert into a list of
    features vectors and a list of target labels. Return a tuple (features, labels).

    features vectors should be a list of lists, where each list contains the
    57 features vectors

    labels should be the corresponding list of labels, where each label
    is 1 if spam, and 0 otherwise.
    """
    features = []      #save the date of csv file in List named attributesList
    lableList = []           #save the output of csv file which is the last value in the file (spame = 0, not=1)

    with open(filename, 'r') as file:
        csv_reader = csv.reader(file)

        for row in csv_reader:
            #add vlaue of the featureVector to the list
            featureVector = [float(value) for value in row[:-1]]
            features.append(featureVector)   #input List
            #add vlaue of label to the lable list
            label = int(row[-1])
            lableList.append(label)  #output List

    return  features , lableList

    #raise NotImplementedError


def preprocess(features):
    """
    normalize each feature by subtracting the mean value in each
    feature and dividing by the standard deviation
    """
    ##𝑓𝑖 =(𝑓𝑖 − 𝑓𝑖̅)/𝜎𝑖  this is the formula than we want to calculate
    features = np.array(features)  #convert features to numpy array
    means = np.mean(features , axis=0)     #calculate means os the features (average of the features)
    standardDiv = np.std(features, axis=0)     #calculate standerd division
    Fi = (features - means) / standardDiv    #finally find 𝑓𝑖 =(𝑓𝑖 − 𝑓𝑖̅)/𝜎𝑖
    #raise NotImplementedError

def train_mlp_model(features, labels):
    """
    Given a list of features lists and a list of labels, return a
    fitted MLP model trained on the data using sklearn implementation.
    """
    raise NotImplementedError


def evaluate(labels, predictions):
    """
    Given a list of actual labels and a list of predicted labels,
    return (accuracy, precision, recall, f1).

    Assume each label is either a 1 (positive) or 0 (negative).
    """
    #calculate accurancy , precision, recall and f1
    accuracy = accuracy_score(labels, predictions)
    precision = precision_score(labels, predictions)
    recall = recall_score(labels, predictions)
    f1 = f1_score(labels, predictions)
    return accuracy, precision, recall, f1
    #raise NotImplementedError


def main():
    load_data("spambase.csv")
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
    print("**** 1-Nearest Neighbor Results ****")
    print("Accuracy: ", accuracy)
    print("Precision: ", precision)
    print("Recall: ", recall)
    print("F1: ", f1)

    # Train an MLP model and make predictions
    model = train_mlp_model(X_train, y_train)
    predictions = model.predict(X_test)
    accuracy, precision, recall, f1 = evaluate(y_test, predictions)

    # Print results
    print("**** MLP Results ****")
    print("Accuracy: ", accuracy)
    print("Precision: ", precision)
    print("Recall: ", recall)
    print("F1: ", f1)



if __name__ == "__main__":
    main()


