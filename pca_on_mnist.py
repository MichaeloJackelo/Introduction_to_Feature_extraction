from keras.datasets import mnist
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import KFold
from statistics import mean
from sklearn.svm import SVC
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn.tree import DecisionTreeClassifier
import time


f1_tag = 'f1_score'
training_time_tag = 'training_time'
testing_time_tag = 'testing_time'


def make_classification(amount_variance, classifier, features, labels):
    kf = KFold(n_splits=10)

    f1_scores = []
    accuracy_scores = []
    training_last = []
    testing_last = []
    scaler = StandardScaler()
    pca = PCA(amount_variance)

    for train_index, test_index in kf.split(features):
        X_train, X_test = features[train_index], features[test_index]
        y_train, y_test = labels[train_index], labels[test_index]
        scaler.fit(X_train)
        X_train = scaler.transform(X_train)
        X_test = scaler.transform(X_test)
        pca.fit(X_train)
        X_train = pca.transform(X_train)
        X_test = pca.transform(X_test)
        start_training_time = time.time()
        classifier.fit(X_train, y_train)
        end_training_time = time.time()

        start_testing_time = time.time()
        predicted_labels = classifier.predict(X_test)
        end_testing_time = time.time()

        f1_scores.append(f1_score(y_test, predicted_labels, average='macro'))
        accuracy_scores.append(accuracy_score(y_test, predicted_labels))
        training_last.append(end_training_time - start_training_time)
        testing_last.append(end_testing_time - start_testing_time)

    f1_result = mean(f1_scores)
    accuracy = mean(accuracy_scores)
    testing_time = mean(testing_last)
    training_time = mean(training_last)
    return {f1_tag: f1_result, training_time_tag: training_time, testing_time_tag: testing_time}


# input image dimensions
img_rows, img_cols = 28, 28

# the data, split between train and test sets
(x_train, y_train), (x_test, y_test) = mnist.load_data()
features = np.concatenate((x_train, x_test))
labels = np.concatenate((y_train, y_test))

features = features.reshape(70000, 784)

classifiers = {
    "LogisticRegression": LogisticRegression(solver='lbfgs'),
    "KNeighborsClassifier": KNeighborsClassifier(3),
    "SVC": SVC(kernel="linear", C=0.025),
    "GaussianProcessClassifier": GaussianProcessClassifier(1.0 * RBF(1.0)),
    "DecisionTreeClassifier": DecisionTreeClassifier(max_depth=5)}

precentages_of_variance = np.linspace(0.70, 1, 15, endpoint=False)

fig, axes = plt.subplots(1, 3)
for classifier_name, classifier in classifiers.items():
    results_per_classifier = []
    for amount_variance in precentages_of_variance:
        results = make_classification(amount_variance, classifier, features, labels)
        results_per_classifier.append(results)

    f1 = [x[f1_tag] for x in results_per_classifier]
    train_time = [x[training_time_tag] for x in results_per_classifier]
    test_time = [x[testing_time_tag] for x in results_per_classifier]

    axes[0].set_title('Average training time depending of PCA amount of variance')
    axes[0].plot(precentages_of_variance, train_time, label=classifier_name)
    axes[0].legend()
    axes[0].set_xlabel("Amount of variance")
    axes[0].set_ylabel("training time")
    axes[1].set_title('Average testing time depending of PCA amount of variance')
    axes[1].plot(precentages_of_variance, test_time, label=classifier_name)
    axes[1].legend()
    axes[1].set_xlabel("Amount of variance")
    axes[1].set_ylabel("testing time")
    axes[2].set_title('Average F1 score time depending of PCA amount of variance')
    axes[2].plot(precentages_of_variance, f1, label=classifier_name)
    axes[2].legend()
    axes[2].set_xlabel("Amount of variance")
    axes[2].set_ylabel("F1 score")


plt.savefig('results')
