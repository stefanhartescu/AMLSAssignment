import landmarks as l2
import numpy as np
from sklearn import svm
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.tree import DecisionTreeClassifier
import csv
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier


def get_features():
    X = l2.extract_features_labels()
    return X


X = get_features()


def divide_x(no_samples):
    trainingX = X[:no_samples]
    testingX = X[no_samples:]
    return trainingX, testingX


def get_labels(i, test_samples):
    Y = l2.extract_labels(i)
    trainingY = Y[:test_samples]
    testingY = Y[test_samples:]
    return trainingY, testingY


def SVM_eyeglasses(x_tr, y_tr, x_te):
    clf = svm.SVC(C=1, degree=2, gamma='scale', kernel='poly')
    clf.fit(x_tr, y_tr)
    return np.ravel(clf.predict(x_te))


def SVM_smile(x_tr, y_tr, x_te):
    clf = svm.SVC(C=1, degree=2, gamma='scale', kernel='poly')
    clf.fit(x_tr, y_tr)
    return np.ravel(clf.predict(x_te))


def SVM_age(x_tr, y_tr, x_te):
    clf = svm.SVC(C=2, degree=2, gamma='scale', kernel='rbf')
    clf.fit(x_tr, y_tr)
    return np.ravel(clf.predict(x_te))


def SVM_human(x_tr, y_tr, x_te):
    clf = svm.SVC(C=1, degree=2, gamma='scale', kernel='poly')
    clf.fit(x_tr, y_tr)
    return np.ravel(clf.predict(x_te))

def DTree_smile(x_tr, y_tr, x_te):
    dtree_model = DecisionTreeClassifier(max_depth=3).fit(x_tr, y_tr)
    dtree_prediction = dtree_model.predict(x_te)
    return np.ravel(dtree_prediction)

def DTree_age(x_tr, y_tr, x_te):
    dtree_model = DecisionTreeClassifier(max_depth=3).fit(x_tr, y_tr)
    dtree_prediction = dtree_model.predict(x_te)
    return np.ravel(dtree_prediction)

def DTree_eyeglasses(x_tr, y_tr, x_te):
    from sklearn.tree import DecisionTreeClassifier
    dtree_model = DecisionTreeClassifier(max_depth=3).fit(x_tr, y_tr)
    dtree_prediction = dtree_model.predict(x_te)
    return np.ravel(dtree_prediction)

def DTree_human(x_tr, y_tr, x_te):
    from sklearn.tree import DecisionTreeClassifier
    dtree_model = DecisionTreeClassifier(max_depth=3).fit(x_tr, y_tr)
    dtree_prediction = dtree_model.predict(x_te)
    return np.ravel(dtree_prediction)

def DTree_haircolor(x_tr, y_tr, x_te):
    from sklearn.tree import DecisionTreeClassifier
    dtree_model = DecisionTreeClassifier(max_depth=7,max_features=7).fit(x_tr, y_tr)
    dtree_prediction = dtree_model.predict(x_te)
    return np.ravel(dtree_prediction)

def KNN_haircolor(x_tr, y_tr, x_te):
    knn = KNeighborsClassifier(n_neighbors=6).fit(x_tr, y_tr)
    knn_predictions = knn.predict(x_te)
    return knn_predictions


def classify_smile(test_samples):
    Xtrain, Xtest = divide_x(test_samples)
    Xtrain = Xtrain.reshape(Xtrain.shape[0], Xtrain.shape[1] * Xtrain.shape[2])
    Xtest = Xtest.reshape(Xtest.shape[0], Xtest.shape[1] * Xtest.shape[2])
    Ytrain, Ytest = get_labels(3, test_samples)
    Y_pred_te = SVM_smile(Xtrain, Ytrain, Xtest)
    size_te = len(Y_pred_te)
    Y_pred_tr = SVM_smile(Xtrain, Ytrain, Xtrain)
    size_tr = len(Y_pred_tr)
    test_accuracy = accuracy_score(Ytest, Y_pred_te) * 100
    train_accuracy = accuracy_score(Ytrain, Y_pred_tr) * 100
    print("Accuracy obtained on test data for smile detection:")
    print(accuracy_score(Ytest, Y_pred_te) * 100, '%')
    print("Accuracy obtained on train data for smile detection:")
    print(accuracy_score(Ytrain, Y_pred_tr) * 100, '%')
    print("The confusion matrix is:")
    print(confusion_matrix(Ytest, Y_pred_te))
    Y_pred = np.concatenate((Y_pred_tr, Y_pred_te), axis=0)
    size_pred = len(Y_pred)
    np.savetxt("task_2labels.csv", Y_pred, delimiter=',')
    precision = (test_accuracy * size_te) + (train_accuracy * size_tr)
    print(precision / size_pred)
    percentage = [precision / size_pred]
    np.savetxt("task_1precision.csv", percentage, delimiter=',')
    a = pd.read_csv("noise_classified.csv")
    b = pd.read_csv("task_1labels.csv")
    c = pd.read_csv("task_1precision.csv")
    merged = pd.concat([a, b], axis=1)
    merged.to_csv("task_1.csv", index=False)
    labelled = pd.read_csv("task_1.csv")
    final = pd.concat([labelled, c], axis=1)
    final.to_csv("task_1.csv", index=False)


def classify_age(test_samples):
    Xtrain, Xtest = divide_x(test_samples)
    Xtrain = Xtrain.reshape(Xtrain.shape[0], Xtrain.shape[1] * Xtrain.shape[2])
    Xtest = Xtest.reshape(Xtest.shape[0], Xtest.shape[1] * Xtest.shape[2])
    Ytrain, Ytest = get_labels(4, test_samples)
    Y_pred_te = SVM_age(Xtrain, Ytrain, Xtest)
    size_te = len(Y_pred_te)
    Y_pred_tr = SVM_age(Xtrain, Ytrain, Xtrain)
    size_tr = len(Y_pred_tr)
    test_accuracy = accuracy_score(Ytest, Y_pred_te) * 100
    train_accuracy = accuracy_score(Ytrain, Y_pred_tr) * 100
    print("Accuracy obtained on test data for age detection:")
    print(accuracy_score(Ytest, Y_pred_te) * 100, '%')
    print("Accuracy obtained on train data for age detection:")
    print(accuracy_score(Ytrain, Y_pred_tr) * 100, '%')
    print("The confusion matrix is:")
    print(confusion_matrix(Ytest, Y_pred_te))
    Y_pred = np.concatenate((Y_pred_tr, Y_pred_te), axis=0)
    size_pred = len(Y_pred)
    np.savetxt("task_2labels.csv", Y_pred, delimiter=',')
    precision = (test_accuracy * size_te) + (train_accuracy * size_tr)
    print(precision / size_pred)
    percentage = [precision / size_pred]
    np.savetxt("task_2precision.csv", percentage, delimiter=',')
    a = pd.read_csv("noise_classified.csv")
    b = pd.read_csv("task_2labels.csv")
    c = pd.read_csv("task_2precision.csv")
    merged = pd.concat([a, b], axis=1)
    merged.to_csv("task_2.csv", index=False)
    labelled = pd.read_csv("task_2.csv")
    final = pd.concat([labelled, c], axis=1)
    final.to_csv("task_2.csv", index=False)


def classify_eyeglasses(test_samples):
    Xtrain, Xtest = divide_x(test_samples)
    Xtrain = Xtrain.reshape(Xtrain.shape[0], Xtrain.shape[1] * Xtrain.shape[2])
    Xtest = Xtest.reshape(Xtest.shape[0], Xtest.shape[1] * Xtest.shape[2])
    Ytrain, Ytest = get_labels(2, test_samples)
    Y_pred_te = SVM_eyeglasses(Xtrain, Ytrain, Xtest)
    size_te = len(Y_pred_te)
    Y_pred_tr = SVM_eyeglasses(Xtrain, Ytrain, Xtrain)
    size_tr = len(Y_pred_tr)
    test_accuracy = accuracy_score(Ytest, Y_pred_te) * 100
    train_accuracy = accuracy_score(Ytrain, Y_pred_tr) * 100
    print("Accuracy obtained on test data for eyeglass detection:")
    print(accuracy_score(Ytest, Y_pred_te) * 100, '%')
    print("Accuracy obtained on train data for eyeglass detection:")
    print(accuracy_score(Ytrain, Y_pred_tr) * 100, '%')
    print("The confusion matrix is:")
    print(confusion_matrix(Ytest, Y_pred_te))
    Y_pred = np.concatenate((Y_pred_tr, Y_pred_te), axis=0)
    size_pred = len(Y_pred)
    np.savetxt("task_3labels.csv", Y_pred, delimiter=',')
    precision = (test_accuracy * size_te) + (train_accuracy * size_tr)
    print(precision / size_pred)
    percentage = [precision / size_pred]
    np.savetxt("task_3precision.csv", percentage, delimiter=',')
    a = pd.read_csv("noise_classified.csv")
    b = pd.read_csv("task_3labels.csv")
    c = pd.read_csv("task_3precision.csv")
    merged = pd.concat([a, b], axis=1)
    merged.to_csv("task_3.csv", index=False)
    labelled = pd.read_csv("task_3.csv")
    final = pd.concat([labelled, c], axis=1)
    final.to_csv("task_3.csv", index=False)


def classify_human(test_samples):
    Xtrain, Xtest = divide_x(test_samples)
    Xtrain = Xtrain.reshape(Xtrain.shape[0], Xtrain.shape[1] * Xtrain.shape[2])
    Xtest = Xtest.reshape(Xtest.shape[0], Xtest.shape[1] * Xtest.shape[2])
    Ytrain, Ytest = get_labels(5, test_samples)
    Y_pred_te = SVM_human(Xtrain, Ytrain, Xtest)
    size_te = len(Y_pred_te)
    Y_pred_tr = SVM_human(Xtrain, Ytrain, Xtrain)
    size_tr = len(Y_pred_tr)
    test_accuracy = accuracy_score(Ytest, Y_pred_te) * 100
    train_accuracy = accuracy_score(Ytrain, Y_pred_tr) * 100
    print("Accuracy obtained on test data for human detection:")
    print(accuracy_score(Ytest, Y_pred_te) * 100, '%')
    print("Accuracy obtained on train data for human detection:")
    print(accuracy_score(Ytrain, Y_pred_tr) * 100, '%')
    print("The confusion matrix is:")
    print(confusion_matrix(Ytest, Y_pred_te))
    Y_pred = np.concatenate((Y_pred_tr, Y_pred_te), axis=0)
    size_pred = len(Y_pred)
    np.savetxt("task_4labels.csv", Y_pred, delimiter=',')
    precision = (test_accuracy * size_te) + (train_accuracy * size_tr)
    print(precision / size_pred)
    percentage = [precision / size_pred]
    np.savetxt("task_4precision.csv", percentage, delimiter=',')
    a = pd.read_csv("noise_classified.csv")
    b = pd.read_csv("task_4labels.csv")
    c = pd.read_csv("task_4precision.csv")
    merged = pd.concat([a, b], axis=1)
    merged.to_csv("task_4.csv", index=False)
    labelled = pd.read_csv("task_4.csv")
    final = pd.concat([labelled, c], axis=1)
    final.to_csv("task_4.csv", index=False)

def classify_smile_dtree(test_samples):
    Xtrain, Xtest = divide_x(test_samples)
    Xtrain = Xtrain.reshape(Xtrain.shape[0], Xtrain.shape[1] * Xtrain.shape[2])
    Xtest = Xtest.reshape(Xtest.shape[0], Xtest.shape[1] * Xtest.shape[2])
    Ytrain, Ytest = get_labels(3, test_samples)
    Y_pred_te = DTree_smile(Xtrain, Ytrain, Xtest)
    size_te = len(Y_pred_te)
    Y_pred_tr = DTree_smile(Xtrain, Ytrain, Xtrain)
    size_tr = len(Y_pred_tr)
    test_accuracy = accuracy_score(Ytest, Y_pred_te) * 100
    train_accuracy = accuracy_score(Ytrain, Y_pred_tr) * 100
    print("Accuracy obtained on test data for smile detection:")
    print(accuracy_score(Ytest, Y_pred_te) * 100, '%')
    print("Accuracy obtained on train data for smile detection:")
    print(accuracy_score(Ytrain, Y_pred_tr) * 100, '%')
    print("The confusion matrix is:")
    print(confusion_matrix(Ytest, Y_pred_te))
    Y_pred = np.concatenate((Y_pred_tr, Y_pred_te), axis=0)
    size_pred = len(Y_pred)
    np.savetxt("task_1labelsdtree.csv", Y_pred, delimiter=',')
    precision = (test_accuracy * size_te) + (train_accuracy * size_tr)
    print(precision / size_pred)
    percentage = [precision / size_pred]
    np.savetxt("task_1dtreeprecision.csv", percentage, delimiter=',')
    a = pd.read_csv("noise_classified.csv")
    b = pd.read_csv("task_1labelsdtree.csv")
    c = pd.read_csv("task_1dtreeprecision.csv")
    merged = pd.concat([a, b], axis=1)
    merged.to_csv("task_1dtree.csv", index=False)
    labelled = pd.read_csv("task_1dtree.csv")
    final = pd.concat([labelled, c], axis=1)
    final.to_csv("task_1dtree.csv", index=False)

def classify_age_dtree(test_samples):
    Xtrain, Xtest = divide_x(test_samples)
    Xtrain = Xtrain.reshape(Xtrain.shape[0], Xtrain.shape[1] * Xtrain.shape[2])
    Xtest = Xtest.reshape(Xtest.shape[0], Xtest.shape[1] * Xtest.shape[2])
    Ytrain, Ytest = get_labels(4, test_samples)
    Y_pred_te = DTree_age(Xtrain, Ytrain, Xtest)
    size_te = len(Y_pred_te)
    Y_pred_tr = DTree_age(Xtrain, Ytrain, Xtrain)
    size_tr = len(Y_pred_tr)
    test_accuracy = accuracy_score(Ytest, Y_pred_te) * 100
    train_accuracy = accuracy_score(Ytrain, Y_pred_tr) * 100
    print("Accuracy obtained on test data for age detection:")
    print(accuracy_score(Ytest, Y_pred_te) * 100, '%')
    print("Accuracy obtained on train data for age detection:")
    print(accuracy_score(Ytrain, Y_pred_tr) * 100, '%')
    print("The confusion matrix is:")
    print(confusion_matrix(Ytest, Y_pred_te))
    Y_pred = np.concatenate((Y_pred_tr, Y_pred_te), axis=0)
    size_pred = len(Y_pred)
    np.savetxt("task_2labelsdtree.csv", Y_pred, delimiter=',')
    precision = (test_accuracy * size_te) + (train_accuracy * size_tr)
    print(precision / size_pred)
    percentage = [precision / size_pred]
    np.savetxt("task_2dtreeprecision.csv", percentage, delimiter=',')
    a = pd.read_csv("noise_classified.csv")
    b = pd.read_csv("task_2labelsdtree.csv")
    c = pd.read_csv("task_2dtreeprecision.csv")
    merged = pd.concat([a, b], axis=1)
    merged.to_csv("task_2dtree.csv", index=False)
    labelled = pd.read_csv("task_2dtree.csv")
    final = pd.concat([labelled, c], axis=1)
    final.to_csv("task_2dtree.csv", index=False)

def classify_eyeglasses_dtree(test_samples):
    Xtrain, Xtest = divide_x(test_samples)
    Xtrain = Xtrain.reshape(Xtrain.shape[0], Xtrain.shape[1] * Xtrain.shape[2])
    Xtest = Xtest.reshape(Xtest.shape[0], Xtest.shape[1] * Xtest.shape[2])
    Ytrain, Ytest = get_labels(2, test_samples)
    Y_pred_te = DTree_eyeglasses(Xtrain, Ytrain, Xtest)
    size_te = len(Y_pred_te)
    Y_pred_tr = DTree_eyeglasses(Xtrain, Ytrain, Xtrain)
    size_tr = len(Y_pred_tr)
    test_accuracy = accuracy_score(Ytest, Y_pred_te) * 100
    train_accuracy = accuracy_score(Ytrain, Y_pred_tr) * 100
    print("Accuracy obtained on test data for eyeglass detection:")
    print(accuracy_score(Ytest, Y_pred_te) * 100, '%')
    print("Accuracy obtained on train data for eyeglass detection:")
    print(accuracy_score(Ytrain, Y_pred_tr) * 100, '%')
    print("The confusion matrix is:")
    print(confusion_matrix(Ytest, Y_pred_te))
    Y_pred = np.concatenate((Y_pred_tr, Y_pred_te), axis=0)
    size_pred = len(Y_pred)
    np.savetxt("task_3labelsdtree.csv", Y_pred, delimiter=',')
    precision = (test_accuracy * size_te) + (train_accuracy * size_tr)
    print(precision / size_pred)
    percentage = [precision / size_pred]
    np.savetxt("task_3dtreeprecision.csv", percentage, delimiter=',')
    a = pd.read_csv("noise_classified.csv")
    b = pd.read_csv("task_3labelsdtree.csv")
    c = pd.read_csv("task_3dtreeprecision.csv")
    merged = pd.concat([a, b], axis=1)
    merged.to_csv("task_3dtree.csv", index=False)
    labelled = pd.read_csv("task_3dtree.csv")
    final = pd.concat([labelled, c], axis=1)
    final.to_csv("task_3dtree.csv", index=False)

def classify_human_dtree(test_samples):
    Xtrain, Xtest = divide_x(test_samples)
    Xtrain = Xtrain.reshape(Xtrain.shape[0], Xtrain.shape[1] * Xtrain.shape[2])
    Xtest = Xtest.reshape(Xtest.shape[0], Xtest.shape[1] * Xtest.shape[2])
    Ytrain, Ytest = get_labels(5, test_samples)
    Y_pred_te = DTree_human(Xtrain, Ytrain, Xtest)
    size_te = len(Y_pred_te)
    Y_pred_tr = DTree_human(Xtrain, Ytrain, Xtrain)
    size_tr = len(Y_pred_tr)
    test_accuracy = accuracy_score(Ytest, Y_pred_te) * 100
    train_accuracy = accuracy_score(Ytrain, Y_pred_tr) * 100
    print("Accuracy obtained on test data for human detection:")
    print(accuracy_score(Ytest, Y_pred_te) * 100, '%')
    print("Accuracy obtained on train data for human detection:")
    print(accuracy_score(Ytrain, Y_pred_tr) * 100, '%')
    print("The confusion matrix is:")
    print(confusion_matrix(Ytest, Y_pred_te))
    Y_pred = np.concatenate((Y_pred_tr, Y_pred_te), axis=0)
    size_pred = len(Y_pred)
    np.savetxt("task_4labelsdtree.csv", Y_pred, delimiter=',')
    precision = (test_accuracy * size_te) + (train_accuracy * size_tr)
    print(precision / size_pred)
    percentage = [precision / size_pred]
    np.savetxt("task_4dtreeprecision.csv", percentage, delimiter=',')
    a = pd.read_csv("noise_classified.csv")
    b = pd.read_csv("task_4labelsdtree.csv")
    c = pd.read_csv("task_4dtreeprecision.csv")
    merged = pd.concat([a, b], axis=1)
    merged.to_csv("task_4dtree.csv", index=False)
    labelled = pd.read_csv("task_4dtree.csv")
    final = pd.concat([labelled, c], axis=1)
    final.to_csv("task_4dtree.csv", index=False)

def classify_haircolor_dtree(test_samples):
    Xtrain, Xtest = divide_x(test_samples)
    Xtrain = Xtrain.reshape(Xtrain.shape[0], Xtrain.shape[1] * Xtrain.shape[2])
    Xtest = Xtest.reshape(Xtest.shape[0], Xtest.shape[1] * Xtest.shape[2])
    Ytrain, Ytest = get_labels(1, test_samples)
    Y_pred_te = DTree_haircolor(Xtrain, Ytrain*2, Xtest)
    size_te = len(Y_pred_te)
    Y_pred_tr = DTree_haircolor(Xtrain, Ytrain*2, Xtrain)
    size_tr = len(Y_pred_tr)
    test_accuracy = accuracy_score(Ytest*2, Y_pred_te) * 100
    train_accuracy = accuracy_score(Ytrain*2, Y_pred_tr) * 100
    print("Accuracy obtained on test data for hair color detection:")
    print(accuracy_score(Ytest*2, Y_pred_te) * 100, '%')
    print("Accuracy obtained on train data for hair color detection:")
    print(accuracy_score(Ytrain*2, Y_pred_tr) * 100, '%')
    print("The confusion matrix is:")
    print(confusion_matrix(Ytest*2, Y_pred_te))
    Y_pred = np.concatenate((Y_pred_tr, Y_pred_te), axis=0)
    size_pred = len(Y_pred)
    np.savetxt("task_5labelsdtree.csv", Y_pred, delimiter=',')
    precision = (test_accuracy * size_te) + (train_accuracy * size_tr)
    print(precision / size_pred)
    percentage = [precision / size_pred]
    np.savetxt("task_5dtreeprecision.csv", percentage, delimiter=',')
    a = pd.read_csv("noise_classified.csv")
    b = pd.read_csv("task_5labelsdtree.csv")
    c = pd.read_csv("task_5dtreeprecision.csv")
    merged = pd.concat([a, b], axis=1)
    merged.to_csv("task_5dtree.csv", index=False)
    labelled = pd.read_csv("task_5dtree.csv")
    final = pd.concat([labelled, c], axis=1)
    final.to_csv("task_5dtree.csv", index=False)


def classify_haircolor_KNN(test_samples):
    Xtrain, Xtest = divide_x(test_samples)
    Xtrain = Xtrain.reshape(Xtrain.shape[0], Xtrain.shape[1] * Xtrain.shape[2])
    Xtest = Xtest.reshape(Xtest.shape[0], Xtest.shape[1] * Xtest.shape[2])
    Ytrain, Ytest = get_labels(1, test_samples)
    Y_pred_te = KNN_haircolor(Xtrain, Ytrain*2, Xtest)
    size_te = len(Y_pred_te)
    Y_pred_tr = KNN_haircolor(Xtrain, Ytrain*2, Xtrain)
    size_tr = len(Y_pred_tr)
    test_accuracy = accuracy_score(Ytest*2, Y_pred_te) * 100
    train_accuracy = accuracy_score(Ytrain*2, Y_pred_tr) * 100
    print("Accuracy obtained on test data for hair color detection:")
    print(accuracy_score(Ytest*2, Y_pred_te) * 100, '%')
    print("Accuracy obtained on train data for hair color detection:")
    print(accuracy_score(Ytrain*2, Y_pred_tr) * 100, '%')
    print("The confusion matrix is:")
    print(confusion_matrix(Ytest*2, Y_pred_te))
    Y_pred = np.concatenate((Y_pred_tr, Y_pred_te), axis=0)
    size_pred = len(Y_pred)
    np.savetxt("task_5labelsKNN.csv", Y_pred, delimiter=',')
    precision = (test_accuracy * size_te) + (train_accuracy * size_tr)
    print(precision / size_pred)
    percentage = [precision / size_pred]
    np.savetxt("task_5KNNprecision.csv", percentage, delimiter=',')
    a = pd.read_csv("noise_classified.csv")
    b = pd.read_csv("task_5labelsKNN.csv")
    c = pd.read_csv("task_5KNNprecision.csv")
    merged = pd.concat([a, b], axis=1)
    merged.to_csv("task_5KNN.csv", index=False)
    labelled = pd.read_csv("task_5KNN.csv")
    final = pd.concat([labelled, c], axis=1)
    final.to_csv("task_5KNN.csv", index=False)

classify_smile(4000)
classify_smile_dtree(4000)
classify_age(4000)
classify_age_dtree(4000)
classify_eyeglasses(4000)
classify_eyeglasses_dtree(4000)
classify_human(4000)
classify_human_dtree(4000)
classify_haircolor_dtree(3000)
classify_haircolor_KNN(4000)

