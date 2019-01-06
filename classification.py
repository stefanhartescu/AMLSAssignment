import landmarks as l2
import numpy as np
from sklearn import svm
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
import csv
import pandas as pd

def get_features():
    X = l2.extract_features_labels()
    return X

X = get_features()

def divide_x(no_samples):
    trainingX = X[:no_samples]
    testingX = X[no_samples:]
    return trainingX, testingX

def get_labels(i,test_samples):
     Y = l2.extract_labels(i)
     trainingY = Y[:test_samples]
     testingY = Y[test_samples:]
     return trainingY, testingY
	 
def SVM_age(x_tr, y_tr, x_te):
    
    clf = svm.SVC(C=2, degree=2, gamma='scale', kernel='rbf')
    clf.fit(x_tr, y_tr)
    return np.ravel(clf.predict(x_te))

	 
def classify_age(test_samples):

    Xtrain, Xtest = divide_x(test_samples)
    Xtrain = Xtrain.reshape(Xtrain.shape[0],Xtrain.shape[1]*Xtrain.shape[2])
    Xtest = Xtest.reshape(Xtest.shape[0],Xtest.shape[1]*Xtest.shape[2])
    Ytrain, Ytest = get_labels(4,test_samples)
    Y_pred_te = SVM_age(Xtrain,Ytrain,Xtest)
    size_te = len(Y_pred_te)
    Y_pred_tr = SVM_age(Xtrain,Ytrain,Xtrain)
    size_tr = len(Y_pred_tr)
    test_accuracy = accuracy_score(Ytest, Y_pred_te)*100
    train_accuracy = accuracy_score(Ytrain, Y_pred_tr)*100
    print("Accuracy obtained on test data for age detection:")
    print(accuracy_score(Ytest, Y_pred_te)*100,'%')
    print("Accuracy obtained on train data for age detection:")
    print(accuracy_score(Ytrain, Y_pred_tr)*100,'%')
    print("The confusion matrix is:")
    print(confusion_matrix(Ytest, Y_pred_te))
    Y_pred = np.concatenate((Y_pred_tr, Y_pred_te),axis = 0)
    size_pred = len(Y_pred)
    np.savetxt("task_2labels.csv", Y_pred, delimiter=',')
    precision = (test_accuracy  * size_te) + (train_accuracy * size_tr)
    print(precision/size_pred)
    percentage = [precision / size_pred]
    np.savetxt("task_2precision.csv", percentage, delimiter=',')
    a = pd.read_csv("noise_classified.csv")
    b = pd.read_csv("task_2labels.csv")
    c = pd.read_csv("task_2precision.csv")
    merged = pd.concat([a,b],axis=1)
    merged.to_csv("task_2.csv", index=False)
    labelled = pd.read_csv("task_2.csv")
    final = pd.concat([labelled,c],axis=1)
    final.to_csv("task_2.csv", index=False)
	
classify_age(4000)