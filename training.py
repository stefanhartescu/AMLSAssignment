
train_X = x_train.reshape(x_train.shape[0],x_train.shape[1]*x_train.shape[2])
test_X = x_test.reshape(x_test.shape[0],x_test.shape[1]*x_test.shape[2])
#reshape the data

def SVM(trainX, trainY, testX):
    from sklearn import svm
    clf = svm.SVC(gamma='scale')
    clf.fit(trainX, trainY)
    svm.SVC(C=1.0, cache_size=200, class_weight=None, coef0=0.0,
       decision_function_shape='ovr', degree=3, gamma='scale', kernel='linear',
        max_iter=-1, probability=False, random_state=None, shrinking=True,
        tol=0.001, verbose=False)
    print(clf.predict(testX))
    return np.ravel(clf.predict(testX))
	#trains an SVM with the obtained landmarks from the pictures

predict_Y = SVM(train_X,y_train,test_X)
size = len(predict_Y)

def precisioncalculator(resultdata,testdata,sizedata):
    np.ravel(testdata)
	# ravel used to save both vectors as line vectors
    counter = 0
    for x in range (0, sizedata - 1):
        if resultdata[x] == testdata[x]:
            counter = counter + 1
    return counter/sizedata

precision = precisioncalculator(predict_Y, y_test, size)*100
#calculates percentage of matches between predicted set and test set