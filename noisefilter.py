import os
import numpy as np
from keras.preprocessing import image
import cv2
import dlib
# change path to wherever the dataset is found
# initializes vector of found faces
# imports face detector from dlib
datasetdir = 'assignmentdataset/dataset2'
facedetector = dlib.get_frontal_face_detector()
facesfound = []

for i in range (1,5001):
    imgpath = (datasetdir +'/'+ str(i) + '.png')
    img = cv2.imread(imgpath)

    gray = img.astype('uint8')
    gray = cv2.cvtColor(gray, cv2.COLOR_BGR2GRAY)
    gray = gray.astype('uint8')
	#creates a bounding box for the faces
    rects = facedetector(gray, 1)
    if len(rects) == 0:
        facesfound.append([i,0])
    else:
        facesfound.append([i,1])
	# adds a 1 for every rectangle detected, or a 0 if no face was found
    npfaces = np.array(facesfound)
    np.savetxt("noise_classified.csv", npfaces, delimiter = ',')
	#conversion to numpy array and writing the file