import os
import numpy as np
from keras.preprocessing import image
import cv2
import dlib

datasetdir = 'assignmentdataset/dataset2'
facedetector = dlib.get_frontal_face_detector()
facesfound = []

for i in range (1,5001):
    imgpath = (datasetdir +'/'+ str(i) + '.png')
    img = cv2.imread(imgpath)

    gray = img.astype('uint8')
    gray = cv2.cvtColor(gray, cv2.COLOR_BGR2GRAY)
    gray = gray.astype('uint8')
    rects = facedetector(gray, 1)
    if len(rects) == 0:
        facesfound.append([i,0])
    else:
        facesfound.append([i,1])
    npfaces = np.array(facesfound)
    np.savetxt("noise_classified.csv", npfaces, delimiter = ',')