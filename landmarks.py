import os
import numpy as np
from keras.preprocessing import image
import cv2
import dlib
import csv

# PATH TO ALL IMAGES
global basedir, image_paths, target_size

basedir = 'assignmentdataset'
images_dir = os.path.join(basedir,'dataset2')
labels_filename = 'attribute_list.csv'
#change to wherever the dataset is found

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')


# how to find frontal human faces in an image using 68 landmarks.  These are points on the face such as the corners of the mouth, along the eyebrows, on the eyes, and so forth.

# The face detector we use is made using the classic Histogram of Oriented
# Gradients (HOG) feature combined with a linear classifier, an image pyramid,
# and sliding window detection scheme.  The pose estimator was created by
# using dlib's implementation of the paper:
# One Millisecond Face Alignment with an Ensemble of Regression Trees by
# Vahid Kazemi and Josephine Sullivan, CVPR 2014
# and was trained on the iBUG 300-W face landmark dataset (see https://ibug.doc.ic.ac.uk/resources/facial-point-annotations/):
#     C. Sagonas, E. Antonakos, G, Tzimiropoulos, S. Zafeiriou, M. Pantic.
#     300 faces In-the-wild challenge: Database and results.
#     Image and Vision Computing (IMAVIS), Special Issue on Facial Landmark Localisation "In-The-Wild". 2016.


def shape_to_np(shape, dtype="int"):
    # initialize the list of (x, y)-coordinates
    coords = np.zeros((shape.num_parts, 2), dtype=dtype)

    # loop over all facial landmarks and convert them
    # to a 2-tuple of (x, y)-coordinates
    for i in range(0, shape.num_parts):
        coords[i] = (shape.part(i).x, shape.part(i).y)

    # return the list of (x, y)-coordinates
    return coords

def rect_to_bb(rect):
    # take a bounding predicted by dlib and convert it
    # to the format (x, y, w, h) as we would normally do
    # with OpenCV
    x = rect.left()
    y = rect.top()
    w = rect.right() - x
    h = rect.bottom() - y

    # return a tuple of (x, y, w, h)
    return (x, y, w, h)


def run_dlib_shape(image):
    # in this function we load the image, detect the landmarks of the face, and then return the image and the landmarks
    # load the input image, resize it, and convert it to grayscale
    resized_image = image.astype('uint8')

    gray = cv2.cvtColor(resized_image, cv2.COLOR_BGR2GRAY)
    gray = gray.astype('uint8')

    # detect faces in the grayscale image
    rects = detector(gray, 1)
    num_faces = len(rects)

    if num_faces == 0:
        return None, resized_image

    face_areas = np.zeros((1, num_faces))
    face_shapes = np.zeros((136, num_faces), dtype=np.int64)

    # loop over the face detections
    for (i, rect) in enumerate(rects):
        # determine the facial landmarks for the face region, then
        # convert the facial landmark (x, y)-coordinates to a NumPy
        # array
        temp_shape = predictor(gray, rect)
        temp_shape = shape_to_np(temp_shape)

        # convert dlib's rectangle to a OpenCV-style bounding box
        # [i.e., (x, y, w, h)],
        #   (x, y, w, h) = face_utils.rect_to_bb(rect)
        (x, y, w, h) = rect_to_bb(rect)
        face_shapes[:, i] = np.reshape(temp_shape, [136])
        face_areas[0, i] = w * h
    # find largest face and keep
    dlibout = np.reshape(np.transpose(face_shapes[:, np.argmax(face_areas)]), [68, 2])
    return dlibout, resized_image

image_paths = [os.path.join(images_dir, l) for l in os.listdir(images_dir)]
target_size = None
labels_file = open(os.path.join(basedir, labels_filename), 'r')
lines = labels_file.readlines()
flag=np.zeros((1,5000))


def extract_features_labels():

    all_features = []
    faces = []
    #initialises faces vector
    if os.path.isdir(images_dir):
        n = 1
        for img_path in image_paths:
            file_name= img_path.split('\\')[2].split('.')[0]
            #loads the image
            img = image.img_to_array(
                image.load_img(img_path,
                               target_size=target_size,
                               interpolation='bicubic'))
            features, _ = run_dlib_shape(img)
            #searches for a face in the image
            if features is not None:
                all_features.append(features)
                faces.append(int(file_name))
            #adds file to list if face is found
            else:
                flag[0, int(file_name)-1] = 1
            n = n + 1

    facesnp = np.array(faces) #conversion to np array
    np.savetxt("noise_classified.csv", facesnp, delimiter = ',') #writes a csv with all the file numbers of detected faces
    landmark_features = np.array(all_features)
    return landmark_features


def extract_labels(k):

    all_labels = []
    label = {line.split(',')[0] : int(line.split(',')[k]) for line in lines[2:]} #Selects column from which data is read
    if os.path.isdir(images_dir):
        n = 0
        for img_path in image_paths:
            file_name= img_path.split('\\') [2].split('.')[0]
            if(flag[0,int(file_name)-1]==0):
                all_labels.append(label[file_name])
            n = n + 1
    label_no = (np.array(all_labels) + 1)/2 #Eliminates -1 values
    return label_no  #Returns label number
