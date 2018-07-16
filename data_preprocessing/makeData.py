import json
import os
from scipy import misc
import dlib
import tqdm
import numpy as np
import cv2

import sys
sys.path.append('./..')
from utils import utils


shape_predictor_path = "../model/shape_predictor_68_face_landmarks.dat"
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(shape_predictor_path)

folder_in_training = '../data/yale_sample/train'
folder_in_testing = '../data/yale_sample/test'
folder_out_training = '../data/prepared_data/train'
if not os.path.exists(folder_out_training):
	os.makedirs(folder_out_training)
folder_out_testing = '../data/prepared_data/test'
if not os.path.exists(folder_out_testing):
	os.makedirs(folder_out_testing)
folder_out_features = '../data/prepared_data/features'
if not os.path.exists(folder_out_features):
	os.makedirs(folder_out_features)


def prepare_data(filepath, folder_out):
	img = misc.imread(filepath,mode='L')
	detectedFaces = detector(img, 1)
	largestArea = 0
	if len(detectedFaces) >= 1: # Consider only the most prominent face in the frame if multiple faces are detected
		for k, d in enumerate(detectedFaces):
			length = abs(d.right() - d.left())
			breadth = abs(d.bottom() - d.top())
			area = length * breadth
			if area > largestArea:
				faceIndex = k
				faceBB = d
				largestArea = area
		w = faceBB.right()-faceBB.left()
		h = faceBB.bottom()-faceBB.top()

		img_cropped = img[int(faceBB.top()-h*0.2):int(faceBB.bottom()+h*0.2),int(faceBB.left()-0.2*w):int(faceBB.right()+0.2*w)]
		outfilepath = os.path.join(folder_out,file+'.jpg')
		misc.imsave(outfilepath,img_cropped)

		img_features = img.copy()
		shape = predictor(img, faceBB)
		features = np.empty([68*2], dtype = "double")
		img_cropped_features = img_features[int(faceBB.top()-h*0.2):int(faceBB.bottom()+h*0.2),int(faceBB.left()-0.2*w):int(faceBB.right()+0.2*w)]
		for itr in range(68):
			features[itr] = int(shape.part(itr).x - (faceBB.left()-w*0.2))
			features[itr+68] = int(shape.part(itr).y - (faceBB.top()-h*0.2))

			cv2.circle(img_cropped_features, (int(features[itr]), int(features[itr+68])), 1, (255,0,0), -1)
		outfilepath_features = os.path.join(folder_out_features,file+'.jpg')
		misc.imsave(outfilepath_features,img_cropped_features)

	return outfilepath, features

data_training = []
for file in tqdm.tqdm(os.listdir(folder_in_training)):
	filepath = os.path.join(folder_in_training,file)

	outfilepath, features = prepare_data(filepath,folder_out_training)

	data_training.append({
		'imgpath': outfilepath,
		'features': features
	})

data_testing = []
for file in tqdm.tqdm(os.listdir(folder_in_testing)):
	filepath = os.path.join(folder_in_testing,file)

	outfilepath, features = prepare_data(filepath,folder_out_testing)

	data_testing.append({
		'imgpath': outfilepath,
		'features': features
	})

with open(os.path.join(folder_out_training,'../data_training.json'),'w') as f:
	json.dump(data_training,f,indent=4,cls=utils.customJsonEncoder)

with open(os.path.join(folder_out_testing,'../data_testing.json'),'w') as f:
	json.dump(data_testing,f,indent=4,cls=utils.customJsonEncoder)

