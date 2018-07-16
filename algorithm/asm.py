import numpy as np
import json
import cv2
from scipy import misc
import math
import os

import sys
sys.path.append('./..')
from utils import utils

class ASM():

	def __init__(self):
		self.num_features = 68
		self.bConvergenceThresh = 0.001
		self.k = 10
		self.m = 30 #m>2k
		self.maxIterations = 20
		self.FEATURE_TO_USE = 'normal_profile' #options are norml_profile and gradient

		with open('./../data/features_neighbours_index.json','r') as f:
			self.features_neighbours_index = json.load(f)

		model_path = './../model/model.json'
		if os.path.exists(model_path):
			with open(model_path,'r') as f:
				modelJson = json.load(f)

				self.b = utils.decodeComplex(modelJson['b'])
				self.eigen_values = modelJson['eigen_values']
				self.P = utils.decodeComplex(modelJson['P'])
				self.Sg_arr = np.array(modelJson['Sg_arr'])

				self.g_arr = np.array(modelJson['g_arr'])
				self.X_mean = modelJson['X_mean']


	#Args:
	#X: matrix of shape (num_examples,num_dimensions)
	#Returns:
	#X_cov: covariance of X
	def covariance(self, X):
		X_mean = np.mean(X,axis=0)
		X_dev = X-X_mean
		X_cov = np.matmul(np.transpose(X_dev),X_dev)
		return X_cov

	#Args:
	#X: a covariance matrix
	#Returns:
	#ev_sorted: the sorted eigen values (descending order)
	#eig_sorted: the sorted eigen vectors
	def PCA(self, X):
		ev, eig = np.linalg.eig(X)
		ev_sorted_indices = np.argsort(ev)[::-1]
		ev_sorted = ev[ev_sorted_indices]
		eig_sorted = eig[:,ev_sorted_indices]
		return ev_sorted, eig_sorted


	def norm(self, X):
		return np.sum([x**2 for x in X])

	#Args:
	#X1: the image shape
	#X2: the model shape
	#Returns:
	#scale: the scale of transformation
	#rotation: the rotation transformation
	def alignTwoShapes(self, X1, X2):
		X1_norm = self.norm(X1)
		a = np.dot(X1,X2)/X1_norm
		if isinstance(a, complex):
			assert a.imag==0, "imaginary part should be zero"
			a = a.real
		half_len = len(X1)//2
		b = np.sum([X1[i]*X2[i+half_len]-X1[i+half_len]*X2[i] for i in range(half_len)])/X1_norm
		if isinstance(b, complex):
			assert b.imag==0, "imaginary part should be zero"
			b = b.real
		scale = np.sqrt(a**2+b**2)

		rotation = np.arctan2(b,a)

		return scale, rotation

	#Args:
	#X: matrix of shape (num_examples,dim)
	#Returns:
	#mean along the columns
	def mean(self, X):
		return np.mean(X,axis=0)

	#Args:
	#gs: mean of gradients; array of shape (dim)
	#g: gradients along the profile; shape (dim)
	#Sg: covariance matrix of g in the training examples; shape (dim,dim)
	#Returns:
	#the mahalanobis distance
	def mahalanobisDist(self, gs, g, Sg):
		#TODO: store the inverse?
		return np.matmul(np.matmul((gs-g),np.linalg.pinv(Sg)),np.transpose(gs-g))

	def getNormalProfile(self, point_1, point_2, point_3):
		#TODO: handle complex numbers
		x1,y1 = point_1[0], point_1[1]
		x2,y2 = point_2[0], point_2[1]
		x3,y3 = point_3[0], point_3[1]

		if isinstance(y2-y1,complex):
			assert (y2-y1).imag==0, "imaginary part should be zero"
			ty1 = (y2-y1).real
		else:
			ty1 = y2-y1

		if isinstance(x2-x1,complex):
			assert (x2-x1).imag==0, "imaginary part should be zero"
			tx1 = (x2-x1).real
		else:
			tx1 = x2-x1

		if isinstance(y3-y2,complex):
			assert (y3-y2).imag==0, "imaginary part should be zero"
			ty2 = (y3-y2).real
		else:
			ty2 = y3-y2

		if isinstance(x3-x2,complex):
			assert (x3-x2).imag==0, "imaginary part should be zero"
			tx2 = (x3-x2).real
		else:
			tx2 = x3-x2

		theta_1 = np.arctan2(ty1,tx1)
		theta_2 = np.arctan2(ty2,tx2)
		theta_1_normal = theta_1-np.pi/2
		theta_2_normal = theta_2-np.pi/2

		theta_profile = (theta_1+theta_2)/2

		return theta_profile

	def getCurPos(self, cur_row, cur_col, k, num_rows, num_cols):
		if cur_row<k+2:
			cur_row = k+2
		if cur_row>num_rows-k-2:
			cur_row = num_rows-k-2
		if cur_col<k+2:
			cur_col = k+2
		if cur_col>num_cols-k-2:
			cur_col = num_cols-k-2

		return cur_row, cur_col

	def getProfilePoints(self, profile_point, profile_theta, k, img):
		points_above = []
		points_below = []
		intensity_above = []
		intensity_below = []

		num_rows = img.shape[0]
		num_cols = img.shape[1]

		point_x = profile_point[0]
		point_y = profile_point[1]
		if isinstance(point_x,complex):
			assert point_x.imag==0, 'Imaginary part must be zero'
			point_x = point_x.real
		if isinstance(point_y,complex):
			assert point_y.imag==0, 'Imaginary part must be zero'
			point_y = point_y.real

		if profile_theta<0:
			profile_theta = profile_theta + np.pi


		#NOTE: x is along cols and y is along rows
		#Assume (0,0) is at the top left
		#TODO: handle case when profile goes out of the image
		count = 0
		if profile_theta<np.pi/4 or profile_theta>3*np.pi/4:
			if np.isinf(1/np.tan(profile_theta)):
				x_per_y = 1000000000000000000
			else:
				x_per_y = np.ceil(1/np.tan(profile_theta))

			#sample in positive direction of profile
			cur_row = int(point_y)
			cur_col = int(point_x)
			cur_row, cur_col = self.getCurPos(cur_row,cur_col,k,num_rows,num_cols)

			while count<k+1 and cur_row>=0 and cur_row<num_rows and cur_col>=0 and cur_col<num_cols:
				cur_point_intensity = img[cur_row][cur_col]
				cur_point = (cur_col,cur_row)

				if k%x_per_y==0:
					cur_row -= 1
				if profile_theta<np.pi/2:
					cur_col += 1
				elif profile_theta>np.pi/2:
					cur_col -= 1
				
				count += 1
				points_above.append(cur_point)
				intensity_above.append(cur_point_intensity)

			cur_row = int(point_y)
			cur_col = int(point_x)
			cur_row, cur_col = self.getCurPos(cur_row,cur_col,k,num_rows,num_cols)
			while count<2*k+2 and cur_row>=0 and cur_row<num_rows and cur_col>=0 and cur_col<num_cols:
				cur_point_intensity = img[cur_row][cur_col]
				cur_point = (cur_col,cur_row)
				if k%x_per_y==0:
					cur_row += 1
				if profile_theta<np.pi/2:
					cur_col -= 1
				elif profile_theta>np.pi/2:
					cur_col += 1

				count += 1
				points_below.append(cur_point)
				intensity_below.append(cur_point_intensity)

		elif profile_theta>=np.pi/4 or profile_theta<=3*np.pi/4:
			if np.isinf(np.tan(profile_theta)):
				y_per_x = 1000000000000000000
			else:
				y_per_x = np.ceil(np.tan(profile_theta))


			#sample in positive direction of profile
			cur_row = int(point_y)
			cur_col = int(point_x)
			cur_row, cur_col = self.getCurPos(cur_row,cur_col,k,num_rows,num_cols)

			while count<k+1 and cur_row>=0 and cur_row<num_rows and cur_col>=0 and cur_col<num_cols:
				cur_point_intensity = img[cur_row][cur_col]
				cur_point = (cur_col,cur_row)
				if k%y_per_x==0:
					if profile_theta<np.pi/2:
						cur_col += 1
					elif profile_theta>np.pi/2:
						cur_col -= 1
				cur_row -= 1
				
				count += 1
				points_above.append(cur_point)
				intensity_above.append(cur_point_intensity)

			cur_row = int(point_y)
			cur_col = int(point_x)
			cur_row, cur_col = self.getCurPos(cur_row,cur_col,k,num_rows,num_cols)
			while count<2*k+2 and cur_row>=0 and cur_row<num_rows and cur_col>=0 and cur_col<num_cols:
				cur_point_intensity = img[cur_row][cur_col]
				cur_point = (cur_col,cur_row)
				if k%y_per_x==0:
					if profile_theta<np.pi/2:
						cur_col -= 1
					elif profile_theta>np.pi/2:
						cur_col += 1
				cur_row += 1

				count += 1
				points_below.append(cur_point)
				intensity_below.append(cur_point_intensity)


		intensity = np.concatenate((intensity_below[::-1][:-1],intensity_above))
		points = np.concatenate((points_below[::-1][:-1],points_above))

		return intensity, points

	def gradientsAlongProfile(self, intensity_along_profile):
		return [intensity_along_profile[i+1]-intensity_along_profile[i] for i in range(len(intensity_along_profile)-1)]


	def getX(self, X_mean, P, b):
		X = X_mean + np.matmul(P,b)*1
		return X 

	def getBestPointAlongProfile(self, img, feature_point, profile_theta, m, k, feature_point_index):
		m_profile_intensity, m_profile_points = self.getProfilePoints(feature_point, profile_theta, m, img)

		best_profile_index = None
		min_dist = None
		for i in range(m-2*k-2):
			k_profile_intensity = m_profile_intensity[i:i+2*k+1]
			gs = self.gradientsAlongProfile(k_profile_intensity)

			dist = self.mahalanobisDist(gs,self.g_arr[feature_point_index],self.Sg_arr[feature_point_index])
			
			if min_dist is None:
				min_dist = dist
				best_profile_index = i
			else:
				if dist<min_dist:
					min_dist = dist
					best_profile_index = i

		
		if min_dist is not None and min_dist<100:
			best_feature_point = m_profile_points[best_profile_index+k]
		else:
			best_feature_point = feature_point
		return best_feature_point

	#Args:
	#X: matrix of shape (num_examples,2)
	#Returns:
	#centroid: (xc,yc)
	def findCentroid(self, X):
		x_arr = [X[i] for i in range(len(X)//2)]
		y_arr = [X[i] for i in range(len(X)//2,len(X))]

		centroid_x = np.sum(x_arr,axis=0)/len(x_arr)
		centroid_y = np.sum(y_arr,axis=0)/len(y_arr)

		return [centroid_x, centroid_y]

	#finds translation for going from model to image
	def findTranslation(self, model_shape, image_shape):
		centroid_model = self.findCentroid(model_shape)
		centroid_image = self.findCentroid(image_shape)

		translation = np.subtract(centroid_image,centroid_model)

		return translation[0], translation[1]

	def getGradient(self, img, row, col):
		padded_img = np.zeros((img.shape[0]+2,img.shape[1]+2))
		padded_img[1:-1,1:-1] = img
		row += 1
		col += 1


		img_mat = img[int(row-1):int(row+2)][:,int(col-1):int(col+2)]

		sobel_x = np.array([[1,0,-1],[2,0,-2],[1,0,-1]])
		sobel_y = np.array([[1,2,1],[0,0,0],[-1,-2,-1]])

		x_derivative = np.sum(np.multiply(sobel_x,img_mat))
		y_derivative = np.sum(np.multiply(sobel_y,img_mat))

		return x_derivative, y_derivative

	def gradientsAroundPoint(self, point, img, k):
		x = point[0]
		y = point[1]
		g = np.zeros((2*(2*k+1)**2))
		for row_offset in range(2*k+1):
			for col_offset in range(2*k+1):
				cur_row = y-k+row_offset
				cur_col = x-k+col_offset

				xd, yd = self.getGradient(img,cur_row,cur_col)

				g[row_offset*(2*k+1)+col_offset] = xd
				g[row_offset*(2*k+1)+col_offset+(2*k+1)] = yd

		return g

	def getBestPointInRegion(self, img, point, m, k, feature_index):
		x = point[0]
		y = point[1]

		min_dist = None
		best_point = None
		for row_offset in range(2*m+1):
			for col_offset in range(2*m+1):
				cur_row = y-k+row_offset
				cur_col = x-k+col_offset

				g = self.gradientsAroundPoint((cur_col,cur_row),img,k)
				dist = self.mahalanobisDist(g,self.g_arr[feature_index],self.Sg_arr[feature_index])

				if min_dist==None:
					min_dist = dist
					best_point = (cur_col,cur_row)

		return best_point

	def estimateFeaturePoints(self, img, X_mean, P, b, X):
		points = [(X[i],X[i+len(X)//2]) for i in range(len(X)//2)]
		new_points = []
		for index,point in enumerate(points):
			p1 = points[self.features_neighbours_index[index][0]]
			p2 = point
			p3 = points[self.features_neighbours_index[index][1]]
			if self.FEATURE_TO_USE=='gradient':
				best_point = self.getBestPointInRegion(img,point,self.m,self.k,index)
			elif self.FEATURE_TO_USE=='normal_profile':
				profile_theta = self.getNormalProfile(p1,p2,p3)
				best_point = self.getBestPointAlongProfile(img,point,profile_theta,self.m,self.k,index)
			new_points.append(best_point)
		new_features_x = [new_points[i][0] for i in range(len(new_points))]
		new_features_y = [new_points[i][1] for i in range(len(new_points))]
		new_features = np.concatenate((new_features_x,new_features_y))

		return new_features

	def constrainB(self, b, b_prime, eigen_values):
		new_b = b_prime.copy()
		constr = 0.001*np.sqrt(eigen_values)

		diff = new_b-b

		for index, el in enumerate(diff):
			if abs(el)>constr[index]:
				new_b[index] = b[index] + np.sign(diff[index])*constr[index]
		return new_b

	def asm(self, img):
		img = img.copy()
		b_prev = np.ones(self.num_features*2)
		X_mean = self.X_mean
		P = self.P
		# b = self.b
		b = np.zeros(self.num_features*2)

		features = self.getX(X_mean,P,b)
		# centroid = self.findCentroid(X_mean)
		# features_x = [centroid[0] for _ in range(self.num_features)]
		# features_y = [centroid[1] for _ in range(self.num_features)]
		# features = np.concatenate((features_x,features_y))


		count = 0
		while self.bNotConverged(b,b_prev) and count<self.maxIterations:
			b_prev = b
			features = self.estimateFeaturePoints(img,X_mean,P,b,features)

			Xt, Yt, s, theta, new_b = self.updateModelParams(X_mean,P,b,features)
			b = self.constrainB(b,new_b,self.eigen_values)
			count += 1

		return self.getX(X_mean,P,b)
		# return features

	def loadData(self):
		with open('../data/prepared_data/data_training.json','r') as f:
			data = json.load(f)

		features_arr = []
		imgs_arr = []
		for obj in data:
			features_arr.append(obj['features'])
			#TODO: don't store entire images in memory
			imgs_arr.append(misc.imread(obj['imgpath'],mode='L'))

		X_mean = self.mean(features_arr)
		cov_mat = self.covariance(features_arr)
		eigen_values, P = self.PCA(cov_mat)

		return X_mean, P, abs(eigen_values), features_arr, imgs_arr
		

	def bNotConverged(self, b, b_prev):
		diff = np.sum(np.abs(b-b_prev))/len(b)
		# print('----------------------diff--------------------', diff)
		return diff>self.bConvergenceThresh

	def translateShape(self, shape_to_translate, Xt, Yt):
		Xt_prime = [Xt for _ in range(len(shape_to_translate)//2)]
		Yt_prime = [Yt for _ in range(len(shape_to_translate)//2)]

		translation_vector = np.concatenate((Xt_prime,Yt_prime))

		return np.add(shape_to_translate,translation_vector)

	def scaleShape(self, shape, s, centroid):
		centroid_prime_x = np.array([centroid[0] for _ in range(len(shape)//2)])
		centroid_prime_y = np.array([centroid[1] for _ in range(len(shape)//2)])
		centroid_prime = np.concatenate((centroid_prime_x,centroid_prime_y))
		shape_prime = shape-centroid_prime
		shape_scaled = shape_prime*s+centroid_prime 
		return shape_scaled

	def rotate(self, features, theta, centroid):
		points = [(features[i],features[i+len(features)//2]) for i in range(len(features)//2)]
		points = np.array(points)

		points = points - centroid
		rotated_points = np.matmul([[np.cos(theta),-1*np.sin(theta)],[np.sin(theta),np.cos(theta)]], np.transpose(points))
		points = np.transpose(rotated_points) + centroid

		rotated_features_x = [point[0] for point in points]
		rotated_features_y = [point[1] for point in points]

		rotated_features = np.concatenate((rotated_features_x,rotated_features_y))
		return rotated_features

	def inverseTransform(self, features, Xt, Yt, s, theta, X):
		Y = self.translateShape(features,-1*Xt,-1*Yt)
		# Y = features
		centroid = self.findCentroid(X)
		# centroid = self.findCentroid(Y)
		Y = self.scaleShape(Y,1/s,centroid)
		Y = self.rotate(Y,-1*theta,centroid)

		return Y

	def findB(self, P, y, x):
		return np.matmul(np.transpose(P),y-x)

	def updateModelParams(self, X_mean, P, b, cur_features):
		X = self.getX(self.X_mean,self.P,self.b)

		Xt, Yt = self.findTranslation(X_mean,cur_features)
		translated_mean = self.translateShape(self.X_mean,Xt,Yt)
		s, theta = self.alignTwoShapes(translated_mean,cur_features)

		cur_features_transformed = self.inverseTransform(cur_features,Xt,Yt,s,theta,X)
		cur_features_norm = cur_features_transformed/np.dot(cur_features_transformed,X_mean)

		b = self.findB(P,cur_features_norm,X_mean)

		return Xt, Yt, s, theta, b

	def train(self):
		self.b = np.zeros(self.num_features*2)
		self.X_mean, self.P, self.eigen_values, features, imgs = self.loadData()

		#find b
		b_prev = np.ones(self.num_features*2)
		cur_img_index = 0
		while self.bNotConverged(self.b,b_prev):
			b_prev = self.b

			Xt, Yt, s, theta, new_b = self.updateModelParams(self.X_mean,self.P,self.b, features[cur_img_index])
			self.b = self.constrainB(self.b,new_b,self.eigen_values)

			cur_img_index = (cur_img_index+1)%len(imgs)

		#find profile stats
		g_features = []
		for i in range(len(imgs)):
			cur_img = imgs[i]
			cur_features = features[i]
			cur_g = []
			for j in range(len(cur_features)//2):
				profile_point = (cur_features[j],cur_features[j+len(cur_features)//2])

				if self.FEATURE_TO_USE=='normal_profile':
					p1 = (cur_features[self.features_neighbours_index[j][0]],\
						cur_features[self.features_neighbours_index[j][0]+len(cur_features)//2])
					p2 = profile_point
					p3 = (cur_features[self.features_neighbours_index[j][1]],\
						cur_features[self.features_neighbours_index[j][1]+len(cur_features)//2])
					profile_theta = self.getNormalProfile(p1,p2,p3)
					profile_intensity, profile_points = self.getProfilePoints(profile_point, profile_theta, self.k, cur_img)

					assert profile_intensity.shape[0]==2*self.k+1, 'The length of profile points must be {}, but is {}'.format(2*self.k+1,profile_intensity.shape[0])
					
					g = self.gradientsAlongProfile(profile_intensity)

					cur_g.append(g)
				elif self.FEATURE_TO_USE=='gradient':
					cur_g.append(self.gradientsAroundPoint(profile_point,cur_img,self.k))

			g_features.append(cur_g)

		g_features = np.array(g_features)

		self.g_arr = []
		self.Sg_arr = []

		for feature_num in range(g_features.shape[1]):
			cur_feature_stats = g_features[:,feature_num,:]

			g = self.mean(cur_feature_stats)
			Sg = self.covariance(cur_feature_stats)

			self.g_arr.append(g)
			self.Sg_arr.append(Sg)

		model = {
			'b': self.b,
			'P': self.P,
			'eigen_values': self.eigen_values,
			'X_mean': self.X_mean,
			'g_arr': self.g_arr,
			'Sg_arr': self.Sg_arr
		}

		with open('../model/model.json','w') as f:
			json.dump(model,f,indent=4,cls=utils.customJsonEncoder)

