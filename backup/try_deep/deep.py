import pickle
import keras
import cv2
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras import backend as K

import glob
from sklearn.preprocessing import StandardScaler

import matplotlib.image as mpimg
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from keras.models import load_model


deep_model = None

def deep_init():
	global deep_model

	#save_ok = False
	save_ok = True

	if save_ok == True:
                deep_model = load_model('deep_model.h5')
	else:
		# Read in cars and notcars
		cars = glob.glob('data/vehicles/*/*.png')
		notcars = glob.glob('data/non-vehicles/*/*.png')

		car_features = []
		notcar_features = []

		for f in cars:
			#image = cv2.imread(f)
			image = mpimg.imread(f)
			car_features.append(image)

		for f in notcars:
			#image = cv2.imread(f)
			image = mpimg.imread(f)
			notcar_features.append(image)


		print("length of cars: ",len(cars))
		print("length of notcars: ",len(notcars))



		# Create an array stack of feature vectors
		X = np.vstack((car_features, notcar_features)).astype(np.float64)

		# Define the labels vector
		y = np.hstack((np.ones(len(car_features)), np.zeros(len(notcar_features))))

		# Split up data into randomized training and test sets
		rand_state = np.random.randint(0, 100)
		X_train, X_test, y_train, y_test = train_test_split( X, y, test_size=0.2, random_state=rand_state)

		X_test,X_valid, y_test, y_valid = train_test_split( X_test, y_test, test_size=0.1, random_state=rand_state)

		print(type(X_test[0]))
		print(X_test[0].shape)
		    
		# Fit a per-column scaler
		#X_scaler = StandardScaler().fit(X_train)
		# Apply the scaler to X
		#X_train = X_scaler.transform(X_train)
		#X_test = X_scaler.transform(X_test)


		####################################################################
		####################################################################

		"""
		>>> print(X_train.shape)
		(34799, 32, 32, 3)

		>>> print(X_train[0].shape)
		(32, 32, 3)

		>>> print(y_train.shape)
		(34799,)
		>>>
		"""

		print(X_train.shape)

		batch_size = 128
		num_classes = 2
		epochs =50

		# input image dimensions
		img_rows, img_cols = 64,64

		# the data, shuffled and split between train and test sets
		if K.image_data_format() == 'channels_first':
		    X_train = X_train.reshape(X_train.shape[0], 3, img_rows, img_cols)
		    X_valid = X_valid.reshape(X_valid.shape[0], 3, img_rows, img_cols)
		    X_test = X_test.reshape(X_test.shape[0], 3, img_rows, img_cols)
		    input_shape = (3, img_rows, img_cols)
		else:
		    X_train = X_train.reshape(X_train.shape[0], img_rows, img_cols, 3)
		    X_valid = X_valid.reshape(X_valid.shape[0], img_rows, img_cols, 3)
		    X_test = X_test.reshape(X_test.shape[0], img_rows, img_cols, 3)
		    input_shape = (img_rows, img_cols, 3)

		X_train = X_train.astype('float32')
		X_valid = X_valid.astype('float32')
		X_test = X_test.astype('float32')
		#
		#X_train /= 255
		#X_valid /= 255
		#X_test /= 255
		##
		print('X_train shape:', X_train.shape)
		print(X_test.shape[0], 'test samples')

		print("Training...11111111111 ", X_train[0,0,0,0])

		# convert class vectors to binary class matrices
		y_train = keras.utils.to_categorical(y_train, num_classes)
		y_valid = keras.utils.to_categorical(y_valid, num_classes)
		y_test = keras.utils.to_categorical(y_test, num_classes)

		deep_model = Sequential()
		deep_model.add(Conv2D(32, kernel_size=(3, 3),
				 activation='relu',
				 input_shape=input_shape))
		deep_model.add(Conv2D(64, (3, 3), activation='relu'))
		deep_model.add(MaxPooling2D(pool_size=(2, 2)))
		deep_model.add(Flatten())
		deep_model.add(Dense(32, activation='relu'))
		deep_model.add(Dropout(0.5))
		deep_model.add(Dense(21, activation='relu'))
		deep_model.add(Dropout(0.5))
		deep_model.add(Dense(8, activation='relu'))
		deep_model.add(Dropout(0.5))
		deep_model.add(Dense(num_classes, activation='softmax'))

		deep_model.compile(loss=keras.losses.categorical_crossentropy,
			      optimizer=keras.optimizers.Adadelta(),
			      metrics=['accuracy'])

		deep_model.fit(X_train, y_train,
			  batch_size=batch_size,
			  epochs=epochs,
			  verbose=1,
			  validation_data=(X_valid, y_valid))
		score = deep_model.evaluate(X_test, y_test, verbose=0)


		print('Test loss:', score[0])
		print(X_test[0:1].shape)
		deep_model.save('deep_model.h5') 
                
                



def deep_predict(X):
	global deep_model

	#print("DDDDDDDDDDDDDDDDD", X[0,0,0], " ", X[1,0,1])

	X = X/255
	k=[]
	k.append(X)   #add demension.
	#print("k.shape", np.array(k).shape)
	out = deep_model.predict(np.array(k))
	return np.argmax(out)

def test__():
	out = deep_model.predict( X_test[0:10])
	for i in range(5):
		print( np.argmax(out[i]))

	for i in range(5):
		print( np.argmax(y_test[i]))
