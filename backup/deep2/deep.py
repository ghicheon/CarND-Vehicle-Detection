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

def deep_init(X,y):
    length_ = len(X[0])
    rand_state = 99
        
    global deep_model

    save_ok = False
    #save_ok = True

    if save_ok == True:
        deep_model = load_model('deep_model.h5')
    else:
        X_train, X_test, y_train, y_test = train_test_split( X, y, test_size=0.2, random_state=rand_state)

        X_test,X_valid, y_test, y_valid = train_test_split( X_test, y_test, test_size=0.1, random_state=rand_state)

        print(type(X_test[0]))
        print(X_test[0].shape)
            
        ####################################################################
        ####################################################################
        print(X_train.shape)

        batch_size = 128
        num_classes = 2
        epochs =50

        # input image dimensions
        img_rows, img_cols = 64,64


        X_train = X_train.astype('float32')
        X_valid = X_valid.astype('float32')
        X_test = X_test.astype('float32')

        # convert class vectors to binary class matrices
        y_train = keras.utils.to_categorical(y_train, num_classes)
        y_valid = keras.utils.to_categorical(y_valid, num_classes)
        y_test = keras.utils.to_categorical(y_test, num_classes)

        deep_model = Sequential()
        deep_model.add(Dense(32, activation='relu',input_dim = length_ ))
        deep_model.add(Dropout(0.2))
        deep_model.add(Dense(8, activation='relu'))
        deep_model.add(Dropout(0.2))
        deep_model.add(Dense(num_classes, activation='softmax'))

        deep_model.compile(loss=keras.losses.categorical_crossentropy,
                  optimizer=keras.optimizers.Adadelta(),
                  metrics=['accuracy'])

        #print("TTTTTTTTT", X_train[0])

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
    out = deep_model.predict(X)
    return np.argmax(out)

def test__():
    out = deep_model.predict( X_test[0:10])
    for i in range(5):
        print( np.argmax(out[i]))

    for i in range(5):
        print( np.argmax(y_test[i]))
