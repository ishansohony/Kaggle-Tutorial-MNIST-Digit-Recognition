#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 27 21:07:34 2017

@author: nodlehs
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 27 19:29:40 2017

@author: nodlehs
"""

import numpy 
import os
import pandas as pd
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import Flatten
from keras.layers.convolutional import Convolution2D
from keras.layers.convolutional import MaxPooling2D
from keras.utils import np_utils
from keras import backend as K
K.set_image_dim_ordering('th')

# fix random seed for reproducibility
seed = 7
numpy.random.seed(seed)

# load data
(X_train, y_train), (X_test, y_test) = mnist.load_data()
# reshape to be [samples][pixels][width][height]
X_train = X_train.reshape(X_train.shape[0], 1, 28, 28).astype('float32')
X_test = X_test.reshape(X_test.shape[0], 1, 28, 28).astype('float32')

# normalize inputs from 0-255 to 0-1
X_train = X_train / 255
X_test = X_test / 255
# one hot encode outputs
y_train = np_utils.to_categorical(y_train)
y_test = np_utils.to_categorical(y_test)
num_classes = y_test.shape[1]

X_train = numpy.concatenate((X_train,X_test),axis = 0)
y_train = numpy.concatenate((y_train,y_test),axis = 0)

print(numpy.shape(X_train))
print(numpy.shape(y_train))


def larger_model():
	# create model
        model = Sequential()
        model.add(Convolution2D(32, 3, 3, input_shape=(1, 28, 28), activation='relu'))
        model.add(Convolution2D(32, 3, 3, input_shape=(1, 28, 28), activation='relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Convolution2D(64, 3, 3, activation='relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Flatten())
        model.add(Dense(1024, activation='relu'))
        model.add(Dropout(0.5))
        model.add(Dense(num_classes, activation='softmax'))
	# Compile model
        model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
        return model
    
    
# build the model
model = larger_model()
# Fit the model
model.fit(X_train, y_train, validation_data=(X_test, y_test), nb_epoch=10, batch_size=200, verbose=2)
# Final evaluation of the model
scores = model.evaluate(X_test, y_test, verbose=0)
print("Baseline Error: %.2f%%" % (100-scores[1]*100))

test  = pd.read_csv("/home/nodlehs/Downloads/test.csv").values
test = test.reshape(test.shape[0], 1, 28, 28)
test = test.astype(float)
test /= 255.0

test_pred_prob = model.predict(test)

test_pred = numpy.argmax(test_pred_prob, axis=1)
print (test_pred)                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                   

if not os.path.exists('/home/nodlehs/results/'):
    os.makedirs('/home/nodlehs/results/')
numpy.savetxt('/home/nodlehs/results/mnist-predictions%s.csv' , numpy.c_[range(1, len(test_pred) + 1), test_pred], delimiter = ',', header = 'ImageId,Label', comments = '', fmt='%d')
print("Saved predictions to a CSV file.")

