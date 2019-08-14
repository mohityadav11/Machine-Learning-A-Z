#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug 14 17:41:12 2019

@author: mohityadav
"""
#Building CNN
from keras.models import Sequential
from keras.layers import Convolution2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense

#Initiliazing the CNN
classifier = Sequential()

#Step - 1 Convolution
classifier.add(Convolution2D(filters = 32 , kernel_size = (3 ,3), data_format='channels_last', input_shape=(64, 64, 3), activation = 'relu'))

# Step - 2 Pooling
classifier.add(MaxPooling2D(pool_size=(2,2)))

# Step - 3 Flattening
classifier.add(Flatten())

# Step - 4 Full Connection
classifier.add(Dense(units=128, activation='relu'))
classifier.add(Dense(units=1, activation='sigmoid'))

#Compiling the CNN
classifier.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

