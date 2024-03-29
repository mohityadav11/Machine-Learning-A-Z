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

# Adding a Second Convolutional Layer
classifier.add(Convolution2D(filters = 32 , kernel_size = (3 ,3), activation = 'relu'))
classifier.add(MaxPooling2D(pool_size=(2,2)))

# Step - 3 Flattening
classifier.add(Flatten())

# Step - 4 Full Connection
classifier.add(Dense(units=128, activation='relu'))
classifier.add(Dense(units=1, activation='sigmoid'))

#Compiling the CNN
classifier.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Part - 2 Fitting the CNN to images
from keras.preprocessing.image import ImageDataGenerator
 
train_datagen = ImageDataGenerator(
        rescale=1./255,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True)

test_datagen = ImageDataGenerator(rescale=1./255)

training_set = train_datagen.flow_from_directory(
                                                'dataset/training_set',
                                                target_size=(64, 64),
                                                batch_size=32,
                                                class_mode='binary')

test_set = test_datagen.flow_from_directory(
                                            'dataset/test_set',
                                            target_size=(64, 64),
                                            batch_size=32,
                                            class_mode='binary')

#steps_per_epoch = steps_samples_per_epoch/batch_size which is 8000/32 and 2000/32

classifier.fit_generator(training_set,
                        steps_per_epoch=250,
                        epochs=25,
                        validation_data=test_set,
                        validation_steps=2000) 

