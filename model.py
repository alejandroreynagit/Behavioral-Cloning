# coding: utf-8

# Load libraries
import os
import glob
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
import csv
import cv2
import numpy as np
from keras.models import Sequential
from keras.layers.core import Dense, Lambda, Activation, Flatten, Dropout
from keras.layers import Cropping2D, Conv2D, MaxPooling2D


# ## LOAD DATA SET

data_path = './data'
data_folders = glob.glob(os.path.join(data_path, '*'))
samples_sim = []
for folder in range(len(data_folders)):
    with open(data_folders[folder]+'/driving_log.csv') as csvfile:
        reader = csv.reader(csvfile)
        for line in reader:
            samples_sim.append(line)
print("Done Loading Data") 

images = []
measurements = []
for sample_data in samples_sim:    
    source_path = sample_data # Store the center image only [1] is left, [2] is right
    filename = (source_path[0]).split('/')[-1]
    img = cv2.imread(source_path[0])[:,:,::-1]
    images.append(img)
    measurement = float(sample_data[3]) # Steering angle, predicted variable
    measurements.append(measurement)
print("Done Merging Data") 

# Assign images and measurements to variables
X_train = np.array(images)
y_train = np.array(measurements)

# Number of training examples
n_train = len(X_train)

# What's the shape of an traffic sign image?
image_shape = X_train[0].shape

# How many unique classes/labels there are in the dataset.
n_classes = len(np.unique(y_train))

print("Number of training examples =", n_train)
print("Image data shape =", image_shape)
print("Number of classes =", n_classes)


# ### Augmenting Image Collection
collection_images = []
collection_measurements = []
for image, measurement in zip(images, measurements):
    collection_images.append(image)
    collection_measurements.append(measurement)
    collection_images.append(cv2.flip(image,1))
    collection_measurements.append(measurement*-1.0)
    
X_train = np.array(collection_images)
y_train = np.array(collection_measurements)

# Number of training examples
n_train = len(X_train)

# What's the shape of a simulator image?
image_shape = X_train[0].shape

# How many unique classes/labels there are in the dataset.
n_classes = len(np.unique(y_train))

print("NEW Number of training examples =", n_train)
print("Image data shape =", image_shape)
print("Number of classes =", n_classes)

# ### Training Model
# 
# Normalisation and Zero mean centering

n_out = 1 # Number of classes to predict. This case: steering angle

# Create the Sequential model
model = Sequential()
print(image_shape)
model.add(Cropping2D(cropping=((70,20), (0,0)), input_shape=image_shape))
print(model.output_shape)
# Pre-process the Data Set. Normalisation and Zero mean centering. 
# The -.5 will shift the mean (0.5) to zero!
model.add(Lambda(lambda x: (x / 255.0) - 0.5, input_shape=image_shape))
print(model.output_shape)

#Layer 1
#Conv Layer 1
model.add(Conv2D(filters = 8, kernel_size = 5, strides = 1, 
                 activation = 'relu'))
print(model.output_shape)

#Pooling layer 1
model.add(MaxPooling2D(pool_size = 2, strides = 2))
print(model.output_shape)
#Layer 2
#Conv Layer 2
model.add(Conv2D(filters = 16, kernel_size = 5,strides = 1,
                 activation = 'relu'))
#Pooling Layer 2
model.add(MaxPooling2D(pool_size = 2, strides = 2))
#Flatten
model.add(Flatten())
print(model.output_shape)

#Layer 3
#Fully connected layer 1
model.add(Dense(units = 120, activation = 'relu'))
print(model.output_shape)

'''#Adding a dropout layer to avoid overfitting. Here we are have given the dropout rate as 25% after first fully connected layer
model.add(Dropout(0.25))'''

#Layer 4
#Fully connected layer 2
model.add(Dense(units = 84, activation = 'relu'))
print(model.output_shape)

#Layer 5
#Output Layer , activation = 'softmax'
model.add(Dense(units = n_out))
print(model.output_shape)

#Compile and Fit
model.compile(loss='mean_squared_error', optimizer='adam',metrics=['accuracy'])
model.fit(X_train,y_train,validation_split=0.2,shuffle=True,nb_epoch=5)
print('Finished training')

model.save('models/model_LeNet.h5')
print('Model Saved')