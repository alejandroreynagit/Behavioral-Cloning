from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
import csv
import cv2
import numpy as np
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers.core import Dense, Activation, Flatten, Dropout

samples_sim = []
with open('data/driving_log.csv') as csvfile:
    reader = csv.reader(csvfile)
    for line in reader:
        samples_sim.append(line)
print("Done Loading Data")    

# Split the data. 20% for validation 80% training
train_samples, validation_samples = train_test_split(samples_sim,test_size=0.20)

images = []
measurements = []

for sample_data in train_samples:
    source_path = sample_data[0] # Store the center image only [1] is left, [2] is right
    filename = str(sample_data).split('/')[-1]
    img_path = 'data/IMG/' + filename
    img = cv2.imread(img_path)
    images.append(img)
    measurement = float(sample_data[3]) # Steering angle, predicted variable
    measurements.append(measurement)

X_train = np.array(images)
y_train = np.array(measurements)

# TRAINING ...
# Create the Sequential model
model = Sequential()
#1st Layer - Add a flatten layer
model.add(Flatten(input_shape=(160, 320, 3)))
model.add(Dense(1))

model.compile(loss='mse', optimizer='adam')
model.fit(X_train,y_train,validation_split=0.2,shuffle=True,nb_epoch=5)
model.save('models/model_basic.h5')
print('finished training')
