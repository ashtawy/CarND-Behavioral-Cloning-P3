import numpy as np
import pandas as pd
import cv2
from keras.models import Sequential
from keras.layers import Dense, Flatten, Lambda, Cropping2D
from keras.layers.convolutional import Convolution2D
from keras.layers.pooling import MaxPooling2D
import matplotlib
%matplotlib inline


# Read the driving log data
driving_log = pd.read_csv('data/driving_log.csv')

# loop through the Pandas dataframe and read the images whose paths
# are the rows of the dataframe.
images = []
measurements = []
correction = 0.2
for i in range(driving_log.shape[0]):
    center_image = cv2.imread(('data/'+driving_log['center'].iloc[i]).replace(' ', ''))
    left_image = cv2.imread(('data/'+driving_log['left'].iloc[i]).replace(' ', ''))
    right_image = cv2.imread(('data/'+driving_log['right'].iloc[i]).replace(' ', ''))
    center_steering_angle = driving_log['steering'].iloc[i]
    left_steering_angle = center_steering_angle + correction
    right_steering_angle = center_steering_angle - correction
    images.extend([center_image, np.fliplr(center_image.copy()), 
                  left_image,
                  right_image])
    measurements.extend([center_steering_angle, -center_steering_angle,
                        left_steering_angle,
                        right_steering_angle])

# Convert the lists to numpy arrays
X_train = np.array(images)
y_train = np.array(measurements)


# Print the number of images and their dimenstions & number of channels
print(X_train.shape)


# Construct the model
model = Sequential()
model.add(Lambda(lambda x: (x/255.0) - 0.5, input_shape=(160, 320, 3)))
model.add(Cropping2D(cropping=((70,25), (0,0))))
model.add(Convolution2D(6, 5, 5, activation='relu'))
model.add(MaxPooling2D())
model.add(Convolution2D(6, 5, 5, activation='relu'))
model.add(MaxPooling2D())
model.add(Flatten())
model.add(Dense(120))
model.add(Dense(84))
model.add(Dense(1))
# Compile, fit and save the model
model.compile(loss='mse', optimizer='adam')
model.fit(X_train, y_train, validation_split=0.2, shuffle=True, nb_epoch=10)
model.save('model.h5')