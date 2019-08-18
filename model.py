import csv
from PIL import Image
import numpy as np
import re
import sklearn
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
import math


def flip_image(image):
    return (np.fliplr(image))    

#Calculates the measurement value for different camera angles
def measure(center, i):
    # when center 1, trig_part = 1
    # when center 0, trig_part = 0
    # when center -1, trig_part = -1
    trig_part = math.cos( (1-center) * math.pi/2 )
    trig_part = round(trig_part, 2)
    if(i==0):
        measurement = center
    elif(i==1): #Left measurement
        measurement = 0.05 + trig_part
        measurement = min(measurement, 1)
        measurement = max(measurement, -1)
    else: #Right measurement
        measurement = -0.05 + trig_part
        measurement = min(measurement, 1)
        measurement = max(measurement, -1)
    return measurement

#Reads the image at a given path, and converts the color space to RGB
def readAndConvert(path):
    image = Image.open(path)
    image_array = np.asarray(image)
    return image_array

#Flips image and negates measurement to augment the data
def augment(image, measurement):
    flipped_image = flip_image(image)
    flipped_measurement = -measurement
    image_list = [image, flipped_image]
    measurement_list = [measurement, flipped_measurement]
    return (image_list, measurement_list)

def generator(samples, batch_size=32):
    num_samples = len(samples)
    while 1: # Loop forever so the generator never terminates
        shuffle(samples)
        for offset in range(0, num_samples, batch_size):
            batch_samples = samples[offset:offset+batch_size]

            images = []
            angles = []
            for batch_sample in batch_samples:
                    center_measurement = float(batch_sample[3])
                    for i in range(3):
                        img_list = []
                        angle_list = []
                        measurement = measure(center_measurement, i)
                        
                        source_path = batch_sample[i]
                        filename = re.split('/|\\\\', source_path)[-1]
                        current_path = PATH_TO_IMAGE_FOLDER + filename
                        image = readAndConvert(current_path)
                        img_list, angle_list = augment(image, measurement)
                        images.extend(img_list)
                        angles.extend(angle_list)

            X_train = np.array(images)
            y_train = np.array(angles)
            yield sklearn.utils.shuffle(X_train, y_train)
        


if __name__ == '__main__':
    lines = []
    PATH_TO_CSV = 'my-driving-data/driving_log.csv'
    PATH_TO_IMAGE_FOLDER = 'my-driving-data/IMG/'
    with open(PATH_TO_CSV) as csvfile:
        reader = csv.reader(csvfile)
        for line in reader:
            lines.append(line)
    
    train_samples, validation_samples = train_test_split(lines, test_size=0.2)
    
    images=[]
    measurements = []
        
    batch_size = 64
    
    # compile and train the model using the generator function
    train_generator = generator(train_samples, batch_size=batch_size)
    validation_generator = generator(validation_samples, batch_size=batch_size)
    
    from keras.models import Sequential
    from keras.layers import Flatten, Dense, Lambda, Conv2D, Cropping2D, Dropout
    #from keras.layers.pooling import MaxPool2D
    model = Sequential()
    model.add(Cropping2D(cropping = ((70,25),(0,0)), input_shape = (160,320,3)))
    model.add(Lambda(lambda x: x / 255.0 - 0.5))
    model.add(Dropout(0.2))
    model.add(Conv2D(24,5,5, activation="relu", subsample = (2,2)))
    model.add(Conv2D(36,5,5, activation="relu", subsample = (2,2)))
    model.add(Conv2D(48,5,5, activation="relu", subsample = (2,2)))
    model.add(Conv2D(64,3,3, activation="relu"))
    model.add(Conv2D(64,3,3, activation="relu"))
    model.add(Flatten())
    model.add(Dropout(0.5))
    model.add(Dense(100))
    model.add(Dropout(0.5))
    model.add(Dense(50))
    model.add(Dense(10))
    model.add(Dense(1))
    
    
    model.compile(loss='mse', optimizer ='adam')
    model.fit_generator(train_generator,
                steps_per_epoch=np.ceil(len(train_samples)/batch_size),
                validation_data=validation_generator,
                validation_steps=np.ceil(len(validation_samples)/batch_size),
                epochs=3, verbose=1)
    
    model.save('model.h5')
    print('Model was saved successfully.')
