# -*- coding: utf-8 -*-
"""
Created on Mon Jan 29 23:47:33 2018

@author: Poojitha
"""

# part1 - Building the CNN

#importing Keras
import tensorflow as tf

from keras.models import Sequential
from keras.layers import Convolution2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense

from keras import metrics
# Intialising the cnn
classifier= Sequential()

#step 1- convolution
classifier.add(Convolution2D(32,3,3,input_shape=(64,64,3), activation='relu'))

# step 2= max pooling
classifier.add(MaxPooling2D(pool_size=(2,2)))

# adding 2nd Convolutional layer
classifier.add(Convolution2D(32,3,3, activation='relu'))
classifier.add(MaxPooling2D(pool_size=(2,2)))
#adding 3rd Convolutional layer
classifier.add(Convolution2D(32,3,3, activation='relu'))
classifier.add(MaxPooling2D(pool_size=(2,2)))

#adding 4th Convolutional layer

classifier.add(Convolution2D(32,3,3, activation='relu'))
classifier.add(MaxPooling2D(pool_size=(2,2)))


#step-3- Flattening
classifier.add(Flatten())

# step 4- Full connection
classifier.add(Dense(128,activation = 'relu'))

classifier.add(Dense(4,activation = 'softmax',))

# Compiling CNN
classifier.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['categorical_accuracy'])

### Image Pre-processing

from sklearn.preprocessing import StandardScaler
sc = StandardScaler()


from sklearn.preprocessing import LabelEncoder

import keras
import keras.utils
from keras import utils as np_utils
from keras.preprocessing.image import ImageDataGenerator

train_datagen = ImageDataGenerator(
        rescale=1./255,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True)

test_datagen = ImageDataGenerator(rescale=1./255)



train_set = train_datagen.flow_from_directory('training_set',
        target_size=(64, 64),
        batch_size=32,
        class_mode='categorical')
#train_set= sc.fit_transform(train_set)
#train_labels_one_hot = np_utils.to_categorical(train_set)
#test_labels_one_hot = to_categorical(test_labels)

test_set = test_datagen.flow_from_directory('test_set',
        target_size=(64, 64),
        batch_size=32,
        class_mode='categorical')

#### Part 2 - Fitting CNN to Images

classifier.fit_generator(train_set,
        steps_per_epoch=430,
        epochs=10,
        validation_data=test_set,
        validation_steps=128)

### Confusion matrix
from sklearn.metrics import classification_report,confusion_matrix
import numpy as np
import matplotlib.pyplot as plt  
import math 

size_test= int(math.ceil(len(test_set.classes) / 32))

Y_pred = classifier.predict_generator(test_set,size_test)
y_pred = np.argmax(Y_pred, axis=1)



y_true = np.array([0] * 30 + [1] * 30 + [2]* 30  + [3]* 38)


print('Confusion Matrix')
print(confusion_matrix(y_true, y_pred))
print('Classification Report')
target_names = ['Chytrids', 'Phylum Basidiomycota', 'Sac fungi','Zygote fungi']
print(classification_report(test_set.classes, y_pred, target_names=target_names))



# Making new predictions
import numpy as np
from keras.preprocessing import image

test_image= image.load_img('prediction/1.jpg',target_size=(64, 64) )
test_image= image.img_to_array(test_image)
test_image= np.expand_dims(test_image, axis=0)
classifier.predict(test_image)

result= classifier.predict(test_image)
train_set.class_indices
if result[0][0]==0 :
     prediction='Chytrids'
elif result[0][0]==1:
    prediction= 'Phylum Basidiomycota'
elif result[0][0]==2:
    prediction='Sac fungi'
else:
    prediction='Zygote fungi'
prediction