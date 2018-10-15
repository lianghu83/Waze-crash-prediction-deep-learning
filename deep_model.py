

from __future__ import print_function
import os
import numpy as np
dirpath = '/home/me592/'
dat = np.loadtxt(dirpath+"dat_norm.txt",delimiter=",")

X = dat[:,1:]
y = dat[:,0]

from sklearn.cross_validation import train_test_split
x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)
x_test, x_val, y_test, y_val = train_test_split(x_test, y_test, test_size=0.5, random_state=0)


import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Activation
from keras.layers.normalization import BatchNormalization
from keras.optimizers import SGD

batch_size = 2000
num_classes = 11
epochs = 30

# convert class vectors to binary class matrices
y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)
y_val = keras.utils.to_categorical(y_val, num_classes)

input_shape = (x_train.shape[1],)

model = Sequential()
model.add(Dense(40,input_shape=input_shape))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(Dense(33))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(Dense(26))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(Dense(18))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(Dropout(0.2))
model.add(Dense(num_classes))
model.add(Activation('softmax'))

learning_rate = 0.1
decay_rate = learning_rate/epochs
momentum=0.8

opt = SGD(lr=learning_rate, momentum=momentum, decay=decay_rate)

model.compile(loss=keras.losses.categorical_crossentropy,
              optimizer=opt,
              metrics=['accuracy'])

model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs,
          verbose=1, validation_data=(x_val, y_val))


score = model.evaluate(x_test, y_test, verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])



