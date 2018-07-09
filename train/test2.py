import os
import cv2
import numpy as np
ImageSize = 50
from random import shuffle
pathFloderStore = 'train/StoreFace/'

listName = []
for folder , dirs, files in os.walk(pathFloderStore):
    nameFolder = folder.split('/')
    if nameFolder[2] != "" :
        listName.append(nameFolder[2])


listLabel = []
numClass = len(listName)
for i in range(numClass):
    listLabel.append([0]*numClass)
    listLabel[i][i] = 1
from keras.utils import to_categorical
import matplotlib.pyplot as plt
training_data = []
for folder , dirs, files in os.walk(pathFloderStore):
    for file in files:
        path = os.path.join(folder,file)
        word_label = path.split('/')
        index = listName.index(word_label[2])
        img = cv2.imread(path,cv2.IMREAD_GRAYSCALE)
        img = cv2.resize(img,(ImageSize,ImageSize))
        training_data.append([np.array(img),to_categorical(index)])
shuffle(training_data)


num_test = len(training_data) * 20 // 100


train = training_data[:-num_test]
test = training_data[-num_test:]

X_train = np.array([i[0] for i in train])
Y_train = [i[1] for i in train]
X_test = np.array([i[0] for i in test])
Y_test = [i[1] for i in test]

X_train = X_train.astype('float32')
X_test = X_test.astype('float32')
X_train /= 255
X_test /= 255

X_train = X_train.reshape(X_train.shape[0],ImageSize,ImageSize,1)
X_test = X_test.reshape(X_test.shape[0],ImageSize,ImageSize,1)

# mlp = Sequential()
# mlp.add( Dense(100, input_dim=784, activation='tanh') )
# mlp.add( Dense(10, activation='softmax') )
# mlp.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
# print(Y_test)
# print(X_train.shape,X_test.shape,Y_train.shape,Y_test.shape)
# print(Y_test)

import keras
from keras.models import Sequential,Input,Model
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.layers.normalization import BatchNormalization
from keras.layers.advanced_activations import LeakyReLU

batch_size = 64
epochs = 20
num_classes = 10

fashion_model = Sequential()
fashion_model.add(Conv2D(32, kernel_size=(3, 3),activation='linear',input_shape=(50,50,1),padding='same'))
fashion_model.add(LeakyReLU(alpha=0.1))
fashion_model.add(MaxPooling2D((2, 2),padding='same'))
fashion_model.add(Conv2D(64, (3, 3), activation='linear',padding='same'))
fashion_model.add(LeakyReLU(alpha=0.1))
fashion_model.add(MaxPooling2D(pool_size=(2, 2),padding='same'))
fashion_model.add(Conv2D(128, (3, 3), activation='linear',padding='same'))
fashion_model.add(LeakyReLU(alpha=0.1))                  
fashion_model.add(MaxPooling2D(pool_size=(2, 2),padding='same'))
fashion_model.add(Flatten())
fashion_model.add(Dense(128, activation='linear'))
fashion_model.add(LeakyReLU(alpha=0.1))                  
fashion_model.add(Dense(num_classes, activation='softmax'))
fashion_model.compile(loss=keras.losses.categorical_crossentropy, optimizer=keras.optimizers.Adam(),metrics=['accuracy'])
print(fashion_model.summary())

fashion_train = fashion_model.fit(X_train, Y_train, batch_size=batch_size,epochs=epochs,verbose=1,validation_data=(X_test, Y_test))