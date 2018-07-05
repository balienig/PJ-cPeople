import os
import cv2
import numpy as np
from random import shuffle
pathFloderStore = 'train/StoreFace/'
LR = 1e-3

MODEL_NAME = 'classifyPeople2-{}-{}.model'.format(LR, '2conv-basic')

listName = []
ImageSize = 50
for folder , dirs, files in os.walk(pathFloderStore):
    # print(folder)
    nameFolder = folder.split('/')
    if nameFolder[2] != "" :
        listName.append(nameFolder[2])


listLabel = []
numClass = len(listName)
for i in range(numClass):
    listLabel.append([0]*numClass)
    listLabel[i][i] = 1


training_data = []
for folder , dirs, files in os.walk(pathFloderStore):
    for file in files:
        path = os.path.join(folder,file)
        word_label = path.split('/')
        index = listName.index(word_label[2])
        # print(listLabel[index])
        img = cv2.imread(path,cv2.IMREAD_GRAYSCALE)
        img = cv2.resize(img,(ImageSize,ImageSize))
        training_data.append([np.array(img),np.array(listLabel[index])])
shuffle(training_data)
# print(training_data)
# print(listName)

import tflearn
from tflearn.layers.conv import conv_2d, max_pool_2d
from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn.layers.estimator import regression

# import tensorflow as tf
# tf.reset_default_graph()

convnet = input_data(shape=[None, ImageSize, ImageSize, 1], name='input')

convnet = conv_2d(convnet, 32, 5, activation='relu')
convnet = max_pool_2d(convnet, 2)

convnet = conv_2d(convnet, 64, 5, activation='relu')
convnet = max_pool_2d(convnet, 2)

convnet = fully_connected(convnet, numClass*10, activation='relu')
convnet = dropout(convnet, 0.8)

convnet = fully_connected(convnet, numClass, 
                          activation='softmax')
convnet = regression(convnet, optimizer='adam', learning_rate=LR, loss='categorical_crossentropy', name='targets')

model = tflearn.DNN(convnet, tensorboard_dir='log')

num_test = len(training_data) * 20 // 100

print(len(training_data))
print(num_test)

train = training_data[:-num_test]
test = training_data[-num_test:]

X = np.array([i[0] for i in train]).reshape(-1,ImageSize,ImageSize,1)/255
Y = [i[1] for i in train]

test_x = np.array([i[0] for i in test]).reshape(-1,ImageSize,ImageSize,1)/255
test_y = [i[1] for i in test]

model.fit({'input': X}, {'targets': Y}, n_epoch=400, validation_set=({'input': test_x}, {'targets': test_y}), 
    snapshot_step=100, show_metric=True, run_id=MODEL_NAME)

model.save(MODEL_NAME)