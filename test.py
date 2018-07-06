from flask import Flask
import json
import requests
import os
import cv2
import numpy as np
from random import shuffle

pathFloderStore = 'train/StoreFace/'
LR = 1e-3

MODEL_NAME = 'classifyPeople-{}-{}.model'.format(LR, '2conv-basic')

app = Flask(__name__)
listName = []
for folder , dirs, files in os.walk(pathFloderStore):
    nameFolder = folder.split('/')
    if nameFolder[2] != "" :
        listName.append(nameFolder[2])


numClass = len(listName)


ImageSize = 50

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

if os.path.exists('{}.meta'.format(MODEL_NAME)):
    model.load(MODEL_NAME)
    print('model loaded!')

img = cv2.imread('StoreFaceimage/KRc48xOv/0.png')
img = cv2.resize(img,(50,50))
img = np.array(img).reshape(-1,50,50,1)/255
model_out = model.predict(img)
print(numClass)