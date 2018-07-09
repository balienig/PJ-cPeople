from flask import Flask
import json
import requests
import os
import cv2
import numpy as np
from random import shuffle

pathFloderStore = 'train/StoreFace/'
LR = 1e-3
pathA = 'StoreFaceimage/ujf8hYGG/'
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

convnet = conv_2d(convnet, 6, 2, activation='relu')
convnet = max_pool_2d(convnet, 2)

convnet = conv_2d(convnet, 16, 2, activation='relu')
convnet = max_pool_2d(convnet, 2)

convnet = fully_connected(convnet, 32, activation='relu')
convnet = dropout(convnet, 0.5)

convnet = fully_connected(convnet, numClass, 
                          activation='softmax')
convnet = regression(convnet, optimizer='adam', learning_rate=LR, loss='categorical_crossentropy', name='targets')

model = tflearn.DNN(convnet, tensorboard_dir='log')

if os.path.exists('{}.meta'.format(MODEL_NAME)):
    model.load(MODEL_NAME)
    print('model loaded!')
# print('AAA')
for folder , dirs, files in os.walk(pathA):
    for file in files:
        path = os.path.join(folder,file)
        img = cv2.imread(path)
        img = cv2.resize(img,(50,50))
        img = np.array(img).reshape(-1,50,50,1)/255
        model_out = model.predict(img)
        index = np.argmax(model_out[0])
        print(listName[index],model_out[0][index])
    
