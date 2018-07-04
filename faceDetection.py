from flask import Flask
import json
import requests
import tflearn
import os
from tflearn.layers.conv import conv_2d, max_pool_2d
from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn.layers.estimator import regression
import cv2
import numpy as np
app = Flask(__name__)

IMAGE_SIZE = 50
LR = 1e-3
MODEL_NAME = 'classifyPeople-{}-{}.model'.format(LR, '2conv-basic')

convnet = input_data(shape=[None, IMAGE_SIZE, IMAGE_SIZE, 1], name='input')

convnet = conv_2d(convnet, 32, 5, activation='relu')
convnet = max_pool_2d(convnet, 2)

convnet = conv_2d(convnet, 64, 5, activation='relu')
convnet = max_pool_2d(convnet, 2)

convnet = fully_connected(convnet, 30, activation='relu')
convnet = dropout(convnet, 0.8)

convnet = fully_connected(convnet, 3, 
                          activation='softmax')
convnet = regression(convnet, optimizer='adam', learning_rate=LR, loss='categorical_crossentropy', name='targets')

model = tflearn.DNN(convnet, tensorboard_dir='log')

if os.path.exists('{}.meta'.format(MODEL_NAME)):
    model.load(MODEL_NAME)
    print('model loaded!')

def predit(str):
    url = str.split('*')
    path = url[0]+"/"+url[1]+"/"+url[2]
    img = cv2.imread(path)
    img = cv2.imread(img, cv2.IMREAD_GRAYSCALE)
    img = cv2.resize(img,(50,50))

    predict = model.predict(img)
    return str(predict)

@app.route('/FaceDetection/<str>', methods = ['GET'])        
def FaceDetection(str):
    A = predit(str)
    return json.dumps({"str": A})

if __name__ == '__main__':
    app.run(host='127.0.0.1', port=9000)