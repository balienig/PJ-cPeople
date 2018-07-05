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
from random import shuffle

pathFloderStore = 'train/StoreFace/'
LR = 1e-3

MODEL_NAME = 'classifyPeople2-{}-{}.model'.format(LR, '2conv-basic')

app = Flask(__name__)

ImageSize = 50
LR = 1e-3

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

# if os.path.exists('{}.meta'.format(MODEL_NAME)):
#     model.load(MODEL_NAME)
#     print('model loaded!')

def predit(str):
    url = str.split('*')
    path = url[0]+"/"+url[1]+"/"+url[2]
    print(path)
    img = cv2.imread(path)
    img = cv2.resize(img,(50,50))
    img = np.array(img).reshape(-1,50,50,1)/255
    model_out = model.predict(img)
    print(model_out)
    str_label = ""
    # value = model_out[np.argmax(model_out)]
    if np.argmax(model_out) == 0: str_label = 'big'
    elif np.argmax(model_out) == 1: str_label = 'o'
    elif np.argmax(model_out) == 2 : str_label = 'team'
    return str_label

@app.route('/FaceDetection/<str>', methods = ['GET'])        
def FaceDetection(str):
    A = predit(str)
    return json.dumps({"name": A})

listName = []

@app.route('/statusName/<str>', methods = ['GET'])
def statusName(str):
    status = checkName(str)
    return json.dumps({"status":status})




def checkName(str):

    for folder , dirs, files in os.walk(pathFloderStore):
    # print(folder)
        nameFolder = folder.split('/')
        if nameFolder[2] != "" :
            listName.append(nameFolder[2])

    status = 'True'
    for i in listName:
        if(i == str):
            return 'False'
    return 'True'

def trainning(self):
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


if __name__ == '__main__':
    app.run(host='127.0.0.1', port=9000)