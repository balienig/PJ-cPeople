import tflearn
from tflearn.layers.conv import conv_2d, max_pool_2d
from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn.layers.estimator import regression
import numpy as np
import os
import cv2
IMAGE_SIZE = 50
LR = 1e-3
pathFloderStore = 'train/StoreFace/'
path = 'StoreFaceimage/WDeZJwki'
MODEL_NAME = 'classifyPeople-{}-{}.model'.format(LR, '2conv-basic')

convnet = input_data(shape=[None, 50, 50, 1], name='input')
convnet = conv_2d(convnet, 64, 2, activation='relu',padding= 'same')
convnet = max_pool_2d(convnet, 2,padding= 'same')

convnet = conv_2d(convnet, 64, 2, activation='relu',padding= 'same')
convnet = max_pool_2d(convnet, 2,padding= 'same',strides=2)

convnet = fully_connected(convnet,64, activation= 'relu')
# convnet = dropout(convnet, 0.5)

#2X2 3X3 5X5 7X7
#16 32 64 
#strides = 1 , 2 , 3 
#โหวต เอาไว้หลังสุด หลังจาก หาพารามิตเตอร์ หาค่าดีที่สุด แล้วก็ทำแบบโหวตกับไม่โหวต
#ทำ 5 คลาส แล้วก็ 10 คลาส
#Leru ลองตัดอันแรกๆ ออก 
#ตอนเทสไม่ต้องใส่ 
#dropout ลองใส่ 0
#Asual
#เพิ่ม convolu 3 ชั้น
#max 3 strides 3
#ส่วนมาก dropout ใส่ก่อน fully
#1000 รอบอย่างเดียว
convnet = fully_connected(convnet, 10, 
                          activation='softmax')
convnet = regression(convnet, optimizer='adam', learning_rate=LR, loss='categorical_crossentropy', name='targets')

model = tflearn.DNN(convnet, tensorboard_dir='log')

if os.path.exists('{}.meta'.format(MODEL_NAME)):
    model.load(MODEL_NAME)
    print('model loaded!')

listName = []
for folder , dirs, files in os.walk(pathFloderStore):
    nameFolder = folder.split('/')
    if nameFolder[2] != "" :
        listName.append(nameFolder[2])

for folder , dirs, files in os.walk(path):
    for file in files:
        img = cv2.imread(os.path.join(folder,file))
        img = cv2.resize(img,(IMAGE_SIZE,IMAGE_SIZE))
        img = np.array(img).reshape(-1,IMAGE_SIZE,IMAGE_SIZE,1)/255
        model_out = model.predict(img)[1]
        index = np.argmax(model_out)
        print(index)
        print(listName[index],model_out[index])

