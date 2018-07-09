import sys
import cv2
from PyQt5.QtCore import QTimer
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtWidgets import QDialog, QApplication
from PyQt5.uic import loadUi
import requests
import os
import numpy as np
from random import shuffle
import tflearn
from tflearn.layers.conv import conv_2d, max_pool_2d
from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn.layers.estimator import regression

url = 'http://127.0.0.1:9000/'
numFacePic = 100
pathFloderStore = 'train/StoreFace/'
pathTrainning = 'trainningStatus/'
class Life2Coding(QDialog):
    def __init__(self):
        super(Life2Coding,self).__init__()
        loadUi('train/GUI.ui',self)
        self.image = None
        self.faceCascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
        self.start_webcam()
        self.checkNameButton.clicked.connect(self.checkName)
        self.captureButton.clicked.connect(self.capturePic)
        self.trainButton.clicked.connect(self.train)
        self.statusCapture = False
        self.countPic = 0
        self.status = 'false'
        self.NameFloder = ""

    def start_webcam(self):
        self.capture = cv2.VideoCapture(0)
        self.capture.set(cv2.CAP_PROP_FRAME_HEIGHT,480)
        self.capture.set(cv2.CAP_PROP_FRAME_WIDTH,640)

        self.timer = QTimer(self)
        self.timer.timeout.connect(self.update_frame)
        self.timer.start(5)
    
    def update_frame(self):
        ret,self.image = self.capture.read()
        self.image = cv2.flip(self.image,1)
        self.image = self.detect_face(self.image)
        self.displayImage(self.image,1)

    def displayImage(self,img,window = 1):
        qformat = QImage.Format_Indexed8

        if len(img.shape) == 3 :
          #  print(len(img.shape))
            if img.shape[2] == 4:
                qformat = QImage.Format_RGBA8888
            else :
                qformat = QImage.Format_RGB888
        
        outImage = QImage(img,img.shape[1],img.shape[0],img.strides[0],qformat)
        
        outImage = outImage.rgbSwapped()

        if window == 1:
            self.labelImage.setPixmap(QPixmap.fromImage(outImage))
            self.labelImage.setScaledContents(True)

    def detect_face(self,img):
        # gray = cv2.cvtColor(img ,cv2.COLOR_BAYER_BG2GRAY)
        faces = self.faceCascade.detectMultiScale(img,1.2,5,minSize = (90,90))
        img2 = img  
        for(x,y,w,h) in faces:
            # print(y+h,x+w)
            cv2.rectangle(img,(x,y),(x+w,y+h),(0,0,255),2)
            if(self.statusCapture == True):
                if(x+w > 150 and y+h > 150):
                    self.countPic = self.countPic + 1
                    self.numPicture.setText(str(self.countPic)+"/"+str(numFacePic))
                
                    cv2.imwrite("train/StoreFace/"+self.NameFloder+"/"+self.NameFloder+"."+str(self.countPic)+".png",img2[y:y+h,x:x+w])
                    if self.countPic == numFacePic :
                        self.statusCapture = False
                        self.numPicture.setText("Success")
            
        return img

    def capturePic(self,img):
        if(self.status == 'True'):
            self.statusCapture = True
            self.countPic = 0
            print(self.NameFloder)
            try:
                if not os.path.exists("train/StoreFace/"+self.NameFloder):
                    os.makedirs("train/StoreFace/"+self.NameFloder)
            except OSError:
                print ('Error: Creating directory. ')
        else :
            print("error")
        

    def checkName(self):
        myText = self.textName.toPlainText()
        listName = []
        for folder , dirs, files in os.walk(pathFloderStore):
            nameFolder = folder.split('/')
            if nameFolder[2] != "" :
                listName.append(nameFolder[2])
        count = 0
        for i in listName:
            if(i == myText):
                count = 1
                self.status = 'False'
        if count == 0:
            self.status = 'True'
        print(self.status)
        if(self.status == 'True'):
            self.NameFloder = myText

    def train(self):
        self.percentTrain.setText("...Trainning")

        pathFloderStore = 'train/StoreFace/'
        LR = 1e-3

        MODEL_NAME = 'classifyPeople-{}-{}.model'.format(LR, '2conv-basic')

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
                # print(img)
                img = cv2.resize(img,(ImageSize,ImageSize))
                # print(np.shape(img))
                training_data.append([np.array(img),np.array(listLabel[index])])
        shuffle(training_data)
        # print(training_data)



# import tensorflow as tf
# tf.reset_default_graph()
        
        convnet = input_data(shape=[None, ImageSize, ImageSize, 1], name='input')
        convnet = conv_2d(convnet, 32, (5,5), activation='relu',padding= 'valid')
        convnet = max_pool_2d(convnet, 2,padding= 'valid')

        convnet = conv_2d(convnet, 32, (5,5), activation='relu',padding= 'valid')
        convnet = max_pool_2d(convnet, 2,padding= 'valid')

        convnet = fully_connected(convnet,30, activation= 'relu')
        convnet = dropout(convnet, 0.6)

        convnet = fully_connected(convnet, numClass, 
                          activation='softmax')
        convnet = regression(convnet, optimizer='adam', learning_rate=LR, loss='categorical_crossentropy', name='targets')

        model = tflearn.DNN(convnet, tensorboard_dir='log')

        num_test = len(training_data) * 20 // 100


        train = training_data[:-num_test]
        test = training_data[-num_test:]

        print(np.shape(train[0][0]))
        
        X = np.array([i[0] for i in train]).reshape(-1,ImageSize,ImageSize,1)/255

        print(np.shape(X[0]))
        Y = [i[1] for i in train]
        test_x = np.array([i[0] for i in test]).reshape(-1,ImageSize,ImageSize,1)/255
        test_y = [i[1] for i in test]

        # print(np.shape(X))

        model.fit({'input': X}, {'targets': Y}, n_epoch=200, validation_set=({'input': test_x}, {'targets': test_y}), 
        batch_size= numClass*10,snapshot_step=100, show_metric=True, run_id=MODEL_NAME)

        model.save(MODEL_NAME)
        self.percentTrain.setText('Success')


if __name__=='__main__':
    app = QApplication(sys.argv)
    window = Life2Coding()
    window.setWindowTitle('train Gui')
    window.show()
    sys.exit(app.exec_())