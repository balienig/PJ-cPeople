import sys
import cv2
from PyQt5.QtCore import QTimer
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtWidgets import QDialog, QApplication
from PyQt5.uic import loadUi
import requests
import os
url = 'http://127.0.0.1:9000/'
pathCheckName = 'statusName/'
numFacePic = 100
class Life2Coding(QDialog):
    def __init__(self):
        super(Life2Coding,self).__init__()
        loadUi('train/GUI.ui',self)
        self.image = None
        self.faceCascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
        self.start_webcam()
        self.checkNameButton.clicked.connect(self.checkName)
        self.captureButton.clicked.connect(self.capturePic)
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
        

    def checkName(self):
        myText = self.textName.toPlainText()
        urlCheckName = url+pathCheckName + myText
        respone = requests.get(urlCheckName)
        self.status = respone.json()
        self.status = self.status['status']
        print(self.status)
        if(self.status == 'True'):
            self.NameFloder = myText

        
        # print(status['status'])
    


if __name__=='__main__':
    app = QApplication(sys.argv)
    window = Life2Coding()
    window.setWindowTitle('train Gui')
    window.show()
    sys.exit(app.exec_())