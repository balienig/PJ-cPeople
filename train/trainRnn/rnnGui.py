import sys
import cv2
from PyQt5.QtCore import QTimer
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtWidgets import QDialog, QApplication
from PyQt5.uic import loadUi
import numpy as np
import os
import six.moves.urllib as urllib
from object_detection.utils import label_map_util
from object_detection.utils import visualization_utils as vis_util
import tarfile
import tensorflow as tf
from collections import defaultdict
from matplotlib import pyplot as plt
from PIL import Image
from object_detection.utils import ops as utils_ops
# from numbapro import vectorize, cuda

# @vectorize(['float32(float32, float32)','float64(float64, float64)'], target = 'cuda')

class Life2Coding(QDialog):
    def __init__(self):
        super(Life2Coding,self).__init__()
        loadUi('train/trainRnn/guiRnn.ui',self)
        MODEL_NAME = 'ssd_mobilenet'
        self.PATH_TO_CKPT = MODEL_NAME + '/frozen_inference_graph.pb'
        self.PATH_TO_LABELS = os.path.join('data', 'mscoco_label_map.pbtxt')
        NUM_CLASSES = 90
        self.detection_graph = tf.Graph()
        self.detectionGraph()
        label_map = label_map_util.load_labelmap(self.PATH_TO_LABELS)
        self.categories = label_map_util.convert_label_map_to_categories(label_map, max_num_classes=NUM_CLASSES, use_display_name=True)
        self.category_index = label_map_util.create_category_index(self.categories)    
        self.start_webcam()

    def start_webcam(self):
        self.capture = cv2.VideoCapture(0)
        self.capture.set(cv2.CAP_PROP_FRAME_HEIGHT,480)
        self.capture.set(cv2.CAP_PROP_FRAME_WIDTH,640)

        self.timer = QTimer(self)
        self.timer.timeout.connect(self.update_frame)
        self.timer.start()
    
    def update_frame(self):
        ret,self.image = self.capture.read()
        self.image = cv2.flip(self.image,1)
        self.image = self.detector(self.image)
        self.displayImage(self.image,1)

    def displayImage(self,img,window = 1):
        qformat = QImage.Format_Indexed8
        # print(qformat)
        if len(img.shape) == 3 :
            if img.shape[2] == 4:
                qformat = QImage.Format_RGBA8888
            else :
                qformat = QImage.Format_RGB888
        
        outImage = QImage(img,img.shape[1],img.shape[0],img.strides[0],qformat)
        
        outImage = outImage.rgbSwapped()

        
        if window == 1:
            self.labelImage.setPixmap(QPixmap.fromImage(outImage))
            self.labelImage.setScaledContents(True)

    def detectionGraph(self):
        with self.detection_graph.as_default():
            od_graph_def = tf.GraphDef()
            with tf.gfile.GFile(self.PATH_TO_CKPT, 'rb') as fid:
                serialized_graph = fid.read()
                od_graph_def.ParseFromString(serialized_graph)
                tf.import_graph_def(od_graph_def, name='')

    def detector(self,image_np):
        with self.detection_graph.as_default():
            with tf.Session(graph=self.detection_graph) as sess:
                image_np_expanded = np.expand_dims(image_np, axis=0)
                image_tensor = self.detection_graph.get_tensor_by_name('image_tensor:0')
                boxes = self.detection_graph.get_tensor_by_name('detection_boxes:0')
                scores = self.detection_graph.get_tensor_by_name('detection_scores:0')
                classes = self.detection_graph.get_tensor_by_name('detection_classes:0')
                num_detections = self.detection_graph.get_tensor_by_name('num_detections:0')
                (boxes, scores, classes, num_detections) = sess.run(
                [boxes, scores, classes, num_detections],
                feed_dict={image_tensor: image_np_expanded})
    #   print(category_index)
                vis_util.visualize_boxes_and_labels_on_image_array(
                    image_np,
                    np.squeeze(boxes),
                    np.squeeze(classes).astype(np.int32),
                    np.squeeze(scores),
                    self.category_index,
                use_normalized_coordinates=True,
                line_thickness=3)
        return image_np

        
    


if __name__=='__main__':
    app = QApplication(sys.argv)
    window = Life2Coding()
    window.setWindowTitle('train Gui')
    window.show()
    sys.exit(app.exec_())