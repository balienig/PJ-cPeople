import numpy as np
import cv2
from scipy import ndimage
cap = cv2.VideoCapture('/home/balienig/Documents/Git/PJ-cPeople/train/trainRnn/vid/20180425_202705.mp4')

while(True):
    # Capture frame-by-frame
    ret, frame = cap.read()
    frame = ndimage.rotate(frame, 270)
    # Our operations on the frame come here
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Display the resulting frame
    cv2.imshow('frame',gray)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()