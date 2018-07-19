import os


for folder ,dirs,path in os.walk('train/trainRnn/imageTrain/run'):
    print(folder.join(path))


    
import numpy as np
import tflearn


