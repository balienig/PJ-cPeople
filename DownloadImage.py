import urllib.request as urllib
from urllib.request import urlopen
from bs4 import BeautifulSoup
import cv2
import numpy as np



images_link = 'http://192.168.1.105:8888/mKQlnIKY/'
image_urls = str(urlopen(images_link).read())
pic_num = 0

soup = BeautifulSoup(image_urls, 'html.parser')
soup = soup.find_all('a')

for i in soup:
    print(i['href'])
    url = images_link+str(i['href'])
    urllib.urlretrieve(url, "StoreImage/"+str(i['href']))
 