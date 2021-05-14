import cv2
import numpy as np

image = cv2.imread('kalabalik.jpg')

yuz_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

griton = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
yuzler = yuz_cascade.detectMultiScale(griton,1.1,4)

for (x,y,w,h) in yuzler:
    cv2.rectangle(image,(x,y), (x+w, y+h),(0,255,0),3)

cv2.imshow('yuzler', image)
cv2.waitKey(0)
cv2.destroyAllWindows()