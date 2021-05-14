import cv2
import numpy as np

image = cv2.imread('r2.jpg')
body_cascade = cv2.CascadeClassifier('haarcascade_fullbody.xml')
griton = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
bodies = body_cascade.detectMultiScale(griton, 1.8, 2)

for x,y,w,h in bodies:
    cv2.rectangle(image,(x,y), (x+w,y+h), (255,0,255), 3)

cv2.imshow('body', image)
cv2.waitKey(0)
cv2.destroyAllWindows()