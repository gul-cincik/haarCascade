import cv2
import numpy as np

video = cv2.VideoCapture('video.mp4')
body_cascade = cv2.CascadeClassifier('haarcascade_fullbody.xml')

while True:
    ret,frame = video.read()
    griton = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    insan = body_cascade.detectMultiScale(griton,1.1,5)

    for (x,y,w,h) in insan:
        cv2.rectangle(frame, (x,y), (x+w,x+h), (255,0,255),3)

    cv2.imshow('insanlar', frame)
    if cv2.waitKey(25) & 0xFF == ord('q'):
        break

video.release()
cv2.destroyAllWindows()