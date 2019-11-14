import numpy as np 
import cv2


face_cascade = cv2.CascadeClassifier('cascades/data/haarcascade_frontalface_alt2.xml')
eye_cascade = cv2.CascadeClassifier('cascades/data/haarcascade_eye.xml')
cap = cv2.VideoCapture(2)

while(True):
    # Capture frame by frame
    ret, frame = cap.read()
    gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray,(5,5),0) # blaured frames
    edged = cv2.Canny(blur,50,150)  # edged frames
    faces = face_cascade.detectMultiScale(blur, scaleFactor=1.5, minNeighbors=5) # actually detecting a face
  
    for (x,y,w,h) in faces:   # detected face array
        
        roi_gray = gray[y:y+h, x:x+w]
        roi_color = frame[y:y+h, x:x+w]

        color = (0,255,0)   # BGR rectangle color
        stroke = 2
        end_cord_x = x + w
        end_cord_y = y + h
        cv2.rectangle(frame,(x,y),(end_cord_x,end_cord_y), color, stroke)  # Draw rectange on face
        # Detecting the eyes
        eyes = eye_cascade.detectMultiScale(roi_gray)
        for (ex,ey,ew,eh) in eyes:
            cv2.rectangle(roi_color,(ex,ey),(ex+ew, ey+eh),(0,255,255),2)

    # Display the resulting frame
    cv2.imshow('Face Dtector',frame)
    if cv2.waitKey(20) & 0xFF == ord('q'):
        break

# Everything is done, release the capture
cap.release()
cv2.destroyAllWindows