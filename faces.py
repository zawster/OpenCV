import numpy as np 
import cv2
import pickle

face_cascade = cv2.CascadeClassifier('cascades/data/haarcascade_frontalface_alt2.xml')
recognizer = cv2.face.LBPHFaceRecognizer_create()   #  for face recognization (recognizer) train model
recognizer.read("trainner.yml")

labels = {}
with open("labels.pickle","rb") as f:
    og_labels = pickle.load(f)
    labels = {v:k for k,v in og_labels.items()} # reversing the values of dictionary i.e key2val and val2key

cap = cv2.VideoCapture(2)

while(True):
    # Capture frame by frame
    ret, frame = cap.read()
    gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray,(5,5),0) # blaured frames
    edged = cv2.Canny(blur,50,150)  # edged frames
    faces = face_cascade.detectMultiScale(blur, scaleFactor=1.5, minNeighbors=5) # actually detecting a face
  
    for (x,y,w,h) in faces:   # detected face array
        # print(x,y,w,h)  # print numbers of detected face
        roi_gray = gray[y:y+h, x:x+w]
        roi_color = frame[y:y+h, x:x+w]

        id_, conf = recognizer.predict(roi_gray)

        if conf >= 4 and conf <= 85:
            # print(id_)
            # print(labels[id_])
            font = cv2.FONT_HERSHEY_SIMPLEX
            name = labels[id_]
            color = (255,255,255)
            stroke = 1
            cv2.putText(frame, name, (x,y), font, 1, color, stroke, cv2.LINE_AA)

        img_item = '2.png'  
        cv2.imwrite(img_item,roi_color)

        color = (0,255,0)   # BGR rectangle color
        stroke = 2
        end_cord_x = x + w
        end_cord_y = y + h
        cv2.rectangle(frame,(x,y),(end_cord_x,end_cord_y), color, stroke)  # Draw rectange on face
    # Display the resulting frame
    cv2.imshow('Face Recognizer',frame)
    cv2.imshow('Edged Face Detection',edged)
    if cv2.waitKey(20) & 0xFF == ord('q'):
        break

# Everything is done, release the capture
cap.release()
cv2.destroyAllWindows