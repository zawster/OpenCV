import cv2

face_cascade = cv2.CascadeClassifier('cascades/data/haarcascade_frontalface_alt2.xml')

cap = cv2.VideoCapture(2)

while(True):
    # Capture frame by frame
    ret, frame = cap.read()
    gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray,(5,5),0)
    edged = cv2.Canny(blur,50,150)

    # Display the resulting frame
    cv2.imshow('Colored',frame)
    cv2.imshow('Edged',edged)
    if cv2.waitKey(20) & 0xFF == ord('q'):  # if 'q' is pressed then quit the window
        break

# Everything is done, release the capture
cap.release()
cv2.destroyAllWindows