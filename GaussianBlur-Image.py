import cv2
import numpy as np

image = cv2.imread('test_image.jpg')
lane_image = np.copy(image)  # copying the image array to lane_image

width = 550
height = 500 
gray = cv2.cvtColor(image,cv2.COLOR_RGB2GRAY) # converting RGB to Gray scale image
blur = cv2.GaussianBlur(gray,(5,5),0)  # Reduce Noice form image

resized = cv2.resize(blur,(width,height))
cv2.imshow('Gaussian Blur',resized)

cv2.waitKey(0)
cv2.destroyAllWindows()
