

import cv2
import numpy as np
# import matplotlib.pyplot as plt
def DetectEdge(image):
    gray = cv2.cvtColor(image,cv2.COLOR_RGB2GRAY) # converting RGB colerd image to Gray scale image
    blur = cv2.GaussianBlur(gray,(5,5),0)  # Reduce Noice form image
    edged = cv2.Canny(blur,50,150)
    return edged

image = cv2.imread('test_image.jpg')
lane_image = np.copy(image)  # copying the image array to lane_image

edge=DetectEdge(lane_image)

cv2.imshow('Edged Image',edge)

cv2.waitKey(0)
cv2.destroyAllWindows()



