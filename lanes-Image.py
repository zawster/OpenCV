import cv2
import numpy as np
# import matplotlib.pyplot as plt
def garyScale(image):
    gray = cv2.cvtColor(image,cv2.COLOR_RGB2GRAY) # converting RGB colerd image to Gray scale image
    return gray

image = cv2.imread('test_image.jpg')
lane_image = np.copy(image)  # copying the image array to lane_image

gray=garyScale(lane_image)

cv2.imshow('Gray Image',gray)

cv2.waitKey(0)
cv2.destroyAllWindows()

# plt.imshow(cannied)
# plt.show()


