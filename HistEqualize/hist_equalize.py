import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib

# Read image
image = cv2.imread('jisoo.jpg', cv2.IMREAD_GRAYSCALE)

# Original hist
flatten_image = image.reshape(1,-1)
image_hist = np.histogram(flatten_image, 256, [0,256])
plt.bar(np.arange(0,256,1), image_hist[0])
plt.title('Original hist')
plt.show()

# Equalize hist
num_pixels = flatten_image.shape[1]
cdf = np.cumsum(image_hist[0])
h = np.round((cdf - np.min(cdf))*255/(num_pixels)).astype(np.uint8)
new_image = image.copy()
for i in range(len(h)):
    new_image[image == i] = h[i]

# Equalized hist
flatten_new_image = new_image.reshape(1,-1)
new_image_hist = np.histogram(flatten_new_image, 256, [0,256])
plt.bar(np.arange(0,256,1), new_image_hist[0])
plt.title('Equalized hist')
plt.show()

cv2.imshow('Original image', image)
cv2.waitKey(0)
cv2.imshow('Hist-Equalized image', new_image)
cv2.waitKey(0)