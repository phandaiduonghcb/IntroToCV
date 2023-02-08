import numpy as np
import cv2
from scipy.ndimage import correlate

def compute_convolution2d(src_im, kernel):
    return compute_correlation2d(src_im, kernel[::-1,::-1])

def compute_correlation2d(src_im, kernel):
    h, w = src_im.shape
    kernel_h, kernel_w = kernel.shape
    if kernel_h %2==0: kernel = np.concatenate((kernel,np.array([[0]*kernel.shape[1]])), axis=0)
    if kernel_w %2==0: kernel = np.concatenate((kernel,np.array([[0]*kernel.shape[0]]).T), axis=1)

    kernel_h, kernel_w = kernel.shape
    pad_size = max(kernel_w//2, kernel_h//2)
    padded_im = np.pad(src_im, pad_size, 'reflect')
    new_im = []

    x = kernel_w//2
    y = kernel_h//2

    for i in range(pad_size,h + pad_size):
        new_row = []
        for j in range(pad_size, w + pad_size):
            sub_im = padded_im[i-y:i+y+1,j-x:j+x+1]
            c = np.sum(sub_im*(kernel))
            new_row.append(c)
        new_im.append(new_row)
    return np.array(new_im).astype(np.uint8)

im = cv2.imread('jisoo.jpg')
im = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)

kernel = np.ones((5, 5), np.float32) / 25


correlated_im = correlate(im, kernel)
correlated_im2 = compute_correlation2d(im, kernel=kernel)

cv2.imshow('original grayscale',im)
cv2.waitKey(0)
cv2.imshow('cv2 function',correlated_im)
cv2.waitKey(0)
cv2.imshow('correlation',correlated_im2)
cv2.waitKey(0)
cv2.destroyAllWindows()