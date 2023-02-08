import cv2
import numpy as np
import matplotlib.pyplot as plt
import math
def hough(img, rho=1, theta=np.pi/180, threshold=100):
    img_height, img_width = img.shape[:2]
    diagonal_length = int(math.sqrt(img_height*img_height + img_width*img_width))
    num_rho = int(diagonal_length / rho)
    num_theta = int(np.pi / theta)

    thetas = np.arange(0,np.pi,theta)
    num_theta = len(thetas)
    
    sin_thetas = np.sin(thetas)
    cos_thetas = np.cos(thetas)

    edge_points = np.argwhere(img!=0)
    all_rhos_values = np.matmul(edge_points, np.array([sin_thetas,cos_thetas]))
    A = np.zeros((num_rho*2 + 1, num_theta))
    for i in range(len(all_rhos_values)):
        for j,rho_value in enumerate(all_rhos_values[i]):
            A[int(round(rho_value))+num_rho, j] += 1
    # line_idx = np.where(edge_matrix > threshold)
    idxs = np.where(A > threshold)
    rho_values = idxs[0] - num_rho
    theta_values = np.deg2rad(idxs[1])
    return rho_values, theta_values
if __name__ == '__main__':
    im = cv2.imread('geometry.jpg')
    edges = cv2.Canny(im,100,100)
    rho_values, theta_values = hough(edges, rho=1, theta=np.pi/180, threshold=100)
    for rho, theta in zip(rho_values, theta_values):
        a = np.cos(theta)
        b = np.sin(theta)
        x0 = a*rho
        y0 = b*rho
        x1 = int(x0 + 1000*(-b))
        y1 = int(y0 + 1000*(a))
        x2 = int(x0 - 1000*(-b))
        y2 = int(y0 - 1000*(a))
        cv2.line(im,(x1,y1),(x2,y2),(0,0,255),2)
    # print(detected_lines.shape)
    cv2.imshow('line detection', im)
    cv2.waitKey(0)
