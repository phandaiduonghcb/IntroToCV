import cv2
import numpy as np

def getEdges(im):

    kernel_H = np.array([
    [[0.125,0.25,0.125],
    [0,0,0],
    [-0.125,-0.25,-0.125]]
    ])

    kernel_V = np.array([[0.125,0,-0.125],
                    [0.25,0,-0.25],
                    [0.125,0,-0.125]])

    edge_H = cv2.filter2D(im, ddepth=-1, kernel=kernel_H.T)
    edge_V = cv2.filter2D(im, ddepth=-1, kernel=kernel_V.T)
    return np.sqrt(edge_H**2 + edge_V**2)

def findCostArr(im):
    gray_im = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    edgeImg = getEdges(gray_im)
    r,c = edgeImg.shape
    cost = np.zeros(edgeImg.shape)
    cost[r-1,:] = edgeImg[r-1,:]
    
    for i in range(r-2,-1,-1):
        
        for j in range(c):
            c1,c2 = max(j-1,0),min(c,j+2)
            cost[i][j] = edgeImg[i][j] + cost[i+1,c1:c2].min()
                
    return cost

def get_a_seam_mask(cost):
  paddedCost = np.pad(cost, pad_width=1, mode='constant', constant_values=float('inf'))
  seam_idxs = [np.argmin(paddedCost[1])]
  mask = np.zeros_like(cost,dtype=np.bool8)
  mask[0][seam_idxs[-1] - 1] = True

  for i in range(2, paddedCost.shape[0]-1):
    s = np.argmin(paddedCost[i][seam_idxs[-1]-1:seam_idxs[-1]+2])
    seam_idxs.append(seam_idxs[-1] + s - 1)
    mask[i-1][seam_idxs[-1] - 1] = True
  return mask

def remove_vertical_seams(src_im, new_w):
  new_im = src_im.copy()
  h, w, c = new_im.shape
  assert 1 <new_w <= w
  
  cost = findCostArr(new_im)
  for _ in range(w - new_w):
    mask = get_a_seam_mask(cost)
    new_im = new_im[np.logical_not(mask)].reshape(h,-1, c)
    cost = cost[np.logical_not(mask)].reshape(h,-1)
  return new_im

def remove_horizontal_seams(src_im, new_h):
  new_im = cv2.rotate(src_im, cv2.ROTATE_90_COUNTERCLOCKWISE)
  new_im = remove_vertical_seams(new_im, new_h)
  new_im = cv2.rotate(new_im, cv2.ROTATE_90_CLOCKWISE)
  return new_im

def remove_seams(src_im, new_h, new_w):
  new_im = remove_vertical_seams(src_im, new_w)
  new_im = remove_horizontal_seams(new_im, new_h)
  return new_im

if __name__ == "__main__":

    src_im = cv2.imread('image.png')
    print(src_im.shape)
    new_im = remove_seams(src_im, 599, 730)
    print(new_im.shape)
    cv2.imshow('a',new_im)
    cv2.waitKey(0)

