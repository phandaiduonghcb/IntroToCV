import cv2
import numpy as np

src_im = cv2.imread('cells.png')
src_im = cv2.cvtColor(src_im, cv2.COLOR_BGR2GRAY)
cv2.imshow('morphology', src_im)
cv2.waitKey(0)

im = cv2.adaptiveThreshold(src_im,255,cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 11, 2)
ret, im = cv2.threshold(src_im,127,255,cv2.THRESH_BINARY_INV)
cv2.imshow('morphology', im)
cv2.waitKey(0)

kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (10,10))
erosion = cv2.erode(im,kernel,iterations = 3)
dilation = cv2.dilate(erosion,kernel,iterations = 3)
cv2.imshow('morphology', dilation)
cv2.waitKey(0)

#Connected-component labeling

def get_neighbor_idxs(cur_idx,  connect_4=True):
  i,j = cur_idx
  arr = [[i-1,j], [i+1,j], [i,j-1], [i,j+1]]
  if not connect_4:
    arr += [[i-1,j+1],[i+1,j-1],[i+1,j+1],[i-1,j-1]]
  return arr

idxs =  np.argwhere(dilation != 0)
labeled_mask = dilation == 0
label_array = np.zeros_like(dilation) - 1
label = 0
for idx in idxs:
  i,j = idx[0], idx[1]
  if not labeled_mask[i, j]:
    queue = [(i,j)]
    while queue:
      cur_i, cur_j = queue.pop()
      if cur_i < 0 or cur_i >= dilation.shape[0] or cur_j < 0 or cur_j >= dilation.shape[1] or labeled_mask[cur_i, cur_j]:
        continue
      labeled_mask[cur_i, cur_j] = True
      label_array[cur_i, cur_j] = label
      neighbor_idxs = get_neighbor_idxs((cur_i,cur_j))
      queue += neighbor_idxs
    label +=1

print('Number of cells:',label)
print('Total area:', (dilation != 0).astype(np.uint8).sum())

chosen_label = 50
test = np.zeros_like(dilation, np.uint8)
test[label_array == chosen_label] = 255
cv2.imshow('morphology', test)
cv2.waitKey(0)