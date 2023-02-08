from cv2 import threshold
import numpy as np
import cv2

def non_max_suppression(boxes, overlapThresh):
	'''Source: https://pyimagesearch.com/2015/02/16/faster-non-maximum-suppression-python/'''
	if len(boxes) == 0:
		return []
	if boxes.dtype.kind == "i":
		boxes = boxes.astype("float")
	pick = []
	x1 = boxes[:,0]
	y1 = boxes[:,1]
	x2 = boxes[:,2]
	y2 = boxes[:,3]
	# compute the area of the bounding boxes and sort the bounding
	# boxes by the bottom-right y-coordinate of the bounding box
	area = (x2 - x1 + 1) * (y2 - y1 + 1)
	idxs = np.argsort(y2)
	# keep looping while some indexes still remain in the indexes
	# list
	while len(idxs) > 0:
		# grab the last index in the indexes list and add the
		# index value to the list of picked indexes
		last = len(idxs) - 1
		i = idxs[last]
		pick.append(i)
		# find the largest (x, y) coordinates for the start of
		# the bounding box and the smallest (x, y) coordinates
		# for the end of the bounding box
		xx1 = np.maximum(x1[i], x1[idxs[:last]])
		yy1 = np.maximum(y1[i], y1[idxs[:last]])
		xx2 = np.minimum(x2[i], x2[idxs[:last]])
		yy2 = np.minimum(y2[i], y2[idxs[:last]])
		# compute the width and height of the bounding box
		w = np.maximum(0, xx2 - xx1 + 1)
		h = np.maximum(0, yy2 - yy1 + 1)
		# compute the ratio of overlap
		overlap = (w * h) / area[idxs[:last]]
		# delete all indexes from the index list that have
		idxs = np.delete(idxs, np.concatenate(([last],
			np.where(overlap > overlapThresh)[0])))
	# return only the bounding boxes that were picked using the
	# integer data type
	return boxes[pick].astype("int")

def normalize(data):
	return (data - np.min(data)) / (np.max(data) - np.min(data))
def matchTemplate(src_im, template, threshold=0.99, normalized=False):
	'''Template matching using normalized cross-correlation'''
	h, w = src_im.shape
	template_h, template_w = template.shape
	if template_h %2==0: template = np.concatenate((template,np.array([[0]*template.shape[1]])), axis=0)
	if template_w %2==0: template = np.concatenate((template,np.array([[0]*template.shape[0]]).T), axis=1)

	template_h, template_w = template.shape
	pad_size = max(template_w//2, template_h//2)
	padded_im = np.pad(src_im, pad_size, 'reflect')
	new_im = []
	x = template_w//2
	y = template_h//2
	if normalized:
		padded_im = normalize(padded_im)
		template = normalize(template)

	n=1
	for i in range(pad_size,h + pad_size):
		new_row = []
		for j in range(pad_size, w + pad_size):
			sub_im = padded_im[i-y:i+y+1,j-x:j+x+1]
			c = np.sum(sub_im*(template))
			if normalized:
				n = np.sqrt(np.sum(np.square(sub_im)))*np.sqrt(np.sum(np.square(template)))
			if n==0:
				new_row.append(0)
			else:
				new_row.append(c/n)

		new_im.append(new_row)
	
	res = np.array(new_im).astype(np.float64)
	res = normalize(res)
	loc = np.where(res >= threshold)

	recs = []
	for n, m in zip(loc[0], loc[1]):
		recs.append([m - x, n - y,m + x, n + y])
	return res, np.array(recs)

# --- Read source image and template ---
src_im = cv2.imread('9-ro.jpeg')
im = cv2.cvtColor(src_im, cv2.COLOR_BGR2GRAY)

template = cv2.imread('template.png')
template = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)


# Cross correlation
cc, _= matchTemplate(im, template,threshold=0.95, normalized=False)



# --- cv2 template matching ---
res_cv2 = cv2.matchTemplate(im, template,method=cv2.TM_CCORR_NORMED)
loc = np.where(res_cv2 >= 0.95)

im_cv2 = src_im.copy()
recs_cv2 = []
for y, x in zip(loc[0], loc[1]):
	recs_cv2.append([x,y,x + template.shape[1], y + template.shape[0]])
recs_cv2 = np.array(recs_cv2)
recs_cv2 = non_max_suppression(recs_cv2, overlapThresh=0.7)
for rec in recs_cv2:
	cv2.rectangle(im_cv2, (rec[0], rec[1]), (rec[2], rec[3]), (0,0,255), 1)


# --- self implemented template matching ---
res_ncc, recs_ncc = matchTemplate(im, template, threshold=0.95, normalized=True)
recs_ncc = non_max_suppression(recs_ncc, overlapThresh=0.8)
im_ncc = src_im.copy()
print("Num recs:", len(recs_ncc))
for rec in recs_ncc:
	cv2.rectangle(im_ncc, (rec[0], rec[1]), (rec[2], rec[3]),(0,0,255), 1)


# --- Show results ---
cv2.imshow('Cross correlation',cc)
cv2.waitKey(0)
cv2.imshow('cv2 TM',im_cv2)
cv2.waitKey(0)
cv2.imshow('My TM',im_ncc)
cv2.waitKey(0)
cv2.destroyAllWindows()