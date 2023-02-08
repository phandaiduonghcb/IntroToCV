import cv2
import numpy as np
import dlib
import time
from facial_landmarks import get_regular_landmark_points, get_dense_landmark_points, get_mediapipe_landmarks, get_dlib_landmark_points
from swap import swap_with_keypoints

def get_triangles_and_idxs(points):
    rect = cv2.boundingRect(convexhull)
    subdiv = cv2.Subdiv2D(rect)
    subdiv.insert(landmarks_points)
    triangles = subdiv.getTriangleList()
    triangles = np.array(triangles, dtype=np.int32)

    indexes_triangles = []
    for t in triangles:
        pt1 = (t[0], t[1])
        pt2 = (t[2], t[3])
        pt3 = (t[4], t[5])

        # cv2.line(img, pt1, pt2, (0,0,255),2)
        # cv2.line(img, pt2, pt3, (0,0,255),2)
        # cv2.line(img, pt1, pt3, (0,0,255),2)


        index_pt1 = np.where((points == pt1).all(axis=1))
        index_pt1 = extract_index_nparray(index_pt1)

        index_pt2 = np.where((points == pt2).all(axis=1))
        index_pt2 = extract_index_nparray(index_pt2)

        index_pt3 = np.where((points == pt3).all(axis=1))
        index_pt3 = extract_index_nparray(index_pt3)
        
        if index_pt1 is not None and index_pt2 is not None and index_pt3 is not None:
            triangle = [index_pt1, index_pt2, index_pt3]
            indexes_triangles.append(triangle)

    return triangles, indexes_triangles


def extract_index_nparray(nparray):
    index = None
    for num in nparray[0]:
        index = num
        break
    return index

image_path = "/ssd/Precision-Image-Face-Swapping-With-OpenCV/jennie.jpg"
image_path2 = "/ssd/Precision-Image-Face-Swapping-With-OpenCV/jisoo.jpg"

img = cv2.imread(image_path)
img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
mask = np.zeros_like(img_gray)
img2 = cv2.imread(image_path2)
img2_gray = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)


landmarks_points, teeth_points = get_dlib_landmark_points(img, get_teeth_points=True)
landmarks_points2, teeth_points2 = get_dlib_landmark_points(img2, get_teeth_points=True)

# landmarks_points = get_mediapipe_landmarks(img, False)
# landmarks_points2 = get_mediapipe_landmarks(img2, False)

# landmarks_points, teeth_points = get_regular_landmark_points(image_path, 2, use_base64=False, get_teeth_points=True)
# landmarks_points2, teeth_points2 = get_regular_landmark_points(image_path2, 2, use_base64=False, get_teeth_points=True)

# landmarks_points, teeth_points = get_dense_landmark_points(image_path,False,False,True)
# landmarks_points2, teeth_points2 = get_dense_landmark_points(image_path2,False,False,True)

seamlessclone = swap_with_keypoints(img, img2, landmarks_points, landmarks_points2)

mask = np.zeros((img2.shape[:2]), np.uint8)
convexhull = cv2.convexHull(np.array(teeth_points2))
cv2.fillConvexPoly(mask, np.array(convexhull), 255)
inv_mask = cv2.bitwise_not(mask)
bg = cv2.bitwise_and(seamlessclone, seamlessclone, mask=inv_mask)
fg = cv2.bitwise_and(img2, img2, mask=mask)
seamlessclone= cv2.add(bg,fg)

cv2.imshow("seamlessclone", seamlessclone)
cv2.waitKey(0)



cv2.destroyAllWindows()
