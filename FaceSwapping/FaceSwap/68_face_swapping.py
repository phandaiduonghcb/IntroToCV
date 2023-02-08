import cv2
import numpy as np
import dlib
import time
from facial_landmarks import get_regular_landmark_points, get_dense_landmark_points, get_mediapipe_landmarks, get_dlib_landmark_points
from swap import swap_with_keypoints
import base64


img = cv2.imread("./justin-bieber.jpg")
img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
# cap = cv2.VideoCapture('test.mp4')
cap = cv2.VideoCapture('cut.mp4')
input_size = (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
fourcc = cv2.VideoWriter_fourcc('X','V','I','D')
save_path = 'results/68.mp4'
print(input_size)
fps = cap.get(cv2.CAP_PROP_FPS)
print(f"{fps} frames per second")
if save_path is not None:
      videoWriter = cv2.VideoWriter(save_path, fourcc, fps , input_size)


landmarks_points, teeth_points = get_dlib_landmark_points(img, get_teeth_points=True)

# landmarks_points, lips = get_mediapipe_landmarks(img, get_lips_points=True)

# retval, buffer = cv2.imencode('.jpg', img)
# im_bytes = buffer.tobytes()
# im_b64 = base64.b64encode(im_bytes).decode('utf-8')
# # landmarks_points, teeth_points = get_regular_landmark_points(im_b64, 2,True, True)
# landmarks_points, lips_points, teeth_points = get_dense_landmark_points(im_b64,True,True, True)

while True:
    ret, img2 = cap.read()
    if not ret:
        break
    img2_gray = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

    landmarks_points2, teeth_points2 = get_dlib_landmark_points(img2, get_teeth_points=True)

    # retval, buffer = cv2.imencode('.jpg', img2)
    # im_bytes = buffer.tobytes()
    # im_b64 = base64.b64encode(im_bytes).decode('utf-8')
    # # landmarks_points2, teeth_points2 = get_regular_landmark_points(im_b64, 2,True, True)
    # landmarks_points2, lips_points2,teeth_points2 = get_dense_landmark_points(im_b64,True,True,True)
    # time.sleep(1)

    # landmarks_points2, lips2 = get_mediapipe_landmarks(img2, get_lips_points=True)
    new_image = swap_with_keypoints(img, img2, landmarks_points, landmarks_points2)

    mask = np.zeros((img2.shape[:2]), np.uint8)
    convexhull = cv2.convexHull(np.array(teeth_points2))
    cv2.fillConvexPoly(mask, np.array(convexhull), 255)
    inv_mask = cv2.bitwise_not(mask)
    bg = cv2.bitwise_and(new_image, new_image, mask=inv_mask)
    fg = cv2.bitwise_and(img2, img2, mask=mask)
    new_image= cv2.add(bg,fg)

    cv2.imshow("result", new_image)
    if save_path is not None:
        videoWriter.write(new_image)
        # videoWriter.write(img2)
    key = cv2.waitKey(1)
    if key == 27:
        break

cap.release()
if save_path is not None:
    videoWriter.release()
cv2.destroyAllWindows()
