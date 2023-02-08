import requests
import base64
import ast
import cv2
import math
import cv2
import mediapipe as mp
import dlib

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("./shape_predictor_68_face_landmarks.dat")

KEY = '69q9Ffp7D5zOteORNqrju6SdTkcDy6uk'
SECRET = '5KEafupRXViEiio2vBA3m43QkVv9nt9M'

_PRESENCE_THRESHOLD = 0.5
_VISIBILITY_THRESHOLD = 0.5

FACEMESH_LIPS = [(61, 146), (146, 91), (91, 181), (181, 84), (84, 17),
                           (17, 314), (314, 405), (405, 321), (321, 375),
                           (375, 291), (61, 185), (185, 40), (40, 39), (39, 37),
                           (37, 0), (0, 267),
                           (267, 269), (269, 270), (270, 409), (409, 291),
                           (78, 95), (95, 88), (88, 178), (178, 87), (87, 14),
                           (14, 317), (317, 402), (402, 318), (318, 324),
                           (324, 308), (78, 191), (191, 80), (80, 81), (81, 82),
                           (82, 13), (13, 312), (312, 311), (311, 310),
                           (310, 415), (415, 308)]
LIPS = [t[0] for t in FACEMESH_LIPS] + [FACEMESH_LIPS[-1][1]]
LIPS = set(LIPS)
LIPS = list(LIPS)

mp_face_mesh = mp.solutions.face_mesh

def _normalized_to_pixel_coordinates(
    normalized_x: float, normalized_y: float, image_width: int,
    image_height: int):
  """Converts normalized value pair to pixel coordinates."""

  # Checks if the float value is between 0 and 1.
  def is_valid_normalized_value(value: float) -> bool:
    return (value > 0 or math.isclose(0, value)) and (value < 1 or
                                                      math.isclose(1, value))

  if not (is_valid_normalized_value(normalized_x) and
          is_valid_normalized_value(normalized_y)):
    # TODO: Draw coordinates even if it's outside of the image bounds.
    return None
  x_px = min(math.floor(normalized_x * image_width), image_width - 1)
  y_px = min(math.floor(normalized_y * image_height), image_height - 1)
  return x_px, y_px

def get_mediapipe_landmarks(bgr_image, get_lips_points=False):
    image_rows, image_cols, _ = bgr_image.shape
    with mp_face_mesh.FaceMesh(
        static_image_mode=True,
        max_num_faces=1,
        refine_landmarks=True,
        min_detection_confidence=0.5) as face_mesh:

        results = face_mesh.process(cv2.cvtColor(bgr_image, cv2.COLOR_BGR2RGB))
        points = []
        lips_points = []
        for face_landmarks in results.multi_face_landmarks:
            idx_to_coordinates = {}
            for idx, landmark in enumerate(face_landmarks.landmark):
                if ((landmark.HasField('visibility') and
                    landmark.visibility < _VISIBILITY_THRESHOLD) or
                    (landmark.HasField('presence') and
                    landmark.presence < _PRESENCE_THRESHOLD)):
                    continue
                landmark_px = _normalized_to_pixel_coordinates(landmark.x, landmark.y,
                                                            image_cols, image_rows)
                if landmark_px:
                    # idx_to_coordinates[idx] = landmark_px
                    if get_lips_points and idx in LIPS:
                        lips_points.append(landmark_px)
                    points.append(landmark_px)

        if get_lips_points:
            return points, lips_points
        return points

def get_dense_landmark_points(image_path, use_base64=False, get_lips_points=False, get_teeth_points=False):
    url = 'https://api-us.faceplusplus.com/facepp/v1/face/thousandlandmark'
    im_b64 = image_path
    lips_points = {}
    lower_lip = []
    upper_lip = []
    teeth_points = []
    if not use_base64:  
        with open(image_path, "rb") as f:
            im_bytes = f.read()        
        im_b64 = base64.b64encode(im_bytes).decode("utf8")
    params = {
        'api_key': KEY,
        'api_secret': SECRET,
        'image_base64':im_b64,
        # 'image_url' : 'https://znews-photo.zingcdn.me/w660/Uploaded/qfssu/2022_11_08/10_BLACKPINK_Jisoo_DIOR_Pop_Up_Store_Event_19_August_2019_1.jpg',
        'return_landmark': 'all',
        }

    x = requests.post(url, data=params)
    if x.status_code != 200:
        raise Exception(str(x.status_code) + ' ' + x.reason)

    byte_str = x.content
    dict_str = byte_str.decode("UTF-8")
    mydata = ast.literal_eval(dict_str)
    landmarks = mydata['face']['landmark']
    points = []
    landmarks_keys = list(landmarks.keys())
    landmarks_keys.sort()
    for key_0 in landmarks_keys:
        keypoints = landmarks[key_0]
        keypoints_keys = list(keypoints.keys())
        keypoints_keys.sort()
        for key in keypoints_keys:
            if key.startswith('face_hair'):
                continue
            if isinstance(keypoints[key],dict):
                x = keypoints[key]['x']
                y = keypoints[key]['y']
                points.append((x,y))
                if (key.startswith('upper_lip') or key.startswith('lower_lip')) and int(key.split('_')[-1]) >= 32:
                    teeth_points.append((x,y))
                if key.startswith('lower_lip'):
                    lower_lip.append((x,y))
                if key.startswith('upper_lip'):
                    upper_lip.append((x,y))
        # image = cv2.circle(image, (x,y), radius=2, color=(0, 0, 255), thickness=-1)
    lips_points['upper_lip'] = upper_lip
    lips_points['lower_lip'] = lower_lip

    result = [points]
    if get_lips_points:
        result.append(lips_points)
    if get_teeth_points:
        result.append(teeth_points)
    return tuple(result)
def get_dlib_landmark_points(image, get_teeth_points=False):
    img_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faces = detector(img_gray)
    for face in faces:
        landmarks = predictor(img_gray, face)
        landmarks_points = []
        for n in range(0, 68):
            x = landmarks.part(n).x
            y = landmarks.part(n).y
            # img = cv2.circle(img, (x,y), radius=2, color=(0, 0, 255), thickness=-1)
            landmarks_points.append((x, y))
    if get_teeth_points:
        return landmarks_points, landmarks_points[60:]
    return landmarks_points
def get_regular_landmark_points(image_path, landmark=2, use_base64=False, get_teeth_points=False):
    url = 'https://api-us.faceplusplus.com/facepp/v3/detect'
    im_b64 = image_path
    if not use_base64:
        with open(image_path, "rb") as f:
            im_bytes = f.read()        
        im_b64 = base64.b64encode(im_bytes).decode("utf8")
    params = {
        'api_key': KEY,
        'api_secret': SECRET,
        'image_base64':im_b64,
        'return_landmark': landmark,
        }

    x = requests.post(url, data=params)
    if x.status_code != 200:
        raise Exception(str(x.status_code) + ' ' + x.reason)
    byte_str = x.content
    dict_str = byte_str.decode("UTF-8")
    mydata = ast.literal_eval(dict_str)

    # image = cv2.imread(image_path)
    points = []
    teeth_points =[]
    for face in mydata['faces']:
        landmarks = face['landmark']
        keys = list(landmarks.keys())
        keys.sort()
        for key_0 in keys:
            chosen = False
            if (((key_0.startswith('mouth_lower_lip_left') or key_0.startswith('mouth_lower_lip_right')) and (int(key_0[-1]) == 1 or int(key_0[-1]) == 4)) or key_0.startswith('mouth_lower_lip_top') or key_0.startswith('mouth_left') or key_0.startswith('mouth_right')):
                chosen = True
            if (((key_0.startswith('mouth_upper_lip_left') or key_0.startswith('mouth_upper_lip_right')) and (int(key_0[-1]) == 3 or int(key_0[-1]) == 4)) or key_0.startswith('mouth_upper_lip_bottom') or key_0.startswith('mouth_left') or key_0.startswith('mouth_right')):
                chosen = True
            keypoints = landmarks[key_0]
            x = keypoints['x']
            y = keypoints['y']
            points.append((x,y))
            if chosen:
                teeth_points.append((x,y))
            # image = cv2.circle(image, (x,y), radius=2, color=(0, 0, 255), thickness=-1)
    # cv2.imshow('a',image)
    # cv2.waitKey(0)
    if get_teeth_points:
        return (points, teeth_points)
    return points
if __name__ == '__main__':
    # image_path = 'bradley_cooper.jpg'
    image_path = '/ssd/Precision-Image-Face-Swapping-With-OpenCV/jisoo.jpg'
    points, teeth_points = get_dlib_landmark_points(cv2.imread(image_path), True)
    # points, lips_points, teeth_points = get_dense_landmark_points(image_path, get_lips_points=True, get_teeth_points=True)
    # points, teeth_points = get_regular_landmark_points(image_path, 2, use_base64=False, get_teeth_points=True)
    points = teeth_points
    # points = get_mediapipe_landmarks(cv2.imread(image_path))

    image = cv2.imread(image_path)
    for i,point in  enumerate(points):
        x,y=point
        image = cv2.circle(image, (x,y), radius=2, color=(0, 0, 255), thickness=-1)
    cv2.imshow('a',image)
    cv2.imwrite('test.jpg',image)
    cv2.waitKey(0)