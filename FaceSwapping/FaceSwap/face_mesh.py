import math
import cv2
import mediapipe as mp
_PRESENCE_THRESHOLD = 0.5
_VISIBILITY_THRESHOLD = 0.5
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

def get_mediapipe_landmarks(bgr_image):
    image_rows, image_cols, _ = bgr_image.shape
    with mp_face_mesh.FaceMesh(
        static_image_mode=True,
        max_num_faces=1,
        refine_landmarks=True,
        min_detection_confidence=0.5) as face_mesh:

        results = face_mesh.process(cv2.cvtColor(bgr_image, cv2.COLOR_BGR2RGB))
        points = []
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
                    points.append(landmark_px)
        return points
    
if __name__ == '__main__':
    image_path = '/ssd/Precision-Image-Face-Swapping-With-OpenCV/jim_carrey.jpg'
    im = cv2.imread(image_path)
    points = get_mediapipe_landmarks(im)
    for point in points:
        x,y = point
        cv2.circle(im, (int(x), int(y)), 2, (255,0,0),-1)
    cv2.imshow('a', im)
    cv2.waitKey(0)
