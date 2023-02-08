import cv2
from sklearn.linear_model import LogisticRegression
import numpy as np


background_im = cv2.imread('background.jpg')
people_im = cv2.imread('people.jpeg')

background_data = background_im.reshape(-1,3)
people_data = people_im.reshape(-1,3)
X = np.concatenate((background_data, people_data))
y = np.concatenate((np.ones((len(background_data),)), np.zeros((len(people_data),))))

reg = LogisticRegression()
reg.fit(X,y)


test_im = cv2.imread('female.jpg')
result_mask = []
for row in test_im:
    for p in row:
        result_mask.append(reg.predict([p])[0].astype(np.bool8))

result_mask  = np.array(result_mask).reshape(test_im.shape[:2])
test_im[result_mask] = [0,0,0]
cv2.imshow('result', test_im)
cv2.waitKey(0)
cv2.destroyAllWindows()