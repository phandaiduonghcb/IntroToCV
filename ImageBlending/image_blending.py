import cv2
import imageio
import numpy as np
def resize_and_keep_ratio(image, max_value):
    h, w = image.shape[:2]
    larger_dimension = max((h,w))
    t = max_value/larger_dimension
    return cv2.resize(image, (int(h*t), int(w*t)))
def padding_for_new_shape(image, new_shape):
    old_h, old_w, c = image.shape
    new_h, new_w = new_shape[:2]
    color = [0]*c
    result = np.full((new_h,new_w,c), color, dtype=np.uint8)
    y_center = (new_h - old_h) // 2
    x_center = (new_w - old_w) // 2
    result[y_center:y_center + old_h, x_center:x_center+old_w] = image
    return result
def get_mask_coordinates(mask):
    arr = (mask[:,:,3] != 0)
    top = None
    bot = None
    for i in range(len(arr)):
        if True in arr[i]:
            if top is None:
                top = i
            else:
                bot = i
    return top, bot
person = cv2.imread('ironman_original.png')
person =cv2.cvtColor(person, cv2.COLOR_BGR2RGB)
mask = cv2.imread('ironman.png', cv2.IMREAD_UNCHANGED)
top, bot = get_mask_coordinates(mask)
person = resize_and_keep_ratio(person, 350)
mask = resize_and_keep_ratio(mask, 350)
smoke_frames = imageio.mimread('smoke02.gif', '.gif')

person = padding_for_new_shape(person, smoke_frames[0].shape)
mask = padding_for_new_shape(mask, smoke_frames[0].shape)

results = []
for i in range(len(smoke_frames)):
    result = smoke_frames[i].copy()
    new_person = person.copy()
    new_person = cv2.cvtColor(new_person, cv2.COLOR_RGB2RGBA)

    for i in range(len(mask)):
        alpha = (i-top)/(bot-top)
        if alpha < 0 or alpha > 1:
            alpha = 1
        elif alpha < 0.2:
            alpha = 0
        elif alpha > 0.9:
            alpha = 0.9
        result[i][mask[i,:,3] != 0] = alpha * result[i][mask[i,:,3] != 0]
        new_person[i][mask[i,:,3] == 0] = 0
        new_person[i][mask[i,:,3] != 0] = (1-alpha)*new_person[i][mask[i,:,3] != 0]
        result[i] = result[i] + new_person[i]
    results.append(result)


imageio.mimsave('result.gif', results)

