import cv2
import numpy as np
import imutils
def show_frame(frame, waitKey=1):
    cv2.imshow('Effect',frame)
    cv2.waitKey(waitKey)
def push_left(im_1_path, im_2_path, step=1):
    im_1 = cv2.imread(im_1_path)
    im_2 = cv2.imread(im_2_path)

    # im_1 = imutils.resize(im_1, height=SIZE)
    # im_2 = imutils.resize(im_2, height=SIZE)
    im_1 = cv2.resize(im_1, (SIZE,SIZE))
    im_2 = cv2.resize(im_2, (SIZE,SIZE))

    im_h = cv2.hconcat([im_1, im_2])

    size_1 = im_1.shape
    size_2 = im_2.shape

    cv2.namedWindow('Effect', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('Effect',height=SIZE,width=SIZE)

    for i in range(size_1[1],size_1[1] + size_2[1], step):
        frame = im_h[:,i-size_1[1]:i]
        show_frame(frame)

    cv2.destroyAllWindows()

def push_right(im_1_path, im_2_path, step=1):
    im_1 = cv2.imread(im_1_path)
    im_2 = cv2.imread(im_2_path)

    # im_1 = imutils.resize(im_1, height=SIZE)
    # im_2 = imutils.resize(im_2, height=SIZE)
    im_1 = cv2.resize(im_1, (SIZE,SIZE))
    im_2 = cv2.resize(im_2, (SIZE,SIZE))

    im_h = cv2.hconcat([im_2, im_1])

    size_1 = im_1.shape
    size_2 = im_2.shape
    size_im_h = im_h.shape

    cv2.namedWindow('Effect', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('Effect',height=SIZE,width=SIZE)

    for i in range(size_im_h[1],size_im_h[1] - size_1[1], -step):
        frame = im_h[:,i-size_1[1]:i]
        show_frame(frame)

    cv2.destroyAllWindows()

def push_up(im_1_path, im_2_path, step=1):
    im_1 = cv2.imread(im_1_path)
    im_2 = cv2.imread(im_2_path)

    # im_1 = imutils.resize(im_1, width=SIZE)
    # im_2 = imutils.resize(im_2, width=SIZE)
    im_1 = cv2.resize(im_1, (SIZE,SIZE))
    im_2 = cv2.resize(im_2, (SIZE,SIZE))

    im_v = cv2.vconcat([im_1, im_2])

    size_1 = im_1.shape
    size_2 = im_2.shape

    cv2.namedWindow('Effect', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('Effect',height=SIZE,width=SIZE)

    for i in range(size_1[0],size_1[0] + size_2[0], step):
        frame = im_v[i-size_1[0]:i,:]
        show_frame(frame)
    cv2.destroyAllWindows()

def push_down(im_1_path, im_2_path, step=1):
    im_1 = cv2.imread(im_1_path)
    im_2 = cv2.imread(im_2_path)

    # im_1 = imutils.resize(im_1, width=SIZE)
    # im_2 = imutils.resize(im_2, width=SIZE)
    im_1 = cv2.resize(im_1, (SIZE,SIZE))
    im_2 = cv2.resize(im_2, (SIZE,SIZE))

    im_v = cv2.vconcat([im_2, im_1])

    size_1 = im_1.shape
    size_2 = im_2.shape
    size_im_v = im_v.shape

    cv2.namedWindow('Effect', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('Effect',height=SIZE,width=SIZE)

    for i in range(size_im_v[0],size_1[0], -step):
        frame = im_v[i - size_1[0]: i,:]
        show_frame(frame)
    cv2.destroyAllWindows()

def cover(im_1_path, im_2_path, step=1):
    im_1 = cv2.imread(im_1_path)
    im_2 = cv2.imread(im_2_path)

    # im_1 = imutils.resize(im_1, width=SIZE)
    # im_2 = imutils.resize(im_2, width=SIZE)
    im_1 = cv2.resize(im_1, (SIZE,SIZE))
    im_2 = cv2.resize(im_2, (SIZE,SIZE))

    size_1 = im_1.shape
    size_2 = im_2.shape

    cv2.namedWindow('Effect', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('Effect',height=SIZE,width=SIZE)

    for i in range(size_2[1],0,-1):
        frame = cv2.hconcat([im_1[:,:i],im_2[:,:size_1[1] - i+1]])
        show_frame(frame)
    cv2.destroyAllWindows()

def uncover(im_1_path, im_2_path, step=1):
    im_1 = cv2.imread(im_1_path)
    im_2 = cv2.imread(im_2_path)

    # im_1 = imutils.resize(im_1, width=SIZE)
    # im_2 = imutils.resize(im_2, width=SIZE)
    im_1 = cv2.resize(im_1, (SIZE,SIZE))
    im_2 = cv2.resize(im_2, (SIZE,SIZE))

    size_1 = im_1.shape
    size_2 = im_2.shape

    cv2.namedWindow('Effect', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('Effect',height=SIZE,width=SIZE)

    for i in range(size_2[1]):
        frame = cv2.hconcat([im_1[:,i:],im_2[:,size_2[1]-i:]])
        show_frame(frame)
    cv2.destroyAllWindows()
def fade(im_1_path, im_2_path):
    im_1 = cv2.imread(im_1_path)
    im_2 = cv2.imread(im_2_path)

    # im_1 = imutils.resize(im_1, width=SIZE)
    # im_2 = imutils.resize(im_2, width=SIZE)
    im_1 = cv2.resize(im_1, (SIZE,SIZE))
    im_2 = cv2.resize(im_2, (SIZE,SIZE))

    cv2.namedWindow('Effect', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('Effect',height=SIZE,width=SIZE)

    for i in range(0,1001,2):
        a = i/1000
        b = 1 - a
        frame = cv2.addWeighted(im_1,a,im_2,b,0.0)
        show_frame(frame)
    cv2.destroyAllWindows()

def go_left_and_down(im_1_path,im_2_path):
    im_1 = cv2.imread(im_1_path)
    im_2 = cv2.imread(im_2_path)

    # im_1 = imutils.resize(im_1, width=SIZE)
    # im_2 = imutils.resize(im_2, width=SIZE)
    im_1 = cv2.resize(im_1, (512,640))
    im_2 = cv2.resize(im_2, (512,640))

    size_1 = im_1.shape
    size_2 = im_2.shape

    cv2.namedWindow('Effect', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('Effect',height=SIZE,width=SIZE)

    for i in range(1,500):
        frame = im_1.copy()
        im_2_width = round(size_2[1] * i / 500)
        im_2_height = round(size_2[0] * i / 500)
        frame[:im_2_height,size_1[1] - im_2_width:] = im_2[size_2[0] - im_2_height:,:im_2_width]
        show_frame(frame)
    cv2.destroyAllWindows()
if __name__ == '__main__':
    SIZE = 640
    go_left_and_down('jennie.jpg','jisoo.jpg')
    push_left('jennie.jpg','jisoo.jpg')
    push_right('jennie.jpg','jisoo.jpg')
    push_down('jennie.jpg','jisoo.jpg')
    push_up('jennie.jpg','jisoo.jpg')
    cover('jennie.jpg','jisoo.jpg')
    uncover('jennie.jpg','jisoo.jpg')
    fade('jennie.jpg','jisoo.jpg')