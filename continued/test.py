import cv2
import sys
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import math
from scipy import misc
import os

def display_img(img):
    imgplot = plt.imshow(img)
    plt.show()

def inverte(img):
    return (255-img)

def HD(imgs):
    hist = [cv2.calcHist([img], [0, 1, 2], None, [8, 8, 8], [0, 256, 0, 256, 0, 256]) for img in imgs]
    hist = [cv2.normalize(hist1).flatten() for hist1 in hist]
    return cv2.compareHist(hist[0], hist[-1], cv2.cv.CV_COMP_CORREL)

def ECR(img):
    kernel = np.ones((6,6),'uint8')
    ep_img = [np.sum(im) for im in img]
    for i in xrange(len(ep_img)):
        if ep_img[i] == 0:
            ep_img[i] = 1
    dil_img = [cv2.dilate(im, kernel) for im in img]
    inv_img = [inverte(im) for im in dil_img]
    pout = np.logical_and(img[0], inv_img[-1])
    pin = np.logical_and(img[-1], inv_img[0])
    pout_cnt = np.sum(pout)
    pin_cnt = np.sum(pin)
    return max(pout_cnt / ep_img[0], pin_cnt / ep_img[-1])

def RGB_to_gray(img):
    return ((img[:,:,0]+img[:,:,1]+img[:,:,2])/3)

img = mpimg.imread("frame361.jpg")
resize = misc.imresize(img, 0.5)
edges1 = cv2.Canny(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY),50,150)
edges2 = cv2.Canny(cv2.cvtColor(resize, cv2.COLOR_BGR2GRAY),50,100)
'''
fig = plt.figure()
a = fig.add_subplot(1,2,2)
plt.subplot(121)
imgplot = plt.imshow(edges1)
plt.subplot(122)
imgplot = plt.imshow(edges2)
plt.show()
'''
print "normal: {0}, resized: {1}".format(np.sum(edges1), np.sum(edges2))
resize = misc.imresize(edges2, 2.0)
display_img(resize)
print np.sum(resize)
cv2.imwrite("resize.jpg", resize)
