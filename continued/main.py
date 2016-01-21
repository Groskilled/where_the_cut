import cv2
import sys
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import math
import os

def display_img(img):
    imgplot = plt.imshow(img)
    plt.show()

def inverte(img):
    return (255-img)

def print_cuts(cuts, imgs):
    for i in cuts:
        print "I found a cut between frame {0} and frame {1}.".format(i[0], i[1])
        fig = plt.figure()
        a = fig.add_subplot(1,2,2)
        plt.subplot(121)
        imgplot = plt.imshow(mpimg.imread(imgs[i[0]]))
        plt.subplot(122)
        imgplot = plt.imshow(mpimg.imread(imgs[i[1]]))
        plt.show()

def HD_1(img1, img2):
    hist1 = cv2.calcHist([img1], [0, 1, 2], None, [8, 8, 8], [0, 256, 0, 256, 0, 256])
    hist2 = cv2.calcHist([img2], [0, 1, 2], None, [8, 8, 8], [0, 256, 0, 256, 0, 256])
    hist1 = cv2.normalize(hist1).flatten()
    hist2 = cv2.normalize(hist1).flatten()
    return cv2.compareHist(hist1, hist2, cv2.cv.CV_COMP_CORREL)

def HD(imgs):
    hist = [cv2.calcHist([img], [0, 1, 2], None, [8, 8, 8], [0, 256, 0, 256, 0, 256]) for img in imgs]
    hist = [cv2.normalize(hist1).flatten() for hist1 in hist]
    return cv2.compareHist(hist[0], hist[-1], cv2.cv.CV_COMP_CORREL)

def check_cuts(cuts, img):
    ret = []
    for i in cuts:
        if (i[1] - i[0]) > 5:
            start = i[0]
            for j in range(i[0], i[1]):
                img1 = mpimg.imread("frame%d.jpg" %j)
                img2 = mpimg.imread("frame%d.jpg" %(j + 1))
                if HD_1(img1, img2) > 0.92:
                    start = j
                else:
                    i[0] = start
                    ret.append(i)
        else:
            ret.append(i)
    return ret

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

if len(sys.argv) < 2:
    print "Video path needed."
    sys.exit()
if len(sys.argv) == 3 and sys.argv[2] == "yes":
    exist = True
else:
    exist = False
vidcap = cv2.VideoCapture(sys.argv[1])
success = True 
count = 0;
imgs = []
cuts = []
while success:
    success,image = vidcap.read()
    if success == False:
        break
    if exist == False:
        cv2.imwrite("frame%d.jpg" % count, image)
    imgs.append("frame%d.jpg" %count)
    count += 1
for i in xrange(count - 1):
    img = [mpimg.imread(imgs[j]) for j in range(i, i+2)]
    edges = [cv2.Canny(cv2.cvtColor(k, cv2.COLOR_BGR2GRAY),50,150) for k in img]
    if ECR(edges) > 0.003 and HD(img) < 0.90:
        if len(cuts) == 0:
            cuts.append([i, i+1])
        elif cuts[-1][1] < i - 3:
            cuts.append([i, i+1])
        else:
            cuts[-1][1] = i + 1
cuts = check_cuts(cuts, imgs)
print_cuts(cuts, imgs)
