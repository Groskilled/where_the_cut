import cv2
import sys
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import math

def display_img(img):
    imgplot = plt.imshow(img)
    plt.show()

def inverte(img):
    return (255-img)

def print_cuts(cuts):
    for i in cuts:
        print "I found a cut between frame {0} and frame {1}.".format(i[0], i[1])

def HD(img1, img2):
    bhist1 = cv2.calcHist([img1],[0],None,[256],[0,256])
    bhist2 = cv2.calcHist([img2],[0],None,[256],[0,256])
    ghist1 = cv2.calcHist([img1],[1],None,[256],[0,256])
    ghist2 = cv2.calcHist([img1],[1],None,[256],[0,256])
    rhist1 = cv2.calcHist([img2],[2],None,[256],[0,256])
    rhist2 = cv2.calcHist([img2],[2],None,[256],[0,256])
    tmp1 = np.power(bhist2 - bhist1, 2)
    tmp2 = np.power(ghist2 - ghist1, 2)
    tmp3 = np.power(rhist2 - rhist1, 2)
    ret = math.sqrt(np.sum(tmp1 + tmp2 + tmp3))
    return ret

def ECR(img1, img2):
    kernel = np.ones((6,6),'uint8')
    ep_img1 = np.sum(img1) #edge pixels in img1
    ep_img2 = np.sum(img2) #edge pixels in img2
    if ep_img1 == 0:
        ep_img1 = 1
    if ep_img2 == 0:
        ep_img2 = 1
    dil_img1 = cv2.dilate(img1,kernel)
    dil_img2 = cv2.dilate(img2,kernel)
    inv_img1 = inverte(dil_img1) #img1 inverted
    inv_img2 = inverte(dil_img2) #img2 inverted
    pout = np.logical_and(img1, inv_img2)
    pin = np.logical_and(img2, inv_img1)
    pout_cnt = np.sum(pout)
    pin_cnt = np.sum(pin)
    return max(pout_cnt / ep_img1, pin_cnt / ep_img2)

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
    img1 = mpimg.imread(imgs[i])
    edges1 = cv2.Canny(RGB_to_gray(img1),50,150)
    img2 = mpimg.imread(imgs[i+1])
    edges2 = cv2.Canny(RGB_to_gray(img2),50,150)
    if ECR(edges1, edges2) > 0.003:
        if len(cuts) == 0:
            cuts.append([i, i+1])
        elif cuts[-1][1] < i - 3:
            cuts.append([i, i+1])
        else:
            cuts[-1][1] = i + 1
print_cuts(cuts)
