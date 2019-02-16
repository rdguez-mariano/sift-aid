import cv2
import sys
sys.path.append(".")
from libLocalDesc import *
from acc_test_library import *

from matplotlib import pyplot as plt
plt.switch_backend('agg')

CosProxThres = 0.4
SignAlingThres = 4000



img1 = cv2.cvtColor(cv2.imread('./acc-test/notredame.1.png'),cv2.COLOR_BGR2GRAY) # queryImage
img2 = cv2.cvtColor(cv2.imread('./acc-test/notredame.2.png'),cv2.COLOR_BGR2GRAY) # trainImage

total, good_HC, ET_KP, ET_M = siftAID(img1,img2, MatchingThres = SignAlingThres, Simi = 'SignProx', Visual=True)


asift_KPlist1, patches1, GT_Avec_list, asift_KPlist2, patches2 = load_acc_test_data('./acc-test/notredame')
good2_asift = [cv2.DMatch(i, i, 0.9) for i in range(0,len(asift_KPlist1))]
img3 = cv2.drawMatches(img1,asift_KPlist1,img2,asift_KPlist2,good2_asift, None,flags=2)
cv2.imwrite('./temp/ARootSIFT_homography_matches.png',img3)