import numpy as np
import cv2
from matplotlib import pyplot as plt
from library import *
import time
from keras import layers
from keras.models import Model
import tensorflow as tf
from keras.backend.tensorflow_backend import set_session
config = tf.ConfigProto(allow_soft_placement=True)

from models import *
from keras.models import load_model
#, device_count = {'CPU' : 1, 'GPU' : 1})
config.gpu_options.per_process_gpu_memory_fraction = 0.1
set_session(tf.Session(config=config))


#  VGG like network
vgg_input_shape = tuple([60,60]) + tuple([1])
MODEL_NAME = 'AID_simCos_BigDesc_dropout'
weights2load = 'model-data/model.'+MODEL_NAME+'_75.hdf5'
train_model, sim_type = create_model(vgg_input_shape, None, model_name = MODEL_NAME, Norm=None, resume = True, ResumeFile = weights2load)

import subprocess
def IMAScaller(img1,img2, desc = 11, MatchingThres = 0, knn_num = 2, Rooted = True, GFilter=2, Visual=False):
    cv2.imwrite("/tmp/img1.png",img1)
    cv2.imwrite("/tmp/img2.png",img2)
    _ = subprocess.check_output("cp acc-test/z_main /tmp", shell=True)
    _ = subprocess.check_output('cd /tmp && ./z_main -im1 "./img1.png" -im2 "./img2.png" -desc %d -applyfilter %d > imas.out'%(desc,GFilter), shell=True)
    # imasout = subprocess.check_output('cd /tmp && cat imas.out', shell=True).decode('utf-8')
    # print(imasout)
    ET_KP = float(subprocess.check_output('cd /tmp && cat imas.out | grep "IMAS-Detector accomplished in" | cut -f 4 -d" " ', shell=True).decode('utf-8'))
    ET_M = float(subprocess.check_output('cd /tmp && cat imas.out | grep "IMAS-Matcher accomplished in" | cut -f 4 -d" " ', shell=True).decode('utf-8'))

    KPs1 = int(subprocess.check_output('cd /tmp && cat imas.out | grep "image 1" | cut -f 7 -d" " ', shell=True).decode('utf-8'))
    KPs2 = int(subprocess.check_output('cd /tmp && cat imas.out | grep "image 2" | cut -f 7 -d" " ', shell=True).decode('utf-8'))
    simus = int(subprocess.check_output('cd /tmp && cat imas.out | grep "image 1" | cut -f 14 -d" " ', shell=True).decode('utf-8'))

    Total = int(subprocess.check_output('cd /tmp && cat imas.out | grep "possible matches have been found" | cut -f 4 -d" " ', shell=True).decode('utf-8'))
    Filtered = int(subprocess.check_output('cd /tmp && cat imas.out | grep "Final number of matches" | cut -f 10 -d" " ', shell=True).decode('utf-8')[:-2])
    return Total, Filtered, ET_KP, ET_M, KPs1, KPs2, simus


import sklearn.preprocessing
def RootSIFT(img1,img2, MatchingThres = 0, knn_num = 2, Rooted = True, GFilter=1, Visual=False):
    start_time = time.time()
    KPlist1, sift_des1 = ComputeSIFTKeypoints(img1, Desc = True)
    KPlist2, sift_des2 = ComputeSIFTKeypoints(img2, Desc = True)
    Identity = np.float32([[1, 0, 0], [0, 1, 0]])
    h, w = img1.shape[:2]
    KPlist1, sift_des1, temp = Filter_Affine_In_Rect(KPlist1,Identity,[0,0],[w,h], desc_list = sift_des1)
    h, w = img2.shape[:2]
    KPlist2, sift_des2, temp = Filter_Affine_In_Rect(KPlist2,Identity,[0,0],[w,h], desc_list = sift_des2)
    if Rooted:
        sift_des1 = np.sqrt(sklearn.preprocessing.normalize(sift_des1, norm='l2',axis=1))
        sift_des2 = np.sqrt(sklearn.preprocessing.normalize(sift_des2, norm='l2',axis=1))

    ET_KP = time.time() - start_time


    bf = cv2.BFMatcher()
    start_time = time.time()
    sift_matches = bf.knnMatch(sift_des1,sift_des2, k=knn_num)
    ET_M = time.time() - start_time

    # Apply ratio test
    lda = CPPbridge('./build/libDA.so')
    sift_all = []
    if knn_num==2:
        for m,n in sift_matches:
            if m.distance < MatchingThres*n.distance:
                sift_all.append(m)
    elif knn_num==1:
        for m in sift_matches:
            if m[0].distance <= MatchingThres:
                sift_all.append(m[0])

    sift_all = OnlyUniqueMatches(sift_all,KPlist1,KPlist2,SpatialThres=5)

    sift_consensus = []
    if GFilter>0 and len(sift_all)>10:
        sift_src_pts = np.float32([ KPlist1[m.queryIdx].pt for m in sift_all ]).ravel()
        sift_dst_pts = np.float32([ KPlist2[m.trainIdx].pt for m in sift_all ]).ravel()
        matchesMask_sift, H_sift = lda.GeometricFilter(sift_src_pts, img1, sift_dst_pts, img2, Filer='ORSA_H')

        for i in range(0,len(matchesMask_sift)):
            if matchesMask_sift[i]==True:
                sift_consensus.append(sift_all[i])

    if Visual:
        img4 = cv2.drawMatches(img1,KPlist1,img2,KPlist2,sift_all, None,flags=2)
        cv2.imwrite('./temp/SIFTmatches.png',img4)
        img4 = cv2.drawMatches(img1,KPlist1,img2,KPlist2,sift_consensus, None,flags=2)
        cv2.imwrite('./temp/SIFT_homography_matches.png',img4)

    return sift_all, sift_consensus, ET_KP, ET_M


def siftAID(img1,img2, MatchingThres = math.inf, Simi='SignProx', knn_num = 1, GFilter=1, Visual=False, safe_sim_thres_pos = 0.8, safe_sim_thres_neg = 0.2, GetAllMatches=False):
    if Simi=='CosProx':
        FastCode = 0
    elif Simi=='SignProx':
        FastCode = 1
    else:
        print('Wrong similarity choice for AI-SIFT !!!')
        exit()

    # find the keypoints with SIFT
    start_time = time.time()
    KPlist1, sift_des1 = ComputeSIFTKeypoints(img1, Desc = True)
    KPlist2, sift_des2 = ComputeSIFTKeypoints(img2, Desc = True)
    Identity = np.float32([[1, 0, 0], [0, 1, 0]])
    h, w = img1.shape[:2]
    KPlist1, sift_des1, temp = Filter_Affine_In_Rect(KPlist1,Identity,[0,0],[w,h], desc_list = sift_des1)
    h, w = img2.shape[:2]
    KPlist2, sift_des2, temp = Filter_Affine_In_Rect(KPlist2,Identity,[0,0],[w,h], desc_list = sift_des2)

    maxoctaves = 4
    pyr1 = buildGaussianPyramid( img1, maxoctaves + 2 )
    pyr2 = buildGaussianPyramid( img2, maxoctaves + 2 )

    patches1, A_list1, Ai_list1 = ComputePatches(KPlist1,pyr1)
    patches2, A_list2, Ai_list2 = ComputePatches(KPlist2,pyr2)

    bP = np.zeros( shape = tuple([len(patches1)])+tuple(np.shape(patches1[0]))+tuple([1]), dtype=np.float32)
    for k in range(0,len(patches1)):
        bP[k,:,:,0] = patches1[k][:,:]/255.0
    emb_1 = train_model.get_layer("aff_desc").predict(bP)

    bP = np.zeros( shape=tuple([len(patches2)])+tuple(np.shape(patches2[0]))+tuple([1]), dtype=np.float32)
    for k in range(0,len(patches2)):
        bP[k,:,:,0] = patches2[k][:,:]/255.0
    emb_2 = train_model.get_layer("aff_desc").predict(bP)

    ET_KP = time.time() - start_time

    desc_dim = np.shape(emb_1)[1]
    lda = CPPbridge('./build/libDA.so')
    lda.CreateMatcher(desc_dim, k = knn_num, sim_thres = MatchingThres)
    start_time = time.time()
    lda.KnnMatch(KPlist1,emb_1, KPlist2,emb_2,FastCode)    
    ET_M = time.time() - start_time

    AID_all = []
    if GetAllMatches:
        # Retreving Matches in a slow way
        f, l = lda.FirstLast_QueryNodes()
        
        while l is not None:
            QueryMatches = lda.GetMatches_from_QueryNode(l)
            qdist = []
            if len(QueryMatches)>1:
                qdist = np.concatenate(([math.inf],
                np.diff(np.array([m.distance for m in QueryMatches]))))
                for mi in range(0,len(qdist)):
                    if (QueryMatches[mi].distance>=safe_sim_thres_pos or -qdist[mi]>safe_sim_thres_neg):
                        AID_all.append(QueryMatches[mi])
                    else:
                        break
            elif len(QueryMatches)==1:
                AID_all.append(QueryMatches[0])
            if (l==f):
                break
            l = lda.PrevQueryNode(l)
        AID_all = OnlyUniqueMatches(AID_all,KPlist1,KPlist2,SpatialThres=5)


    AID_consensus, H_AID = lda.GeometricFilterFromMatcher(img1, img2, Filer='ORSA_H')

    if Visual:
        # img3 = cv2.drawMatches(img1,KPlist1,img2,KPlist2,AID_good, None,flags=2)
        # cv2.imwrite('./temp/AID_total_matches.png',img3)
        img3 = cv2.drawMatches(img1,KPlist1,img2,KPlist2,AID_consensus, None,flags=2)
        cv2.imwrite('./temp/AID_homography_matches.png',img3)
        h, w = img2.shape[:2]
        warp_AID = cv2.warpPerspective(img1, H_AID,(w, h))
        warp_AID = warp_AID
        cv2.imwrite('./temp/AID_panorama.png',warp_AID)

    return AID_all, AID_consensus, ET_KP, ET_M
