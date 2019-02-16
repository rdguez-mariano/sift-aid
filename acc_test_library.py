import numpy as np
import cv2
from matplotlib import pyplot as plt
from library import *
import time
import csv
import glob, os



def DA_ComputeAccuracy(GA, model, inputs, WasNetAffine = True):
    asift_KPlist1, patches1, GT_Avec_list, asift_KPlist2, patches2 = inputs
    assert len(asift_KPlist1)==len(patches1)==len(GT_Avec_list)==len(asift_KPlist2)==len(patches2)

    patchshape = np.shape(patches1[0])
    bPshape = tuple([1]) + tuple( patchshape ) + tuple([2])
    bP = np.zeros(shape=bPshape, dtype = np.float32)
    Identity = np.float32([[1, 0, 0], [0, 1, 0]])
    Ivec = affine_decomp(Identity) # [1,0,1,0,0,0]
    good = 0
    diffs_GT = []
    for k in range(0,len(asift_KPlist1)):
        bP[0,:,:,:] = np.dstack((patches1[k]/GA.imgdivfactor, patches2[k]/GA.imgdivfactor))

        # Estimated
        Avec = model.predict(bP)
        if WasNetAffine:
            Avec = Avec[0]*GA.Avec_factor + GA.Avec_tras
            A = np.reshape( affine_decomp2affine( Avec[0:6] ), (2,3) ) # transforms p2 into p1 coordinates
        else:
            A = GA.AffineFromNormalizedVector( Avec[0] ) # transforms p2 into p1 coordinates
            Avec = affine_decomp(A,doAssert=False)

        Ai = cv2.invertAffineTransform(A)
        Aivec = affine_decomp(Ai,doAssert=False)

        # Groundtruth
        GTAvec = GT_Avec_list[k] # transforms p1 into p2 coordinates
        GTA = np.reshape( affine_decomp2affine( GTAvec ), (2,3) )
        GTAi = cv2.invertAffineTransform( GTA )
        GTAivec = affine_decomp( GTAi )

        diffs_GT.append( np.array(GTAivec) - np.array(Avec) )
        diffs_GT.append( np.array(GTAvec) - np.array(Aivec) )

        if transition_tilt(Avec,GTAivec)<=transition_tilt(Ivec,GTAivec):
            good+=1

    return diffs_GT, np.float(good)/len(asift_KPlist1)

def load_acc_test_data(pathway):
    img1 = cv2.cvtColor(cv2.imread(pathway+'.1.png'),cv2.COLOR_BGR2GRAY) # queryImage
    img2 = cv2.cvtColor(cv2.imread(pathway+'.2.png'),cv2.COLOR_BGR2GRAY) # trainImage

    # sift = cv2.xfeatures2d.SIFT_create(
    # nfeatures = siftparams.nfeatures,
    # nOctaveLayers = siftparams.nOctaveLayers, contrastThreshold = siftparams.contrastThreshold,
    # edgeThreshold = siftparams.edgeThreshold, sigma = siftparams.sigma
    # )

    H = np.loadtxt(pathway+'.txt')
    csvfile = open(pathway+'.csv', 'r')
    sr = csv.reader(csvfile, delimiter=',', quotechar='|')
    asift_KPlist1 = []
    asift_KPlist2 = []
    # matches_details = []
    GT_Avec_list = []
    # list1_sift_des = []
    # list2_sift_des = []
    Identity = np.float32([[1, 0, 0], [0, 1, 0]])
    for row in sr:
        if len(row)==15 and row[0]!='x1':
            center = np.array([float(row[0]),float(row[1]),1]).reshape(3,1)
            Avec = affine_decomp(FirstOrderApprox_Homography(H,center))
            zoom_ratio = Avec[0]
            params = [[o1,l1,o2,l2] for o1 in range(0,3) for l1 in range(0, siftparams.nOctaveLayers+1) for o2 in range(0,3) for l2 in range(0, siftparams.nOctaveLayers+1)]
            mindiff = math.inf
            qmin = [0.0,0.0,0.0,0.0]
            for q in params:
                # we would like: zoom_ratio = zoom_1 / zoom_2
                temp = abs( zoom_ratio - (pow(2,-q[0])*pow(2.0, -q[1]/siftparams.nOctaveLayers))/(pow(2,-q[2])*pow(2.0, -q[3]/siftparams.nOctaveLayers)) )
                if mindiff>temp:
                    mindiff = temp
                    qmin = q

            q = qmin
            q_shift = 1
            temp = random.randint(q[1]-q_shift, q[1]+q_shift)
            while temp<0 or temp>siftparams.nOctaveLayers+1:
                temp = random.randint(q[1]-q_shift, q[1]+q_shift)
            q[1] = int(temp)
            temp = random.randint(q[3]-q_shift, q[3]+q_shift)
            while temp<0 or temp>siftparams.nOctaveLayers+1:
                temp = random.randint(q[3]-q_shift, q[3]+q_shift)
            q[3] = int(temp)

            lambda1 = pow(2,-q[0])*pow(2.0, -q[1]/siftparams.nOctaveLayers)
            angle1 = float(row[3])
            t1 = float(row[5])
            theta1 = np.deg2rad(float(row[6]))
            angle1 = angle1 if angle1>=0 else angle1+2*np.pi
            kp1 = cv2.KeyPoint(x = float(row[0]), y = float(row[1]),
                    _size = 2.0*siftparams.sigma*pow(2.0, q[1]/siftparams.nOctaveLayers)*pow(2.0,q[0]),
                    _angle = np.rad2deg(angle1),#+random.randint(0,10),
                    _response = 0.9, _octave = packSIFTOctave(q[0],q[1]),
                    _class_id = 0)
            A = cv2.invertAffineTransform( affine_decomp2affine( [1.0, 0.0, t1, theta1, 0.0, 0.0] ) )
            kp1 = AffineKPcoor([kp1],A, Pt_mod = False)[0] # it will only change the angle info
            h, w = img1.shape[:2]
            kp1, temp = Filter_Affine_In_Rect([kp1],Identity,[0,0],[w,h])

            lambda2 = pow(2,-q[2])*pow(2.0, -q[3]/siftparams.nOctaveLayers)
            angle2 = float(row[7+3])
            t2 = float(row[7+5])
            theta2 = np.deg2rad(float(row[7+6]))
            angle2 = angle2 if angle2>=0 else angle2+2*np.pi
            kp2 = cv2.KeyPoint(x = float(row[7+0]), y = float(row[7+1]),
                    _size = 2.0*siftparams.sigma*pow(2.0, q[3]/siftparams.nOctaveLayers)*pow(2.0,q[2]),
                    _angle = np.rad2deg(angle2),
                    _response = 0.9, _octave = packSIFTOctave(q[2],q[3]),
                    _class_id = 0)

            A = cv2.invertAffineTransform( affine_decomp2affine( [1.0, 0.0, t2, theta2, 0.0, 0.0] ) )
            kp2 = AffineKPcoor([kp2],A, Pt_mod = False)[0]
            h, w = img2.shape[:2]
            kp2, temp = Filter_Affine_In_Rect([kp2],Identity,[0,0],[w,h])

            if len(kp1)>0 and len(kp2)>0:
                # Uncomment this to discard angle info and set it randomly
                kp1[0].angle = random.randint(0,360)
                # kp2[0].angle = (kp1[0].angle + random.randint(-10,10) )%360
                kp2[0].angle = (kp1[0].angle )%360
                kp2 = AffineKPcoor(kp2, affine_decomp2affine(Avec), Pt_mod = False)

                # kp2, d2 = sift.compute(img2,kp2)
                # kp1, d1 = sift.compute(img1,kp1)
                # list1_sift_des.append( d1 )
                # list2_sift_des.append( d2 )
                GT_Avec_list.append( Avec ) # Still needs to be modified as is from im1 to im2
                asift_KPlist1.append( kp1[0] )
                asift_KPlist2.append( kp2[0] )
                # distance = float(row[14])
                # matches_details.append([t1,theta1, t2,theta2, distance])
    csvfile.close()

    pyr1 = buildGaussianPyramid( img1, siftparams.nOctaves + 2 )
    pyr2 = buildGaussianPyramid( img2, siftparams.nOctaves + 2 )

    patches1, A_list1, Ai_list1 = ComputePatches(asift_KPlist1,pyr1)
    patches2, A_list2, Ai_list2 = ComputePatches(asift_KPlist2,pyr2)
    assert len(patches1)==len(patches2)==len(asift_KPlist1)==len(asift_KPlist2)==len(GT_Avec_list)

    # Lets now make GT_Avec_list really go from patch1 to patch2
    for k in range(0,len(patches1)):
        Avec = GT_Avec_list[k]
        A = affine_decomp2affine(Avec)
        A = ComposeAffineMaps( A_list2[k], ComposeAffineMaps(A, Ai_list1[k]) )
        GT_Avec_list[k] = affine_decomp( A )
    return asift_KPlist1, patches1, GT_Avec_list, asift_KPlist2, patches2
