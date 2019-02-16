import cv2
import sys
sys.path.append(".")
from libLocalDesc import *
from acc_test_library import *

from matplotlib import pyplot as plt
plt.switch_backend('agg')

CosProxThres = 0.4
SignAlingThres = 4000


test_OnFiltered_ARootSIFT = True
test_timePerformances = True

if test_OnFiltered_ARootSIFT:
    totpos = 0
    totneg = 0
    xpos, ypos, xneg, yneg = (),(),(),()

    for file in glob.glob('./acc-test/*.txt'):
        image_name = os.path.basename(file)[:-4]
        pathway = './acc-test/' + image_name
        print(pathway)

        full_info = ()

        img1 = cv2.cvtColor(cv2.imread(pathway+'.1.png'),cv2.COLOR_BGR2GRAY) # queryImage
        img2 = cv2.cvtColor(cv2.imread(pathway+'.2.png'),cv2.COLOR_BGR2GRAY) # trainImage

        asift_KPlist1, patches1, GT_Avec_list, asift_KPlist2, patches2 = load_acc_test_data(pathway)
        Pa = np.zeros(shape=tuple([len(patches1)])+tuple(np.shape(patches1)[1:])+tuple([1]),dtype=np.float32)
        Pp = np.zeros(shape=tuple([len(patches1)])+tuple(np.shape(patches1)[1:])+tuple([1]),dtype=np.float32)
        for k in range(0,len(patches1)):
            Pa[k,:,:,0] = patches1[k][:,:]/255.0
            Pp[k,:,:,0] = patches2[k][:,:]/255.0
        emb_1 = train_model.get_layer("aff_desc").predict(Pa)
        emb_2 = train_model.get_layer("aff_desc").predict(Pp)
        bindesc_1 = emb_1>=0
        bindesc_2 = emb_2>=0

        src_pts = ()
        dst_pts = ()
        cosprox = train_model.get_layer("sim").predict([emb_1,emb_2])
        signalign = np.logical_xor(bindesc_1,bindesc_2)
        localpos = 0.0
        localneg = 0.0
        for k in range(0,np.shape(cosprox)[0]):
            if cosprox[k]>CosProxThres:
            # if 6272-np.sum(1*signalign[k,:])>SignAlingThres:
                xpos += tuple([GT_Avec_list[k][0]])
                ypos += tuple([np.rad2deg(np.arccos(1.0/GT_Avec_list[k][2]))])
                src_pts += tuple([asift_KPlist1[k].pt])
                dst_pts += tuple([asift_KPlist2[k].pt])
                totpos += 1.0
                localpos += 1.0
            else:
                xneg += tuple([GT_Avec_list[k][0]])
                yneg += tuple([np.rad2deg(np.arccos(1.0/GT_Avec_list[k][2]))])
                totneg += 1.0
                localneg += 1.0
        lda = CPPbridge('./build/libDA.so')
        src_pts = np.float32(np.array(src_pts)).ravel()
        dst_pts = np.float32(np.array(dst_pts)).ravel()
        matchesMask_AID, H_AID = lda.GeometricFilter(src_pts, img1, dst_pts, img2, Filer='ORSA_H')
        print("   AID score on SIFT Keypoints created from ASIFT Keypoints %3.0f%%, if applied only on true matches this should hold %d=%d "%(100.0*localpos/(localpos+localneg),len(matchesMask_AID),localpos) )

    plt.figure(1,figsize=(7,7))
    plt.plot(xpos,ypos,'b+',label='Matched')
    plt.plot(xneg,yneg,'rx',label='Missed')

    plt.xlabel('Zoom')
    plt.ylabel('Viewpoint angle')
    plt.legend(loc='upper right')

    plt.savefig('./temp/SeenBysiftAID.png', format='png', dpi=300)
    plt.close(1)
    plt.show()

    print("Overall AID score (%3.0f%%): %d Matched, %d Missed"%(100.0*totpos/(totpos+totneg), totpos, totneg))


if test_timePerformances:
    tableinfo = []
    names = []
    for file in glob.glob('./acc-test/*.txt'):
        image_name = os.path.basename(file)[:-4]
        pathway = './acc-test/' + image_name
        names.append(image_name)
        print(pathway)

        full_info = ()

        img1 = cv2.cvtColor(cv2.imread(pathway+'.1.png'),cv2.COLOR_BGR2GRAY) # queryImage
        img2 = cv2.cvtColor(cv2.imread(pathway+'.2.png'),cv2.COLOR_BGR2GRAY) # trainImage

        # method = 'Optimal Affine-RootSIFT'
        # total, good_HC, ET_KP, ET_M, KPs1, KPs2, simus = IMAScaller(img1,img2, desc = 11)
        # full_info += (good_HC,total,ET_KP,ET_M)
        # print(method+" --> HC = %d, TM = %d, ET_KP = %3.3f, ET_M = %3.3f" %(good_HC,total,ET_KP,ET_M))


        _, _, _, _  = RootSIFT(img1,img2, Rooted = False, MatchingThres = 0.8, knn_num=2)

        # method = 'SIFT'
        # total, good_HC, ET_KP, ET_M = RootSIFT(img1,img2, Rooted = False, MatchingThres = 200, knn_num=1)
        # print("%s --> HC = %d, TM = %d, ET_KP = %3.3f, ET_M = %3.3f" %(method,len(good_HC),len(total),ET_KP,ET_M))
        #
        #
        # method = 'SIFT 2NNratio'
        # total, good_HC, ET_KP, ET_M = RootSIFT(img1,img2, Rooted = False, MatchingThres = 0.8, knn_num=2)
        # print("%s --> HC = %d, TM = %d, ET_KP = %3.3f, ET_M = %3.3f" %(method,len(good_HC),len(total),ET_KP,ET_M))


        # method = 'Root-SIFT'
        # total, good_HC, ET_KP, ET_M = RootSIFT(img1,img2, MatchingThres = 1.78, knn_num=1)
        # print("%s --> HC = %d, TM = %d, ET_KP = %3.3f, ET_M = %3.3f" %(method,len(good_HC),len(total),ET_KP,ET_M))


        method = 'Root-SIFT 2NNratio'
        total, good_HC, ET_KP, ET_M = RootSIFT(img1,img2, MatchingThres = 0.8, knn_num=2)
        # full_info += (len(good_HC),len(total),ET_KP,ET_M)
        print(method+" --> HC = %d, TM = %d, ET_KP = %3.3f, ET_M = %3.3f" %(len(good_HC),len(total),ET_KP,ET_M))



        # do this so train_model.get_layer("aff_desc") gets charged in memory
        _, _, _, _ = siftAID(img1,img2, Simi = 'SignProx', Visual=False, GFilter=0)

        method = 'siftAID_CosProx'
        total, good_HC, ET_KP, ET_M = siftAID(img1,img2, MatchingThres = CosProxThres, Simi = 'CosProx', Visual=True, GetAllMatches=True)
        # full_info += (len(good_HC),len(total),ET_KP,ET_M)
        print(method+" --> HC = %d, TM = %d, ET_KP = %3.3f, ET_M = %3.3f" %(len(good_HC),len(total),ET_KP,ET_M))

        method = 'siftAID_SignProx'
        total, good_HC, ET_KP, ET_M = siftAID(img1,img2, MatchingThres = SignAlingThres, Simi = 'SignProx', Visual=True, GetAllMatches=True)
        # full_info += (len(good_HC),len(total),ET_KP,ET_M)
        print(method+" --> HC = %d, TM = %d, ET_KP = %3.3f, ET_M = %3.3f" %(len(good_HC),len(total),ET_KP,ET_M))

        # tableinfo.append("%d & %d & %3.3f & %3.3f & %d & %d & %3.3f & %3.3f & %d & %d & %3.3f & %3.3f & %d & %d & %3.3f & %3.3f" %full_info)

    # i=0
    # for s in tableinfo:
    #     print(names[i]+" & "+s)
    #     i+=1
