#  VGG like network
from keras import layers
from keras.models import Model
import tensorflow as tf
from keras.backend.tensorflow_backend import set_session
config = tf.ConfigProto(allow_soft_placement=True)
#, device_count = {'CPU' : 1, 'GPU' : 1})
config.gpu_options.per_process_gpu_memory_fraction = 0.1
set_session(tf.Session(config=config))
import numpy as np
from matplotlib import pyplot as plt
plt.switch_backend('agg')
from scipy.stats import norm
import matplotlib.mlab as mlab

import sys
sys.path.append(".")
from library import *
import sklearn.preprocessing


DoAllVecDensities = False
Do49VecDensities = False
DoClasssDensities = True


ConstrastSimu = True
def ProcessData(GA, stacked_patches, groundtruth_pts):
    if ConstrastSimu:
        channels = np.int32(np.shape(stacked_patches)[2]/2)
        val1 = random.uniform(1/3, 3)
        val2 = random.uniform(1/3, 3)
        for i in range(channels):
            stacked_patches[:,:,i] = np.power(stacked_patches[:,:,i],val1)
            stacked_patches[:,:,channels+i] = np.power(stacked_patches[:,:,channels+i],val2)
    return stacked_patches, groundtruth_pts #if ConstrastSimu==False -> Identity

GAval = GenAffine("./imgs-test/", save_path = "./db-gen-test-75/")
Set_FirstThreadTouch(GAval,False)


lenbP = 60000
bP1 = np.zeros( shape = tuple([lenbP])+tuple([60,60])+tuple([1]), dtype=np.float32)
bP2 = np.zeros( shape = tuple([lenbP])+tuple([60,60])+tuple([1]), dtype=np.float32)
for k in range(0,lenbP):
    stacked_patches, groundtruth_pts = GAval.Fast_gen_affine_patches()
    stacked_patches, groundtruth_pts = ProcessData(GAval, stacked_patches, groundtruth_pts)
    bP1[k,:,:,0] = stacked_patches[:,:,0]
    bP2[k,:,:,0] = stacked_patches[:,:,1]

from models import *
vgg_input_shape = tuple([60,60]) + tuple([1])
MODEL_NAME = 'AID_simCos_BigDesc_dropout'
weights2load = 'model-data/model.'+MODEL_NAME+'_75.hdf5'
train_model, sim_type = create_model(vgg_input_shape, None, model_name = MODEL_NAME, Norm=None, resume = True, ResumeFile = weights2load)
emb_1 = train_model.get_layer("aff_desc").predict(bP1)
emb_2 = train_model.get_layer("aff_desc").predict(bP2)


##############################
### Densities on each descriptors' dimension
##############################
if DoAllVecDensities:
    plt.figure(2,figsize=(7,2.5))
    for i in range(0,np.shape(emb_1)[1]):
        n, bins, patches = plt.hist(emb_1[:, i].ravel(), histtype='step', bins='auto', density=True)  # arguments are passed to np.histogram
        # y = mlab.normpdf( bins, mu, sigma)
        # plt.plot(bins, y, 'r--', linewidth=2)
        # print(mui,vari)
    plt.savefig('./temp/6272densities.png', format='png', dpi=150)
    plt.close(2)



##############################
### Densities on blocks of descriptors' dimensions
##############################
if Do49VecDensities:
    plt.figure(1,figsize=(7,2.5))
    # (mu, sigma) = norm.fit(emb_1.ravel())
    # print(mu,sigma)
    n, bins, patches = plt.hist(emb_1.ravel(), bins='auto', density=True)  # arguments are passed to np.histogram

    # y = mlab.normpdf( bins, mu, sigma)
    # plt.plot(bins, y, 'r--', linewidth=2)
    # plt.title("GaussianFit")
    mui = np.mean(emb_1.ravel())
    vari = np.std(emb_1.ravel())
    plt.title("m = %1.3f, std = %1.3f" % (mui,vari))
    plt.savefig('./temp/1Density.png', format='png', dpi=150)
    plt.close(1)


    # # stats on levels
    # # this corresponds to indices (:,:,k)
    # for i in range(0,128):
    #     mui = np.mean(emb_1[:, i::128].ravel())
    #     vari = np.std(emb_1[:, i::128].ravel())
    #     print(mui,vari)
    #
    # stats on vector corresponding to the same zone
    # this is a vector like (i,j,:)
    plt.figure(2,figsize=(28,28))
    for i in range(0,49):
        mui = np.mean(emb_1[:, i*128:(i+1)*128].ravel())
        vari = np.std(emb_1[:, i*128:(i+1)*128].ravel())
        plt.subplot(7, 7, i+1)
        n, bins, patches = plt.hist(emb_1[:, i*128:(i+1)*128].ravel(), bins='auto', density=True)  # arguments are passed to np.histogram
        # y = mlab.normpdf( bins, mu, sigma)
        # plt.plot(bins, y, 'r--', linewidth=2)
        plt.title("m = %1.3f, std = %1.3f" % (mui,vari))

        # print(mui,vari)
    plt.savefig('./temp/49densities.png', format='png', dpi=150)
    plt.close(2)


##############################
### Class Measure Densities
##############################
if DoClasssDensities:
        
    bindesc1 = np.zeros( shape = np.shape(emb_1), dtype=np.bool)
    bindesc2 = np.zeros( shape = np.shape(emb_1), dtype=np.bool)
    bindesc1[emb_1>=0] = 1
    bindesc2[emb_2>=0] = 1


    def b_xor(a,b):
        res = np.zeros(np.shape(a)[0],dtype=np.int)
        for i in range(0,np.shape(a)[0]):
            res[i] = np.shape(a)[1] - np.sum(1*(np.logical_xor(a[i,:],b[i,:])))
        return res

    def b_cos(a,b):
        res = np.zeros(np.shape(a)[0],dtype=np.float)
        for i in range(0,np.shape(a)[0]):
            res[i] = np.dot(a[i,:],b[i,:])/(np.linalg.norm(a[i,:]*np.linalg.norm(b[i,:])))
        return res



    # Proposed binary descriptor
    Npos = b_xor(bindesc1, bindesc2)
    Nneg = b_xor(bindesc1, np.roll(bindesc2, 1, axis=0))

    plt.figure(1,figsize=(7,2.5))
    _, bpos, _ = plt.hist(Npos.ravel(), bins='auto', density=True, alpha=0.5, label='Positives')
    _, bneg, _ = plt.hist(Nneg.ravel(), bins='auto', density=True, alpha=0.5, label='Negatives')
    OurBins = np.unique(np.concatenate((bpos,bneg)))
    plt.figure(3)
    Ppos, bpos, _ = plt.hist(Npos.ravel(), bins=OurBins, density=True, alpha=0.5, label='Positives')
    Pneg, bneg, _ = plt.hist(Nneg.ravel(), bins=OurBins, density=True, alpha=0.5, label='Negatives')
    Ppos = np.cumsum(Ppos*np.diff(bpos))
    Pneg = np.cumsum(Pneg*np.diff(bneg))
    amin = np.argmin( Ppos + 1 - Pneg )
    thres = OurBins[amin]
    plt.axvline(thres, color='k', linestyle='dashed', linewidth=1)
    plt.legend(loc='upper right')
    print("SignAlingmentThreshold = ",thres, ", P(X<=tau|Pos) = ", Ppos[amin], ", P(X>tau|Neg)=", 1-Pneg[amin])
    plt.figure(1)
    plt.axvline(thres, color='k', linestyle='dashed', linewidth=1,label='Threshold')
    plt.legend(loc='upper left')#'upper right')
    plt.savefig('./temp/Histo_SignAlignment.png', format='png', dpi=300)
    plt.close(3)


    # Proposed Full descriptor
    Spos = b_cos(emb_1, emb_2)
    Sneg = b_cos(emb_1, np.roll(emb_2, 1, axis=0))

    plt.figure(2,figsize=(7,2.5))
    _, bpos, _ = plt.hist(Spos.ravel(), bins='auto', density=True, alpha=0.5, label='Positives')
    _, bneg, _ = plt.hist(Sneg.ravel(), bins='auto', density=True, alpha=0.5, label='Negatives')
    OurBins = np.unique(np.concatenate((bpos,bneg)))
    plt.figure(3)
    Ppos, bpos, _ = plt.hist(Spos.ravel(), bins=OurBins, density=True, alpha=0.5, label='Positives')
    Pneg, bneg, _ = plt.hist(Sneg.ravel(), bins=OurBins, density=True, alpha=0.5, label='Negatives')
    Ppos = np.cumsum(Ppos*np.diff(bpos))
    Pneg = np.cumsum(Pneg*np.diff(bneg))
    amin = np.argmin( Ppos + 1 - Pneg )
    thres = OurBins[amin]
    plt.axvline(thres, color='k', linestyle='dashed', linewidth=1)
    print("CosProxThreshold = ",thres, ", P(X<=tau|Pos) = ", Ppos[amin], ", P(X>tau|Neg)=", 1-Pneg[amin])
    plt.figure(2)
    plt.axvline(thres, color='k', linestyle='dashed', linewidth=1,label='Threshold')
    plt.legend(loc='upper left')#'upper right')
    plt.savefig('./temp/Histo_CosProx.png', format='png', dpi=300)
    plt.close(3)


    # RootSIFT
    w, h = 60, 60
    key = [cv2.KeyPoint(x = (w-1)/2.0, y = (h-1)/2.0,
                _size = siftparams.sigma*2, _angle = 0.0,
                _response = 0.9, _octave = packSIFTOctave(0,0),
                _class_id = 0)]
    sift = cv2.xfeatures2d.SIFT_create(
    nfeatures = siftparams.nfeatures,
    nOctaveLayers = siftparams.nOctaveLayers, contrastThreshold = siftparams.contrastThreshold,
    edgeThreshold = siftparams.edgeThreshold, sigma = siftparams.sigma
    )
    _, d1 = sift.compute((bP1[0,:,:,0]*255).astype(np.uint8),key)
    siftdesc1 = np.zeros((lenbP,np.shape(d1)[1]), dtype = np.float)
    siftdesc2 = np.zeros((lenbP,np.shape(d1)[1]), dtype = np.float)
    for k in range(0,lenbP):
        _, d1 = sift.compute((bP1[k,:,:,0]*255).astype(np.uint8),key)
        _, d2 = sift.compute((bP2[k,:,:,0]*255).astype(np.uint8),key)
        siftdesc1[k,:] = np.sqrt(sklearn.preprocessing.normalize(d1, norm='l2'))[0,:]
        siftdesc2[k,:] = np.sqrt(sklearn.preprocessing.normalize(d2, norm='l2'))[0,:]

    SIFTpos = np.linalg.norm(siftdesc1- siftdesc2, axis=1)
    SIFTneg = np.linalg.norm(siftdesc1 - np.roll(siftdesc2, 1, axis=0), axis=1)

    plt.figure(3,figsize=(7,2.5))
    _, bpos, _ = plt.hist(SIFTpos.ravel(), bins='auto', density=True, alpha=0.5, label='Positives')
    _, bneg, _ = plt.hist(SIFTneg.ravel(), bins='auto', density=True, alpha=0.5, label='Negatives')
    OurBins = np.unique(np.concatenate((bpos,bneg)))
    plt.figure(4)
    Ppos, bpos, _ = plt.hist(SIFTpos.ravel(), bins=OurBins, density=True, alpha=0.5, label='Positives')
    Pneg, bneg, _ = plt.hist(SIFTneg.ravel(), bins=OurBins, density=True, alpha=0.5, label='Negatives')
    Ppos = np.cumsum(Ppos*np.diff(bpos))
    Pneg = np.cumsum(Pneg*np.diff(bneg))
    amin = np.argmin( Pneg + 1 - Ppos )
    thres = OurBins[amin]
    plt.axvline(thres, color='k', linestyle='dashed', linewidth=1)
    plt.legend(loc='upper right')
    print("RootSiftThreshold = ",thres, ", P(X>tau|Pos) = ", 1-Ppos[amin], ", P(X<=tau|Neg)=", Pneg[amin])
    plt.figure(3)
    plt.axvline(thres, color='k', linestyle='dashed', linewidth=1,label='Threshold')
    plt.legend(loc='upper left')#'upper right')
    plt.savefig('./temp/Histo_RootSIFT.png', format='png', dpi=300)
    plt.close(4)
