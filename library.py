import numpy as np
import cv2
import math
import time
import glob, os
import random
import psutil
import ctypes
from datetime import datetime

MaxSameKP_dist = 5 # pixels
MaxSameKP_angle = 10 # degrees

class ClassSIFTparams():
    def __init__(self, nfeatures = 0, nOctaveLayers = 3, contrastThreshold = 0.04, edgeThreshold = 10, sigma = 1.6, firstOctave = -1, sift_init_sigma = 0.5, graydesc = True):
        self.nOctaves = 4
        self.nfeatures = nfeatures
        self.nOctaveLayers = nOctaveLayers
        self.contrastThreshold = contrastThreshold
        self.edgeThreshold = edgeThreshold
        self.sigma = sigma
        self.firstOctave = firstOctave
        self.sift_init_sigma = sift_init_sigma
        self.graydesc = graydesc
        self.flt_epsilon = 1.19209e-07
        self.lambda_descr = 6
        self.new_radius_descr = 29.5


siftparams = ClassSIFTparams(graydesc = True)


class GenAffine():
    def __init__(self, path_to_imgs, tmax = 75, zmax = 1.6, save_path = "/tmp", saveMem = True, normalizeIMG=True, ActiveRadius=60, ReducedRadius=0.25, DoBigEpochs=False):
        self.path_to_imgs = path_to_imgs
        self.save_path = save_path
        TouchDir(save_path)
        self.max_simu_zoom = zmax
        self.max_simu_theta = np.deg2rad(tmax)
        self.imgs_path = ()
        self.imgs = []
        self.imgs_gray = []
        self.KPlists = []
        self.saveMem = saveMem
        self.imgdivfactor = 1.0
        if normalizeIMG==True:
            self.imgdivfactor = 255.0
        ar = np.float32(ActiveRadius)
        rr = np.float32(ReducedRadius)
        self.vecdivfactor = ar/rr
        self.vectraslation = (1.0 - rr)/2.0 #+rr/2.0
        self.gen_P_list = []
        self.gen_GT_list = []
        self.gen_dirs = []
        self.Avec_tras = np.float32(  [0.0, -math.pi,    1.0,    -math.pi, -4.0*siftparams.new_radius_descr,   -4.0*siftparams.new_radius_descr])
        self.Avec_factor = np.float32([2.0, 2.0*math.pi, 8.0, 2.0*math.pi, 2.0*4.0*np.sqrt(2)*siftparams.new_radius_descr, 2.0*4.0*siftparams.new_radius_descr])
        self.BigAffineVec = False
        self.DoBigEpochs = DoBigEpochs
        self.LastTimeDataChecked = time.time()
        self.GAid = random.randint(0,1000)
        set_big_epoch_number(self,0)

        for dir in glob.glob(self.save_path+"/*.npz"):
            self.gen_dirs.append(dir)

        imgspaths = (self.path_to_imgs+"/*.png", self.path_to_imgs+"/*.jpg")
        for ips in imgspaths:
            for file in glob.glob(ips):
                self.imgs_path += tuple([file])
                if saveMem==False:
                    self.imgs.append( cv2.imread(file) )
                    self.imgs_gray.append( cv2.cvtColor(self.imgs[len(self.imgs)-1],cv2.COLOR_BGR2GRAY) )
                    self.KPlists.append( ComputeSIFTKeypoints(self.imgs[len(self.imgs)-1]) )
        assert (len(self.imgs_path)>0), 'We need at least one image in folder '+self.path_to_imgs

    def NormalizeVector(self,vec):
        ''' For stability reasons, the network should be trained with a normalized vector.
        Use this function to normalize a vector in patch coordinates and make it compatile
        with the output of the network.
        '''
        return vec/self.vecdivfactor + self.vectraslation

    def UnNormalizeVector(self,vec):
        ''' For stability reasons, the network should be trained with a normalized vector.
        Use this function to unnormalize an output vector of the network.
        The resulting vector info will be now in patch coordinates.
        '''
        return (vec - self.vectraslation)*self.vecdivfactor

    def Nvec2Avec(self, normalizedvec):
        ''' Computes the passage from a normalized vector to the affine_decomp vector
        normalizedvec has the flatten normalized info of points x1,...,x8, such that
              A(ci) = xi  for  i=1,...,4
           A^-1(ci) = xi  for  i=5,...,8
           where ci are the corners of a patch
        '''
        A = np.array(self.AffineFromNormalizedVector(normalizedvec))
        avec = (affine_decomp(A)-self.Avec_tras)/self.Avec_factor
        assert np.greater_equal(avec,np.zeros(np.shape(avec))).all() and np.less_equal(avec,np.ones(np.shape(avec))).all(), 'Failed attempt to Normalize affine parameters in Nvec2Avec \n ' + str(avec)
        if self.BigAffineVec:
            Ai = np.array(cv2.invertAffineTransform(A))
            aivec = (affine_decomp(Ai)-self.Avec_tras)/self.Avec_factor
            assert np.greater_equal(aivec,np.zeros(np.shape(aivec))).all() and np.less_equal(aivec,np.ones(np.shape(aivec))).all(), 'Failed attempt to Normalize inverse affine parameters in Nvec2Avec \n ' + str(aivec)
            return np.concatenate((avec,aivec))
        return avec


    def Avec2Nvec(self, affinevec, d = np.int32(siftparams.new_radius_descr*2)+1):
        ''' Computes the passage from an affine_decomp vector to a normalized vector which
        has the flatten normalized info of points x1,...,x8, such that
              A(ci) = xi  for  i=1,...,4
           A^-1(ci) = xi  for  i=5,...,8
           where ci are the corners of a patch
        '''
        SquarePatch = SquareOrderedPts(d,d,CV=False)
        avec = affine_decomp2affine(affinevec[0:6]*self.Avec_factor + self.Avec_tras)
        A = np.reshape(avec,(2,3))
        evec = np.zeros((16),np.float32)
        evec[0:8] = self.NormalizeVector( Pts2Flatten(AffineArrayCoor(SquarePatch,A)) )

        if self.BigAffineVec:
            aivec = affine_decomp2affine(affinevec[6:12]*self.Avec_factor + self.Avec_tras)
            Ai = np.reshape(aivec,(2,3))
            evec[8:16] = self.NormalizeVector( Pts2Flatten(AffineArrayCoor(SquarePatch,Ai)) )
        else:
            evec[8:16] = self.NormalizeVector( Pts2Flatten(AffineArrayCoor(SquarePatch,cv2.invertAffineTransform(A))) )
        return evec

    def AffineFromNormalizedVector(self,vec0, d = np.int32(siftparams.new_radius_descr*2)+1):
        ''' Computes the affine map fitting vec0.
         vec0 has the flatten normalized info of points x1,...,x8, such that
               A(ci) = xi  for  i=1,...,4
            A^-1(ci) = xi  for  i=5,...,8
         where ci are the corners of a patch
        '''
        vec = self.UnNormalizeVector(vec0.copy())
        X = SquareOrderedPts(d,d,CV=False)
        Y1 = Flatten2Pts(vec[0:8])
        Y2 = Flatten2Pts(vec[8:16])
        return AffineFit(np.concatenate((X, Y2)),np.concatenate((Y1, X)))

    def MountGenData(self, MaxData = 31500):
        start_time = time.time()
        if len(self.gen_dirs)>0:
            i = random.randint(0,len(self.gen_dirs)-1)
            path = self.gen_dirs.pop(i)
            print("\n Loading Gen Data (MaxData = "+str(MaxData)+") from "+path+" \n", end="")
            npzfile = np.load(path)
            vec_list = npzfile['vec_list']
            p1_list = npzfile['p1_list']
            p2_list = npzfile['p2_list']
            assert len(vec_list)==len(p1_list) and len(vec_list)==len(p2_list)
            for i in range(0,len(vec_list)):
                self.gen_P_list.append( np.dstack((p1_list[i].astype(np.float32)/self.imgdivfactor, p2_list[i].astype(np.float32)/self.imgdivfactor)) )
                self.gen_GT_list.append( self.NormalizeVector(vec_list[i]) )

                if np.int32( len(self.gen_P_list) % np.int32(MaxData/10) )==0:
                    elapsed_time = time.time() - start_time
                    tstr = time.strftime("%H:%M:%S", time.gmtime(elapsed_time))
                    start_time = time.time()
                    print("\n "+str(MaxData/10) +" items loaded in "+ tstr, end="")
                    # print("\r "+ str(MaxData/10) +" items loaded in "+ tstr, end="")

                if len(self.gen_P_list)>=MaxData:
                    print("\n Maximal data items attained !")
                    break

    def ScatteredGenData_2_BlockData(self, BlockItems = 31000):
        hours, minutes, seconds = HumanElapsedTime(self.LastTimeDataChecked,time.time())
        if minutes>30 or hours>0:
            self.LastTimeDataChecked = time.time()
            globtxt_list = []
            SaveData = False
            for file in glob.glob(self.save_path+"/*.txt"):
                globtxt_list.append(file)
                if len(globtxt_list)==BlockItems:
                    SaveData = True
                    break
            if SaveData:
                start_time = time.time()
                vec_list =[]
                p1_list = []
                p2_list = []
                for file in globtxt_list:
                    try:
                        vec = np.loadtxt(file)
                        if len(vec)!=16:
                            print("There was an error loading generated vector. That pair will be skipped !")
                            continue
                        os.remove(file)
                        file = file[0:(len(file)-10)]
                        p1 = cv2.imread(file+"p1.png")
                        p2 = cv2.imread(file+"p2.png")
                        if (siftparams.graydesc):
                            p1 = cv2.cvtColor(p1, cv2.COLOR_BGR2GRAY).astype(np.uint8)
                            p2 = cv2.cvtColor(p2, cv2.COLOR_BGR2GRAY).astype(np.uint8)
                        vec_list.append(vec)
                        p1_list.append(p1)
                        p2_list.append(p2)
                        os.remove(file+"p1.png")
                        os.remove(file+"p2.png")
                    except:
                        print("Error loading data. That pair will be skipped !")
                ts = datetime.now().strftime("%d-%m-%Y_%H:%M:%S")
                np.savez(self.save_path+'/block_'+ts,vec_list=vec_list,p1_list=p1_list,p2_list=p2_list)
                elapsed_time = time.time() - start_time
                tstr = time.strftime("%H:%M:%S", time.gmtime(elapsed_time))
                print("A block of data was created in "+ tstr)

    def AvailableGenData(self):
        return len(self.gen_P_list)+len(self.gen_dirs)

    def Fast_gen_affine_patches(self):
        if self.DoBigEpochs and len(self.gen_P_list)<=10 and len(self.gen_dirs)==0:
            set_big_epoch_number(self,get_big_epoch_number(self)+1)
            for dir in glob.glob(self.save_path+"/*.npz"):
                self.gen_dirs.append(dir)
        while len(self.gen_P_list)<=10 and len(self.gen_dirs)>0:
            self.MountGenData()

        if (len(self.gen_P_list)==0):
            return self.gen_affine_patches()
        else:
            i = random.randint(0,len(self.gen_P_list)-1)
            return self.gen_P_list.pop(i), self.gen_GT_list.pop(i)

    def gen_affine_patches(self):
        while True:
            idx_img = random.randint(0,len(self.imgs_path)-1)

            if self.saveMem==True:
                img = cv2.cvtColor(cv2.imread(self.imgs_path[idx_img]), cv2.COLOR_BGR2GRAY)
                KPlist = ComputeSIFTKeypoints(img)
            else:
                img = self.imgs[idx_img]
                KPlist = self.KPlists[idx_img]

            h, w = img.shape[:2]

            im1_zoom = random.uniform(1.0, self.max_simu_zoom)
            im2_zoom = random.uniform(1.0, self.max_simu_zoom)

            theta1 = random.uniform(0.0, self.max_simu_theta)
            theta2 = random.uniform(0.0, self.max_simu_theta)

            im1_t = 1.0/np.cos(theta1)
            im2_t = 1.0/np.cos(theta2)

            im1_phi1 = np.rad2deg( random.uniform(0.0, math.pi) )
            im2_phi1 = np.rad2deg( random.uniform(0.0, math.pi) )

            im1_phi2 = np.rad2deg( random.uniform(0.0, 2*math.pi) )
            im2_phi2 = np.rad2deg( random.uniform(0.0, 2*math.pi) )

            img1, mask1, A1, Ai1 = SimulateAffineMap(im1_zoom, im1_phi2, im1_t, im1_phi1, img)
            KPlist1 = ComputeSIFTKeypoints(img1)
            KPlist1, temp = Filter_Affine_In_Rect(KPlist1,A1,[0,0],[w,h])
            KPlist1_affine = AffineKPcoor(KPlist1,Ai1)

            img2, mask2, A2, Ai2 = SimulateAffineMap(im2_zoom, im2_phi2, im2_t, im2_phi1, img)
            KPlist2 = ComputeSIFTKeypoints(img2)
            KPlist2, temp = Filter_Affine_In_Rect(KPlist2,A2,[0,0],[w,h])
            KPlist2_affine = AffineKPcoor(KPlist2,Ai2)

            for i in np.random.permutation( range(0,len(KPlist)) ):
                idx1 = FilterKPsinList(KPlist[i], KPlist1_affine)
                idx2 = FilterKPsinList(KPlist[i], KPlist2_affine)
                sidx1, sidx2 = FindBestKPinLists( im1_zoom, im2_zoom, [KPlist1_affine[i] for i in idx1],[KPlist2_affine[i] for i in idx2])
                if np.size(idx1)>0 and np.size(idx2)>0 and sidx1 !=None and sidx2 !=None:
                    idx1 = idx1[sidx1:sidx1+1]
                    idx2 = idx2[sidx2:sidx2+1]

                    o, l, s = unpackSIFTOctave(KPlist1[idx1[0]])
                    pyr1 = buildGaussianPyramid( img1, o+2 )
                    o, l, s = unpackSIFTOctave(KPlist2[idx2[0]])
                    pyr2 = buildGaussianPyramid( img2, o+2 )

                    patches1, A_list1, Ai_list1 = ComputePatches(KPlist1[idx1[0]:idx1[0]+1],pyr1)
                    patches2, A_list2, Ai_list2 = ComputePatches(KPlist2[idx2[0]:idx2[0]+1],pyr2)


                    hs, ws = patches1[0].shape[:2]
                    p = np.zeros((hs, ws), np.uint8)
                    p[:] = 1

                    AS1 = ComposeAffineMaps(A_list1[0],A1)
                    AS2 = ComposeAffineMaps(A_list2[0],A2)
                    ASi1 = ComposeAffineMaps(Ai1,Ai_list1[0])
                    ASi2 = ComposeAffineMaps(Ai2,Ai_list2[0])


                    A_from_1_to_2 = ComposeAffineMaps(AS2,ASi1)
                    A_from_2_to_1 = ComposeAffineMaps(AS1,ASi2)

                    kp_sq = SquareOrderedPts(hs,ws)

                    kp_sq1 = AffineKPcoor(kp_sq,A_from_2_to_1)
                    kp_sq2 = AffineKPcoor(kp_sq,A_from_1_to_2)
                    kpin1 = [pt for k in kp_sq1 for pt in k.pt]
                    kpin2 = [pt for k in kp_sq2 for pt in k.pt]

                    stamp = str(time.time())+'.'+ str(np.random.randint(0,9999))
                    cv2.imwrite(self.save_path+"/"+stamp+".p1.png",patches1[0])
                    cv2.imwrite(self.save_path+"/"+stamp+".p2.png",patches2[0])
                    np.savetxt(self.save_path+"/"+stamp+".vector.txt", np.concatenate((kpin1,kpin2)))
                    # to retreive info do:
                    # np.concatenate((kpin1,kpin2)) = np.loadtxt(self.save_path+"/"+stamp+".vector.txt")
                    return np.dstack((patches1[0]/self.imgdivfactor, patches2[0]/self.imgdivfactor)), self.NormalizeVector(np.concatenate((kpin1,kpin2)))




def SimulateAffineMap(zoom_step,psi,t1_step,phi,img0,mask=None, CenteredAt=None, t2_step = 1.0, inter_flag = cv2.INTER_CUBIC, border_flag = cv2.BORDER_CONSTANT, SimuBlur = True):
    '''
    Computing affine deformations of images as in [https://rdguez-mariano.github.io/pages/imas]
    Let A = R_psi0 * diag(t1,t2) * R_phi0    with t1>t2
          = lambda * R_psi0 * diag(t1/t2,1) * R_phi0

    Parameters given should be as:
    zoom_step = 1/lambda
    t1_step = 1/t1
    t2_step = 1/t2
    psi = -psi0 (in degrees)
    phi = -phi0 (in degrees)

    ASIFT proposed params:
    inter_flag = cv2.INTER_LINEAR
    SimuBlur = True

    Also, another kind of exterior could be:
    border_flag = cv2.BORDER_REPLICATE
    '''

    tx = zoom_step*t1_step
    ty = zoom_step*t2_step
    assert tx>=1 and ty>=1, 'Either scale or t are defining a zoom-in operation. If you want to zoom-in do it manually. tx = '+str(tx)+', ty = '+str(ty)

    img = img0.copy()
    arr = []
    DoCenter = False
    if type(CenteredAt) is list:
        DoCenter = True
        arr = np.array(CenteredAt).reshape(-1,2)

    h, w = img.shape[:2]
    tcorners = SquareOrderedPts(h,w,CV=False)
    if mask is None:
        mask = np.zeros((h, w), np.uint8)
        mask[:] = 255
    A1 = np.float32([[1, 0, 0], [0, 1, 0]])

    if phi != 0.0:
        phi = np.deg2rad(phi)
        s, c = np.sin(phi), np.cos(phi)
        A1 = np.float32([[c,-s], [ s, c]])
        tcorners = np.dot(tcorners, A1.T)
        x, y, w, h = cv2.boundingRect(np.int32(tcorners).reshape(1,-1,2))
        A1 = np.hstack([A1, [[-x], [-y]]])
        if DoCenter and tx == 1.0 and ty == 1.0 and psi == 0.0:
            arr = AffineArrayCoor(arr,A1)[0].ravel()
            h0, w0 = img0.shape[:2]
            A1[0][2] += h0/2.0 - arr[0]
            A1[1][2] += w0/2.0 - arr[1]
            w, h = w0, h0
            img = cv2.warpAffine(img, A1, (w, h), flags=inter_flag, borderMode=border_flag)
        else:
            img = cv2.warpAffine(img, A1, (w, h), flags=inter_flag, borderMode=border_flag)


    h, w = img.shape[:2]
    A2 = np.float32([[1, 0, 0], [0, 1, 0]])
    tcorners = SquareOrderedPts(h,w,CV=False)
    if tx != 1.0 or ty != 1.0:
        sx = 0.8*np.sqrt(tx*tx-1)
        sy = 0.8*np.sqrt(ty*ty-1)
        if SimuBlur:
            img = cv2.GaussianBlur(img, (0, 0), sigmaX=sx, sigmaY=sy)
        A2[0] /= tx
        A2[1] /= ty

    if psi != 0.0:
        psi = np.deg2rad(psi)
        s, c = np.sin(psi), np.cos(psi)
        Apsi = np.float32([[c,-s], [ s, c]])
        Apsi = np.matmul(Apsi,A2[0:2,0:2])
        tcorners = np.dot(tcorners, Apsi.T)
        x, y, w, h = cv2.boundingRect(np.int32(tcorners).reshape(1,-1,2))
        A2[0:2,0:2] = Apsi
        A2[0][2] -= x
        A2[1][2] -= y



    if tx != 1.0 or ty != 1.0 or psi != 0.0:
        if DoCenter:
            A = ComposeAffineMaps(A2,A1)
            arr = AffineArrayCoor(arr,A)[0].ravel()
            h0, w0 = img0.shape[:2]
            A2[0][2] += h0/2.0 - arr[0]
            A2[1][2] += w0/2.0 - arr[1]
            w, h = w0, h0
        img = cv2.warpAffine(img, A2, (w, h), flags=inter_flag, borderMode=border_flag)

    A = ComposeAffineMaps(A2,A1)

    if psi!=0 or phi != 0.0 or tx != 1.0 or ty != 1.0:
        if DoCenter:
            h, w = img0.shape[:2]
        else:
            h, w = img.shape[:2]
        mask = cv2.warpAffine(mask, A, (w, h), flags=inter_flag)
    Ai = cv2.invertAffineTransform(A)
    return img, mask, A, Ai


def unpackSIFTOctave(kp, XI=False):
    ''' Opencv packs the true octave, scale and layer inside kp.octave.
    This function computes the unpacking of that information.
    '''
    _octave = kp.octave
    octave = _octave&0xFF
    layer  = (_octave>>8)&0xFF
    if octave>=128:
        octave |= -128
    if octave>=0:
        scale = float(1/(1<<octave))
    else:
        scale = float(1<<-octave)

    if XI:
        yi = (_octave>>16)&0xFF
        xi = yi/255.0 - 0.5
        return octave, layer, scale, xi
    else:
        return octave, layer, scale

def packSIFTOctave(octave, layer, xi=0.0):
    po = octave&0xFF
    pl = (layer&0xFF)<<8
    pxi = round((xi + 0.5)*255)&0xFF
    pxi = pxi<<16
    return  po + pl + pxi

def DescRadius(kp, InPyr=False, SIFT=False):
    ''' Computes the Descriptor radius with respect to either an image
        in the pyramid or to the original image.
    '''
    factor = siftparams.new_radius_descr
    if SIFT:
        factor = siftparams.lambda_descr
    if InPyr:
        o, l, s = unpackSIFTOctave(kp)
        return( np.float32(kp.size*s*factor*0.5) )
    else:
        return( np.float32(kp.size*factor*0.5) )


def AngleDiff(a,b):
    ''' Computes the Angle Difference between a and b.
        0<=a,b<=360
    '''
    assert a>=0 and a<=360 and b>=0 and b<=360, 'a = '+str(a)+', b = '+str(b)
    anglediff = abs(a-b)% 360
    if anglediff > 180:
        anglediff = 360 - anglediff
    return anglediff



def FilterKPsinList(kp0,kp_list,maxdist = MaxSameKP_dist, maxangle = MaxSameKP_angle):
    ''' Filters out all keypoints in kp_list having angle differences and distances above some thresholds.
     Those comparisons should be made with restpect to the groundtruth image.
    '''
    idx = () # void tuple
    for i in range(0,np.size(kp_list)):
        dist = cv2.norm(kp0.pt,kp_list[i].pt)
        anglediff = AngleDiff( kp0.angle , kp_list[i].angle )
        if dist<maxdist and anglediff<maxangle:
            idx += tuple([i])
    return idx


def FindBestKPinLists(lambda1,lambda2, kp_list1, kp_list2):
    ''' Finds the best pair (i,j) such that kp_list1[i] equals kp_list2[j] in
        the groundtruth image. The groundtruth image was zoom-out by a factor lambda1
        for the image corresponding to kp_list1, and same goes for lambda2 and kp_list2.
    '''
    idx1 = None
    idx2 = None
    mindist = MaxSameKP_dist
    mindiffsizes = 10.0
    for i in range(0,np.size(kp_list1)):
        kp1 = kp_list1[i]
        size1 = DescRadius(kp1)*lambda1
        for j in range(0,np.size(kp_list2)):
            kp2 = kp_list2[j]
            size2 = DescRadius(kp2)*lambda2
            dist = cv2.norm(kp1.pt,kp2.pt)
            diffsizes = abs(size1 - size2)
            if diffsizes<mindiffsizes or (dist<mindist and diffsizes==mindiffsizes) :
                mindist = dist
                mindiffsizes = diffsizes
                # print(size1, size2, diffsizes)
                idx1 = i
                idx2 = j
    return idx1, idx2


def features_deepcopy(f):
    return [cv2.KeyPoint(x = k.pt[0], y = k.pt[1],
            _size = k.size, _angle = k.angle,
            _response = k.response, _octave = k.octave,
            _class_id = k.class_id) for k in f]


def Filter_Affine_In_Rect(kp_list, A, p_min, p_max, desc_list = None):
    ''' Filters out all descriptors in kp_list that do not lay inside the
    the parallelogram defined by the image of a rectangle by the affine transform A.
    The rectangle is defined by (p_min,p_max).
    '''
    desc_listing = False
    desc_list_in = []
    desc_pos = 0
    if type(desc_list) is np.ndarray:
        desc_listing = True
        desc_list_in = desc_list.copy()
    x1, y1 = p_min[:2]
    x2, y2 = p_max[:2]
    Ai = cv2.invertAffineTransform(A)
    kp_back = AffineKPcoor(kp_list,Ai)
    kp_list_in = []
    kp_list_out = []
    cyclic_corners = np.float32([[x1, y1], [x2, y1], [x2, y2], [x1, y2], [x1, y1]])
    cyclic_corners = AffineArrayCoor(cyclic_corners,A)
    for i in range(0,np.size(kp_back)):
        if kp_back[i].pt[0]>=x1 and kp_back[i].pt[0]<x2 and kp_back[i].pt[1]>=y1 and kp_back[i].pt[1]<y2:
            In = True
            r = DescRadius(kp_list[i])*1.4142
            for j in range(0,4):
                if r > dist_pt_to_line(kp_list[i].pt,cyclic_corners[j],cyclic_corners[j+1]):
                    In = False
            if In == True:
                if desc_listing:
                    desc_list_in[desc_pos,:]= desc_list[i,:]
                    desc_pos +=1
                kp_list_in.append(kp_list[i])
            else:
                kp_list_out.append(kp_list[i])
        else:
            kp_list_out.append(kp_list[i])
    if desc_listing:
        return kp_list_in, desc_list_in[:desc_pos,:], kp_list_out
    else:
        return kp_list_in, kp_list_out



def dist_pt_to_line(p,p1,p2):
    ''' Computes the distance of a point (p) to a line defined by two points (p1, p2). '''
    x0, y0 = np.float32(p[:2])
    x1, y1 = np.float32(p1[:2])
    x2, y2 = np.float32(p2[:2])
    dist = abs( (y2-y1)*x0 - (x2-x1)*y0 + x2*y1 - y2*x1 ) / np.sqrt( pow(y2-y1,2) + pow(x2-x1,2) )
    return dist


def PolarCoor_from_vector(p_source,p_arrow):
    ''' It computes the \rho and \theta such that
        \rho * exp( i * \theta ) = p_arrow-p_source
    '''
    p = np.array(p_arrow)- np.array(p_source)
    rho = np.linalg.norm(p)
    theta = 0
    if rho>0:
        theta = np.arctan2(p[1],p[0])
        theta = np.rad2deg(theta % (2 * np.pi))
    return  rho, theta


def ComposeAffineMaps(A_lhs,A_rhs):
    ''' Comutes the composition of affine maps:
        A = A_lhs ∘ A_rhs
    '''
    A = np.matmul(A_lhs[0:2,0:2],A_rhs)
    A[:,2] += A_lhs[:,2]
    return A


def AffineArrayCoor(arr,A):
    if type(arr) is list:
        arr = np.array(arr).reshape(-1,2)
    AA = A[0:2,0:2]
    Ab = A[:,2]
    arr_out = []
    for j in range(0,arr.shape[0]):
        arr_out.append(np.matmul(AA,np.array(arr[j,:])) + Ab )
    return np.array(arr_out)

def AffineKPcoor(kp_list,A, Pt_mod = True, Angle_mod = True):
    ''' Transforms information details on each kp_list keypoints by following
        the affine map A.
    '''
    kp_affine = features_deepcopy(kp_list)
    AA = A[0:2,0:2]
    Ab = A[:,2]
    for j in range(0,np.size(kp_affine)):
        newpt = tuple( np.matmul(AA,np.array(kp_list[j].pt)) + Ab)
        if Pt_mod:
            kp_affine[j].pt = newpt
        if Angle_mod:
            phi = np.deg2rad(kp_list[j].angle)
            s, c = np.sin(phi), np.cos(phi)
            R = np.float32([[c,-s], [ s, c]])
            p_arrow = np.matmul( R , [50.0, 0.0] ) + np.array(kp_list[j].pt)
            p_arrow = tuple( np.matmul(AA,p_arrow) + Ab)
            rho, kp_affine[j].angle =  PolarCoor_from_vector(newpt, p_arrow)
    return kp_affine


def affine_decomp2affine(vec):
    lambda_scale = vec[0]
    phi2 = vec[1]
    t = vec[2]
    phi1 = vec[3]

    s, c = np.sin(phi1), np.cos(phi1)
    R_phi1 = np.float32([[c,s], [ -s, c]])
    s, c = np.sin(phi2), np.cos(phi2)
    R_phi2 = np.float32([[c,s], [ -s, c]])

    A = lambda_scale * np.matmul(R_phi2, np.matmul(np.diag([t,1.0]),R_phi1) )
    if np.shape(vec)[0]==6:
        A = np.concatenate(( A, [[vec[4]], [vec[5]]] ), axis=1)
    return A


def affine_decomp(A0,doAssert=True):
    '''Decomposition of a 2x2 matrix A (whose det(A)>0) satisfying
        A = lambda*R_phi2*diag(t,1)*R_phi1.
        where lambda and t are scalars, and R_phi1, R_phi2 are rotations.
    '''
    epsilon = 0.0001
    A = A0[0:2,0:2]
    Adet = np.linalg.det(A)
    if doAssert:
        assert Adet>0

    if Adet>0:
        #   A = U * np.diag(s) * V
        U, s, V = np.linalg.svd(A, full_matrices=True)
        T = np.diag(s)
        K = np.float32([[-1, 0], [0, 1]])

        # K*D*K = D
        if ((np.linalg.norm(np.linalg.det(U)+1)<=epsilon) and (np.linalg.norm(np.linalg.det(V)+1)<=epsilon)):
            U = np.matmul(U,K)
            V = np.matmul(K,V)

         # Computing First Rotation
        phi1 = np.arctan2( V[0,1], V[0,0] )
        s, c = np.sin(phi1), np.cos(phi1)
        R_phi1 = np.float32([[c,s], [ -s, c]])

        # Computing Second Rotation
        phi2 = np.arctan2( U[0,1],U[0,0])
        s, c = np.sin(phi2), np.cos(phi2)
        R_phi2 = np.float32([[c,s], [ -s, c]])

        # Computing Tilt and Lambda
        lambda_scale = T[1,1]
        T[0,0]=T[0,0]/T[1,1]
        T[1,1]=1.0

        temp = lambda_scale*np.matmul( R_phi2 ,np.matmul(T,R_phi1) )

        # Couldnt decompose A
        if doAssert and np.linalg.norm(A - temp,'fro')>epsilon:
            print('Error: affine_decomp couldnt really decompose A')
            print(A0)
            print('----- end of A')

        rvec = [lambda_scale, phi2, T[0,0], phi1]
        if np.shape(A0)[1]==3:
            rvec = np.concatenate(( rvec, [A0[0,2], A0[1,2]] ))
    else:
        rvec = [1.0, 0.0, 1.0, 0.0, 0.0, 0.0]
    return rvec


def transition_tilt( Avec, Bvec ):
    ''' Computes the transition tilt between two affine maps as in [https://rdguez-mariano.github.io/pages/imas]
    Let
    A = lambda1 * R_phi1 * diag(t,1) * psi1
    B = lambda2 * R_phi2 * diag(s,1) * psi2
    then Avec and Bvec are respectively the affine_decomp of A and B
    '''
    t = Avec[2]
    psi1 = Avec[3]
    s = Bvec[2]
    psi2 = Bvec[3]
    cos_2 = pow( np.cos(psi1-psi2), 2.0)
    g = ( pow(t/s, 2.0) + 1.0 )*cos_2 + ( 1.0/pow(s, 2.0) + pow(t,2.0) )*( 1.0 - cos_2 )
    G = (s/t)*g/2.0
    tau = G + np.sqrt( pow(G,2.0) - 1.0 )
    return tau


def ComputeSIFTKeypoints(img, Desc = False):
    gray = []
    if len(img.shape)!=2:
        gray= cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    else:
        gray = img.view()

    sift = cv2.xfeatures2d.SIFT_create(
    nfeatures = siftparams.nfeatures,
    nOctaveLayers = siftparams.nOctaveLayers, contrastThreshold = siftparams.contrastThreshold,
    edgeThreshold = siftparams.edgeThreshold, sigma = siftparams.sigma
    )
    if Desc:
        kp, des = sift.detectAndCompute(gray,None)
        return kp, des
    else:
        kp = sift.detect(gray,None)
        return kp


def ComputePatches(kp_list,gpyr):
    ''' Computes the associated patch to each keypoint in kp_list.
        Returns:
        img_list - list of patches.
        A_list - lists of affine maps A such that A(BackgroundImage)*1_{[0,2r]x[0,2r]} = patch.
        Ai_list - list of the inverse of the above affine maps.

    '''
    img_list = []
    A_list = []
    Ai_list = []
    for i in range(0,np.size(kp_list)):
        kpt = kp_list[i]
        octave, layer, scale = unpackSIFTOctave(kpt)
        assert octave >= siftparams.firstOctave and layer <= siftparams.nOctaveLayers+2, 'octave = '+str(octave)+', layer = '+str(layer)
        # formula in opencv:  kpt.size = sigma*powf(2.f, (layer + xi) / nOctaveLayers)*(1 << octv)*2
        step = kpt.size*scale*0.5 # sigma*powf(2.f, (layer + xi) / nOctaveLayers)
        ptf = np.array(kpt.pt)*scale
        angle = 360.0 - kpt.angle
        if(np.abs(angle - 360.0) < siftparams.flt_epsilon):
            angle = 0.0

        img = gpyr[(octave - siftparams.firstOctave)*(siftparams.nOctaveLayers + 3) + layer]

        r = siftparams.new_radius_descr

        phi = np.deg2rad(angle)
        s, c = np.sin(phi), np.cos(phi)
        A = np.float32([[c,-s], [ s, c]]) / step
        Rptf = np.matmul(A,ptf)
        x = Rptf[0]-r
        y = Rptf[1]-r
        A = np.hstack([A, [[-x], [-y]]])

        dim = np.int32(2*r+1)
        img = cv2.warpAffine(img, A, (dim, dim), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_CONSTANT)
        #print('Octave =', octave,'; Layer =', layer, '; Scale =', scale,'; Angle =',angle)

        oA = np.float32([[1, 0, 0], [0, 1, 0]]) * scale
        A = ComposeAffineMaps(A,oA)
        Ai = cv2.invertAffineTransform(A)
        img_list.append(img.astype(np.float32))
        A_list.append(A)
        Ai_list.append(Ai)
    return img_list, A_list, Ai_list


def buildGaussianPyramid( base, LastOctave ):
    '''
    Computing the Gaussian Pyramid as in opencv SIFT
    '''
    if siftparams.graydesc and len(base.shape)!=2:
        base = cv2.cvtColor(base,cv2.COLOR_BGR2GRAY)
    else:
        base = base.copy()

    if siftparams.firstOctave<0:
        sig_diff = np.sqrt( max(siftparams.sigma * siftparams.sigma - siftparams.sift_init_sigma * siftparams.sift_init_sigma * 4, 0.01) )
        base = cv2.resize(base, (0, 0), fx=2, fy=2, interpolation=cv2.INTER_LINEAR_EXACT)

    rows, cols = base.shape[:2]

    nOctaves = np.round(np.log( np.float32(min( cols, rows ))) / np.log(2.0) - 2) - siftparams.firstOctave;
    nOctaves = min(nOctaves,LastOctave)
    nOctaves = np.int32(nOctaves)

    sig = ([siftparams.sigma])
    k = np.float32(pow( 2.0 , 1.0 / np.float32(siftparams.nOctaveLayers) ))

    for i in range(1,siftparams.nOctaveLayers + 3):
        sig_prev = pow(k, np.float32(i-1)) * siftparams.sigma
        sig_total = sig_prev*k
        sig += ([ np.sqrt(sig_total*sig_total - sig_prev*sig_prev) ])

    assert np.size(sig) == siftparams.nOctaveLayers + 3

    pyr = []
    for o in range(nOctaves):
        for i in range(siftparams.nOctaveLayers + 3):
            pyr.append([])

    assert len(pyr) == nOctaves*(siftparams.nOctaveLayers + 3)


    for o in range(nOctaves):
        for i in range(siftparams.nOctaveLayers + 3):
            if o == 0  and  i == 0:
                pyr[o*(siftparams.nOctaveLayers + 3) + i] = base.copy()
            elif i == 0:
                src = pyr[(o-1)*(siftparams.nOctaveLayers + 3) + siftparams.nOctaveLayers]
                srcrows, srccols = src.shape[:2]
                pyr[o*(siftparams.nOctaveLayers + 3) + i] = cv2.resize(src, (0, 0), fx=0.5, fy=0.5, interpolation=cv2.INTER_NEAREST)
            else:
                src = pyr[o*(siftparams.nOctaveLayers + 3) + i-1]
                pyr[o*(siftparams.nOctaveLayers + 3) + i] = cv2.GaussianBlur(src, (0, 0), sigmaX=sig[i], sigmaY=sig[i])
    return(pyr)

def load_image(path):
    img = cv2.imread(path, 1)
    # OpenCV loads images with color channels
    # in BGR order. So we need to reverse them
    return img[...,::-1]

def FirstOrderApprox_Homography(H0, X0=np.array([[0],[0],[1]])):
    ''' Computes the first order Taylor approximation (which is an affine map)
    of the Homography H0 centered at X0 (X0 is in homogeneous coordinates).
    '''
    H = H0.copy()
    col3 = np.matmul(H,X0)
    H[:,2] = col3.reshape(3)
    A = np.zeros((2,3), dtype = np.float32)
    A[0:2,0:2] = H[0:2,0:2]/H[2,2] - np.array([ H[0,2]*H[2,0:2], H[1,2]*H[2,0:2] ])/pow(H[2,2],2)
    A[:,2] = H[0:2,2]/H[2,2] - np.matmul( A[0:2,0:2], X0[0:2,0]/X0[2,0] )
    return A


def AffineFit(Xi,Yi):
    assert np.shape(Xi)[0]==np.shape(Yi)[0] and np.shape(Xi)[1]==2 and np.shape(Yi)[1]==2
    n = np.shape(Xi)[0]
    A = np.zeros((2*n,6),dtype=np.float32)
    b = np.zeros((2*n,1),dtype=np.float32)
    for i in range(0,n):
        A[2*i,0] = Xi[i,0]
        A[2*i,1] = Xi[i,1]
        A[2*i,2] = 1.0
        A[2*i+1,3] = Xi[i,0]
        A[2*i+1,4] = Xi[i,1]
        A[2*i+1,5] = 1.0

        b[2*i,0] = Yi[i,0]
        b[2*i+1,0] = Yi[i,1]
    result = np.linalg.lstsq(A,b,rcond=None)
    return result[0].reshape((2, 3))


def SquareOrderedPts(hs,ws,CV=True):
    # Patch starts from the origin
    ws = ws - 1
    hs = hs - 1
    if CV:
        return [
            cv2.KeyPoint(x = 0,  y =0, _size = 10, _angle = 0, _response = 1.0, _octave = 0, _class_id = 0),
            cv2.KeyPoint(x = ws, y =0, _size = 10, _angle = 0, _response = 1.0, _octave = 0, _class_id = 0),
            cv2.KeyPoint(x = ws, y =hs, _size = 10, _angle = 0, _response = 1.0, _octave = 0, _class_id = 0),
            cv2.KeyPoint(x = 0,  y =hs, _size = 10, _angle = 0, _response = 1.0, _octave = 0, _class_id = 0)
            ]
    else:
        # return np.float32([ [0,0], [ws+1,0], [ws+1, hs+1], [0,hs+1] ])
        return np.float32([ [0,0], [ws,0], [ws, hs], [0,hs] ])

def Flatten2Pts(vec):
    X = np.zeros( (np.int32(len(vec)/2), 2), np.float32)
    X[:,0] = vec[0::2]
    X[:,1] = vec[1::2]
    return X

def Pts2Flatten(X):
    h,w= np.shape(X)[:2]
    vec = np.zeros( (h*w), np.float32)
    vec[0::2] = X[:,0]
    vec[1::2] = X[:,1]
    return vec

def close_per(vec):
    return( np.array(tuple(vec)+tuple([vec[0]])) )

def Check_FirstThreadTouch(GA):
    for file in glob.glob(GA.save_path+"/"+str(GA.GAid)+".threadsdata"):
        if np.loadtxt(file)>0.5:
            return True
        else:
            return False
    Set_FirstThreadTouch(GA,False)
    return False


def Set_FirstThreadTouch(GA,value):
    np.savetxt(GA.save_path+"/"+str(GA.GAid)+".threadsdata", [value])

def get_big_epoch_number(GA):
    return np.loadtxt(GA.save_path+"/"+str(GA.GAid)+".big_epoch")

def set_big_epoch_number(GA,value):
    # print(GA.save_path+"/big_epoch  -> "+ str(value))
    np.savetxt(GA.save_path+"/"+str(GA.GAid)+".big_epoch", [value])


def get_process_memory():
    process = psutil.Process(os.getpid())
    return process.memory_info().rss

def HumanElapsedTime(start,end):
    hours, rem = divmod(end-start, 3600)
    minutes, seconds = divmod(rem, 60)
    # print("{:0>2}:{:0>2}:{:05.2f}".format(int(hours),int(minutes),seconds))
    return hours, minutes, seconds

def TouchDir(directory):
    ''' Creates a directory if it doesn't exist
    '''
    if not os.path.exists(directory):
        os.makedirs(directory)

def OnlyUniqueMatches(goodM, KPlistQ, KPlistT, SpatialThres=4):
    ''' Filter out non unique matches with less similarity score
    '''
    uniqueM = []
    doubleM = np.zeros(len(goodM),dtype=np.bool)
    for i in range(0,len(goodM)):
        if doubleM[i]:
            continue
        bestsim = goodM[i].distance
        bestidx = i
        for j in range(i+1,len(goodM)):
            if  ( cv2.norm(KPlistQ[goodM[i].queryIdx].pt, KPlistQ[goodM[j].queryIdx].pt) < SpatialThres \
            and   cv2.norm(KPlistT[goodM[i].trainIdx].pt, KPlistT[goodM[j].trainIdx].pt) < SpatialThres ):
                doubleM[j] = True
                if bestsim<goodM[j].distance:
                    bestidx = j
                    bestsim = goodM[j].distance
        uniqueM.append(goodM[bestidx])
    return uniqueM

class CPPbridge(object):
    def __init__(self,libDApath):
        self.libDA = ctypes.cdll.LoadLibrary(libDApath)
        self.MatcherPtr = 0
        self.last_i1_list = []
        self.last_i2_list = []

        self.libDA.GeometricFilter.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_float, ctypes.c_bool]
        self.libDA.GeometricFilter.restype = None

        self.libDA.GeometricFilterFromNodes.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_float, ctypes.c_bool]
        self.libDA.GeometricFilterFromNodes.restype = None
        self.libDA.ArrayOfFilteredMatches.argtypes = [ctypes.c_void_p,ctypes.c_void_p]
        self.libDA.GeometricFilterFromNodes.restype = None
        self.libDA.NumberOfFilteredMatches.argtypes = [ctypes.c_void_p]
        self.libDA.NumberOfFilteredMatches.restype = ctypes.c_int

        self.libDA.newMatcher.argtypes = [ctypes.c_int, ctypes.c_int, ctypes.c_float]
        self.libDA.newMatcher.restype = ctypes.c_void_p
        self.libDA.KnnMatcher.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int,ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int, ctypes.c_int]
        self.libDA.KnnMatcher.restype = None

        self.libDA.GetData_from_QueryNode.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p]
        self.libDA.GetData_from_QueryNode.restype = None
        self.libDA.GetQueryNodeLength.argtypes = [ctypes.c_void_p]
        self.libDA.GetQueryNodeLength.restype = ctypes.c_int

        self.libDA.LastQueryNode.argtypes = [ctypes.c_void_p]
        self.libDA.LastQueryNode.restype = ctypes.c_void_p
        self.libDA.FirstQueryNode.argtypes = [ctypes.c_void_p]
        self.libDA.FirstQueryNode.restype = ctypes.c_void_p
        self.libDA.NextQueryNode.argtypes = [ctypes.c_void_p, ctypes.c_void_p]
        self.libDA.NextQueryNode.restype = ctypes.c_void_p
        self.libDA.PrevQueryNode.argtypes = [ctypes.c_void_p, ctypes.c_void_p]
        self.libDA.PrevQueryNode.restype = ctypes.c_void_p

        self.libDA.FastMatCombi.argtypes = [ctypes.c_int, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
        ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int, ctypes.c_void_p, ctypes.c_void_p]
        self.libDA.FastMatCombi.restype = None

    def GeometricFilter(self, scr_pts, im1, dts_pts, im2, Filer = 'ORSA_H', precision = 10, verb=False):
        filercode=0
        if Filer=='ORSA_F':
            filercode=1
        N = int(len(scr_pts)/2)
        scr_pts = scr_pts.astype(ctypes.c_float)
        dts_pts = dts_pts.astype(ctypes.c_float)
        MatchMask = np.zeros(N, dtype = ctypes.c_bool)
        T = np.zeros(9, dtype = ctypes.c_float)
        floatp = ctypes.POINTER(ctypes.c_float)
        boolp = ctypes.POINTER(ctypes.c_bool)
        h1, w1 = im1.shape[:2]
        h2, w2 = im2.shape[:2]
        self.libDA.GeometricFilter(scr_pts.ctypes.data_as(floatp), dts_pts.ctypes.data_as(floatp),
                                    MatchMask.ctypes.data_as(boolp), T.ctypes.data_as(floatp),
                                    N, w1, h1, w2, h2, filercode, ctypes.c_float(precision), verb)
        return MatchMask.astype(np.bool), T.astype(np.float).reshape(3,3)

    def GeometricFilterFromMatcher(self, im1, im2, Filer = 'ORSA_H', precision=24, verb=False):
        filercode=0
        if Filer=='ORSA_F':
            filercode=1
        T = np.zeros(9, dtype = ctypes.c_float)
        floatp = ctypes.POINTER(ctypes.c_float)
        intp = ctypes.POINTER(ctypes.c_int)
        # boolp = ctypes.POINTER(ctypes.c_bool)
        h1, w1 = im1.shape[:2]
        h2, w2 = im2.shape[:2]
        self.libDA.GeometricFilterFromNodes(self.MatcherPtr, T.ctypes.data_as(floatp),
                                    w1, h1, w2, h2, filercode, ctypes.c_float(precision), verb)
              
        NFM = self.libDA.NumberOfFilteredMatches(self.MatcherPtr)
        FM = np.zeros(3*NFM, dtype = ctypes.c_int)
        self.libDA.ArrayOfFilteredMatches(self.MatcherPtr,FM.ctypes.data_as(intp))
        # print(NFM,FM)                
        Matches = [cv2.DMatch(FM[3*i],FM[3*i+1],FM[3*i+2]) for i in range(0,NFM)]

        return Matches, T.astype(np.float).reshape(3,3)

    def GetMatches_from_QueryNode(self, qn):
        N = self.libDA.GetQueryNodeLength(qn)
        if N>0:
            Query_idx = np.zeros(1, dtype = ctypes.c_int)
            Target_idxs = np.zeros(N, dtype = ctypes.c_int)
            simis = np.zeros(N, dtype = ctypes.c_float)
            floatp = ctypes.POINTER(ctypes.c_float)
            intp = ctypes.POINTER(ctypes.c_int)
            self.libDA.GetData_from_QueryNode(qn, Query_idx.ctypes.data_as(intp), Target_idxs.ctypes.data_as(intp), simis.ctypes.data_as(floatp))
            return [cv2.DMatch(Query_idx[0], Target_idxs[i], simis[i]) for i in range(0,N)]
        else:
            return []

    def FirstLast_QueryNodes(self):
        return self.libDA.FirstQueryNode(self.MatcherPtr), self.libDA.LastQueryNode(self.MatcherPtr)

    def NextQueryNode(self, qn):
        return self.libDA.NextQueryNode(self.MatcherPtr, qn)

    def PrevQueryNode(self, qn):
        return self.libDA.PrevQueryNode(self.MatcherPtr, qn)

    def KnnMatch(self,QKPlist,Qdesc, TKPlist, Tdesc, FastCode):
        Nq = ctypes.c_int(np.shape(Qdesc)[0])
        Nt = ctypes.c_int(np.shape(Tdesc)[0])
        Qkps = np.array([x for kp in QKPlist for x in kp.pt],dtype=ctypes.c_float)
        Tkps = np.array([x for kp in TKPlist for x in kp.pt],dtype=ctypes.c_float)        
        floatp = ctypes.POINTER(ctypes.c_float)
        Qdesc = Qdesc.ravel().astype(ctypes.c_float)
        Tdesc = Tdesc.ravel().astype(ctypes.c_float)
        QdescPtr = Qdesc.ctypes.data_as(floatp)
        TdescPtr = Tdesc.ctypes.data_as(floatp)
        QkpsPtr = Qkps.ctypes.data_as(floatp)
        TkpsPtr = Tkps.ctypes.data_as(floatp)
        
        self.libDA.KnnMatcher(self.MatcherPtr,QkpsPtr, QdescPtr, Nq, TkpsPtr, TdescPtr, Nt, ctypes.c_int(FastCode))

    def CreateMatcher(self,desc_dim, k=1, sim_thres=0.7):
        self.MatcherPtr = self.libDA.newMatcher(k,desc_dim,sim_thres)

    def PrepareForFastMatCombi(self,len_i_list):
        self.last_i1_list = -1*np.ones(shape=(len_i_list), dtype = ctypes.c_int)
        self.last_i2_list = -1*np.ones(shape=(len_i_list), dtype = ctypes.c_int)

    def FastMatCombi(self,bP, i_list, ps1, j_list, ps2, MemStepImg, MemStepBlock):
        intp = ctypes.POINTER(ctypes.c_int)
        floatp = ctypes.POINTER(ctypes.c_float)
        i1_list = i_list.ctypes.data_as(intp)
        i2_list = j_list.ctypes.data_as(intp)
        ps1p = ps1.ctypes.data_as(floatp)
        ps2p = ps2.ctypes.data_as(floatp)
        bPp = bP.ctypes.data_as(floatp)

        last_i1_listp = self.last_i1_list.ctypes.data_as(intp)
        last_i2_listp = self.last_i2_list.ctypes.data_as(intp)

        self.libDA.FastMatCombi(  ctypes.c_int(len(self.last_i1_list)), bPp,
         i1_list, i2_list, ps1p, ps2p, ctypes.c_int(MemStepImg), last_i1_listp, last_i2_listp )

        self.last_i1_list = i_list.copy()
        self.last_i2_list = j_list.copy()
