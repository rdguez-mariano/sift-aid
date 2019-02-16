MODEL_NAME = 'AID_simCos_BigDesc_dropout'
DegMax = 60
Debug = True
Parallel = False
ConstrastSimu = True # if True it randomly simulates contrast changes for each patch
DoBigEpochs = True

batch_number = 32
N_epochs = 5000
steps_epoch=100
NeededData = batch_number * N_epochs * steps_epoch + 1
SHOW_TB_weights = False # Show Net-weights info in TensorBoard


if MODEL_NAME[0:10]=="AID_simCos":
    TripleLoss = True
    NORM = 'hinge'
else:
    TripleLoss = False
    NORM = 'cross-entropy'


# When default GPU is being used... prepare to use a second one
# import os
# os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"   # see issue #152
# os.environ["CUDA_VISIBLE_DEVICES"]="0"


from library import *
from acc_test_library import *
import numpy as np
import time
import random
import cv2


def ProcessData(GA, stacked_patches, groundtruth_pts):
    if ConstrastSimu:
        channels = np.int32(np.shape(stacked_patches)[2]/2)
        val1 = random.uniform(1/3, 3)
        val2 = random.uniform(1/3, 3)
        for i in range(channels):
            stacked_patches[:,:,i] = np.power(stacked_patches[:,:,i],val1)
            stacked_patches[:,:,channels+i] = np.power(stacked_patches[:,:,channels+i],val2)
    return stacked_patches, groundtruth_pts #if ConstrastSimu==False -> Identity



GAval = GenAffine("./imgs-val/", save_path = "./db-gen-val-"+str(DegMax)+"/", DoBigEpochs = DoBigEpochs, tmax = DegMax)
GAtrain = GenAffine("./imgs-train/", save_path = "./db-gen-train-"+str(DegMax)+"/", DoBigEpochs = DoBigEpochs, tmax = DegMax)

Set_FirstThreadTouch(GAval,False)
Set_FirstThreadTouch(GAtrain,False)
stacked_patches, groundtruth_pts = GAtrain.gen_affine_patches()
stacked_patches, groundtruth_pts = ProcessData(GAtrain, stacked_patches, groundtruth_pts)



def affine_generator(GA, batch_num=32, Force2Gen=False, ForceFast=False):
    P_list = []
    GT_list = []
    FastThread = False
    t2sleep = 2*random.random()
    time.sleep(t2sleep)

    assert Force2Gen==False or ForceFast==False
    if ForceFast:
        FastThread = True

    if Force2Gen==False and Check_FirstThreadTouch(GA)==False:
        print("Fast Thread Created ! Needs "+str(NeededData)+" generated data")
        Set_FirstThreadTouch(GA,True)
        FastThread = True

    while True:
        if FastThread and ForceFast==False:
            GA.ScatteredGenData_2_BlockData() # it will be really done every 30 minutes

        stacked_patches, groundtruth_pts = [], []
        if FastThread and Force2Gen==False:
            stacked_patches, groundtruth_pts = GA.Fast_gen_affine_patches()
        else:
            stacked_patches, groundtruth_pts = GA.gen_affine_patches()
        stacked_patches, groundtruth_pts = ProcessData(GA, stacked_patches, groundtruth_pts)

        Pa = stacked_patches[:,:,0]
        Pp = stacked_patches[:,:,1]

        if FastThread and Force2Gen==False:
            stacked_patches, groundtruth_pts = GA.Fast_gen_affine_patches()
        else:
            stacked_patches, groundtruth_pts = GA.gen_affine_patches()
        stacked_patches, groundtruth_pts = ProcessData(GA, stacked_patches, groundtruth_pts)

        Pn = stacked_patches[:,:,0]

        vgg_input_shape = np.shape(Pa)
        vgg_output_shape = np.shape([1])
        bPshape = tuple([batch_num]) + tuple(vgg_input_shape) + tuple([1])
        bGTshape = tuple([batch_num]) + tuple(vgg_output_shape)

        bP1 = np.zeros(shape=bPshape)
        bP2 = np.zeros(shape=bPshape)
        bP3 = np.zeros(shape=bPshape)
        bGT = np.zeros(shape=bGTshape, dtype = np.float32)

        if NORM=='hinge':
            bP1[0,:,:,0] = Pa
            bP2[0,:,:,0] = Pp
            bP3[0,:,:,0] = Pn
        else:
            bP1[0,:,:,0] = Pa
            bP2[0,:,:,0] = Pp
            bGT[0,0] = 1.0


        for i in range(1,batch_num):
            if FastThread and Force2Gen==False:
                stacked_patches, groundtruth_pts = GA.Fast_gen_affine_patches()
            else:
                stacked_patches, groundtruth_pts = GA.gen_affine_patches()
            stacked_patches, groundtruth_pts = ProcessData(GA, stacked_patches, groundtruth_pts)

            Pa = stacked_patches[:,:,0]
            Pp = stacked_patches[:,:,1]

            if FastThread and Force2Gen==False:
                stacked_patches, groundtruth_pts = GA.Fast_gen_affine_patches()
            else:
                stacked_patches, groundtruth_pts = GA.gen_affine_patches()
            stacked_patches, groundtruth_pts = ProcessData(GA, stacked_patches, groundtruth_pts)

            Pn = stacked_patches[:,:,0]

            if NORM=='hinge':
                bP1[i,:,:,0] = Pa
                bP2[i,:,:,0] = Pp
                bP3[i,:,:,0] = Pn
            else:
                if random.randint(0,1)>0.5:
                    bP1[i,:,:,0] = Pa
                    bP2[i,:,:,0] = Pp
                    bGT[i,0] = 1.0
                else:
                    bP1[i,:,:,0] = Pa
                    bP2[i,:,:,0] = Pn
                    bGT[i,0] = 0.0

        # print('These numbers should not repeat in other lines: '+ str(bP[0,0,0,0])+" "+str(bP[-1,0,0,0]))
        # print('Gen batch: '+str(np.shape(bP))+', '+str(np.shape(bGT)))
        if NORM=='hinge':
            yield [bP1, bP2, bP3], None
        else:
            yield [bP1, bP2, bGT], None


#  VGG like network
from keras import layers
from keras.models import Model
import tensorflow as tf
from keras.backend.tensorflow_backend import set_session
config = tf.ConfigProto(allow_soft_placement=True)
#, device_count = {'CPU' : 1, 'GPU' : 1})
config.gpu_options.per_process_gpu_memory_fraction = 0.1
set_session(tf.Session(config=config))


from models import *
vgg_input_shape = np.shape(stacked_patches)[0:2] + tuple([1])
train_model, sim_type = create_model(vgg_input_shape, None, model_name = MODEL_NAME, Norm=NORM, resume = False)




# ---> TRAIN NETWORK
import math
import scipy.special
import random
from sklearn.manifold import TSNE, MDS
from sklearn.metrics import f1_score, accuracy_score
from keras.callbacks import TerminateOnNaN, ModelCheckpoint, TensorBoard, LambdaCallback, ReduceLROnPlateau
import os
from shutil import copyfile
import matplotlib.pyplot as plt
plt.switch_backend('agg')
#modified from http://seoulai.com/2018/02/06/keras-and-tensorboard.html
class TensorboardKeras(object):
    def __init__(self, model, log_dir, GAval, GAtrain, static_val_num=500):
        self.model = model
        self.log_dir = log_dir
        self.session = K.get_session()
        self.lastloss = float('nan')
        self.lastvalloss = float('nan')
        self.GAval = GAval
        self.GAtrain = GAtrain

        self.static_val_num = static_val_num
        self.acc_data_Pa = []
        self.acc_data_Pp = []
        self.acc_data_names = []
        self.lastacc = 0
        self.TKid = random.randint(0,1000)

        self.P1_pos, self.P2_pos, self.P1_neg, self.P2_neg = [], [], [], []

        self.acc_TP_ph = tf.placeholder(shape=(), dtype=tf.float32)
        tf.summary.scalar('accuracy/TruePositives', self.acc_TP_ph)

        self.acc_TN_ph = tf.placeholder(shape=(), dtype=tf.float32)
        tf.summary.scalar('accuracy/TrueNegatives', self.acc_TN_ph)

        self.lr_ph = tf.placeholder(shape=(), dtype=tf.float32)
        tf.summary.scalar('Learning_rate', self.lr_ph)

        self.big_epoch = tf.placeholder(shape=(), dtype=tf.float32)
        tf.summary.scalar('Big_Epoch', self.big_epoch)

        self.val_loss_ph = tf.placeholder(shape=(), dtype=tf.float32)
        tf.summary.scalar('losses/validation', self.val_loss_ph)

        self.train_loss_ph = tf.placeholder(dtype=tf.float32)
        tf.summary.scalar('losses/training', self.train_loss_ph)

        # self.sift = cv2.xfeatures2d.SIFT_create( nfeatures = siftparams.nfeatures,
        # nOctaveLayers = siftparams.nOctaveLayers, contrastThreshold = siftparams.contrastThreshold,
        # edgeThreshold = siftparams.edgeThreshold, sigma = siftparams.sigma)

        self.global_acc_holder = tf.placeholder(dtype=tf.float32)
        tf.summary.scalar('accuracy/_GLOBAL_', self.global_acc_holder)

        self.acc_test_holder = []
        for file in glob.glob('./acc-test/*.txt'):
            self.acc_data_names.append( os.path.basename(file)[:-4] )
            i = len(self.acc_data_names) - 1
            pathway = './acc-test/' + self.acc_data_names[i]
            asift_KPlist1, patches1, GT_Avec_list, asift_KPlist2, patches2 = load_acc_test_data(pathway)
            Pa = np.zeros(shape=tuple([len(patches1)])+tuple(np.shape(patches1)[1:])+tuple([1]),dtype=np.float32)
            Pp = np.zeros(shape=tuple([len(patches1)])+tuple(np.shape(patches1)[1:])+tuple([1]),dtype=np.float32)
            for k in range(0,len(patches1)):
                Pa[k,:,:,0] = patches1[k][:,:]/self.GAval.imgdivfactor
                Pp[k,:,:,0] = patches2[k][:,:]/self.GAval.imgdivfactor
            self.acc_data_Pa.append( Pa )
            self.acc_data_Pp.append( Pp )

            self.acc_test_holder.append(tf.placeholder(dtype=tf.float32))
            tf.summary.scalar('accuracy/'+self.acc_data_names[i], self.acc_test_holder[i])

        if SHOW_TB_weights:
            l = np.shape(self.model.get_layer("aff_desc").get_weights())[0]
            self.weightsholder = []
            for i in range(0,l):
                self.weightsholder.append(tf.placeholder(dtype=tf.float32))
                self.variable_summaries(self.weightsholder[i], 'weights/'+repr(i).zfill(3)+'-layer')


        self.merged = tf.summary.merge_all()
        self.writer = tf.summary.FileWriter(self.log_dir)

        copyfile(os.path.realpath(__file__), self.log_dir+"/"+os.path.basename(__file__))

    def variable_summaries(self,var,name):
        """Attach a lot of summaries to a Tensor (for TensorBoard visualization)."""
        with tf.name_scope(name):
            mean = tf.reduce_mean(var)
            tf.summary.scalar('mean', mean)
            tf.summary.scalar('max', tf.reduce_max(var))
            tf.summary.scalar('min', tf.reduce_min(var))
            tf.summary.histogram('histogram', var)
            stddev = tf.sqrt(tf.reduce_mean(tf.square(var - mean)))
            tf.summary.scalar('stddev', stddev)

    def _get_lr(self):
        return K.eval(self.model.optimizer.lr)

    def _get_weights(self,wpos):
        return self.model.get_layer("aff_desc").get_weights()[wpos]

    def on_epoch_end(self, epoch, logs):
        self.lastloss = np.ravel(logs['loss'])[0]
        self.lastvalloss = np.ravel(logs['val_loss'])[0]

    def on_epoch_begin(self, epoch, logs):
        for d in affine_generator(self.GAval, batch_num=self.static_val_num, ForceFast=True):
            if TripleLoss: #
                self.P1_pos = d[0][0]
                self.P2_pos = d[0][1]
                self.P1_neg = d[0][0]
                self.P2_neg = d[0][2]
            else:
                lpos, lneg = 0, 0
                for i in range(0,len(d[0][2])):
                    if d[0][2][i]>0.5:
                        lpos +=1
                    else:
                        lneg +=1
                self.P1_pos = np.zeros(shape=tuple([lpos])+tuple(np.shape(d[0][0])[1:]), dtype=np.float32)
                self.P2_pos = np.zeros(shape=tuple([lpos])+tuple(np.shape(d[0][0])[1:]), dtype=np.float32)
                self.P1_neg = np.zeros(shape=tuple([lneg])+tuple(np.shape(d[0][0])[1:]), dtype=np.float32)
                self.P2_neg = np.zeros(shape=tuple([lneg])+tuple(np.shape(d[0][0])[1:]), dtype=np.float32)
                i_p, i_n = 0, 0
                for i in range(0,len(d[0][2])):
                    if d[0][2][i]>0.5:
                        self.P1_pos[i_p,:,:,:] = d[0][0][i,:,:,:]
                        self.P2_pos[i_p,:,:,:] = d[0][1][i,:,:,:]
                        i_p += 1
                    else:
                        self.P1_neg[i_n,:,:,:] = d[0][0][i,:,:,:]
                        self.P2_neg[i_n,:,:,:] = d[0][1][i,:,:,:]
                        i_n += 1
            break
        emb_1_pos = self.model.get_layer("aff_desc").predict(self.P1_pos)
        emb_2_pos = self.model.get_layer("aff_desc").predict(self.P2_pos)
        emb_1_neg = self.model.get_layer("aff_desc").predict(self.P1_neg)
        emb_2_neg = self.model.get_layer("aff_desc").predict(self.P2_neg)
        if sim_type=='inlist':
            acc_pos = np.sum( self.model.get_layer("sim").predict([emb_1_pos, emb_2_pos]) )/np.shape(emb_1_pos)[0]
            acc_neg = np.sum( 1 - self.model.get_layer("sim").predict([emb_1_neg,emb_2_neg]) )/np.shape(emb_1_neg)[0]
        elif sim_type=='diff':
            acc_pos = np.sum( self.model.get_layer("sim").predict([emb_1_pos-emb_2_pos]) )/np.shape(emb_1_pos)[0]
            acc_neg = np.sum( 1 - self.model.get_layer("sim").predict([emb_1_neg-emb_2_neg]) )/np.shape(emb_1_neg)[0]
        elif sim_type=='concat':
            acc_pos = np.sum( self.model.get_layer("sim").predict(np.concatenate((emb_1_pos,emb_2_pos),axis=-1)) )/np.shape(emb_1_pos)[0]
            acc_neg = np.sum( 1 - self.model.get_layer("sim").predict(np.concatenate((emb_1_neg,emb_2_neg),axis=-1)) )/np.shape(emb_1_neg)[0]

        my_dict = {
                                                self.lr_ph: self._get_lr(),
                                                self.acc_TP_ph: acc_pos,
                                                self.acc_TN_ph: acc_neg,
                                                self.val_loss_ph: self.lastvalloss,
                                                self.big_epoch: get_big_epoch_number(self.GAtrain),
                                                self.train_loss_ph: self.lastloss,
                                                }
        if SHOW_TB_weights:
            l = np.shape(self.model.get_layer("aff_desc").get_weights())[0]
            for i in range(0,l):
                my_dict.update({self.weightsholder[i]: self._get_weights(i)})

        RealAccPos = []
        acc = 0.0
        for i in range(0,len(self.acc_data_Pa)):
            emb_1 = self.model.get_layer("aff_desc").predict(self.acc_data_Pa[i])
            emb_2 = self.model.get_layer("aff_desc").predict(self.acc_data_Pp[i])
            if sim_type=='inlist':
                acc = np.sum( self.model.get_layer("sim").predict([emb_1,emb_2]) )/np.shape(self.acc_data_Pa[i])[0]
            elif sim_type=='diff':
                acc = np.sum( self.model.get_layer("sim").predict([emb_1-emb_2]) )/np.shape(self.acc_data_Pa[i])[0]
            RealAccPos.append( acc )
            my_dict.update({self.acc_test_holder[i]: acc})
        thisacc = np.mean(np.array(RealAccPos))

        if (acc_pos+acc_neg) > self.lastacc:
            self.lastacc = acc_pos+acc_neg
            self.model.save(self.log_dir+"/model.ckpt.max_acc.hdf5")

        my_dict.update({self.global_acc_holder: thisacc})

        summary = self.session.run(self.merged,
                                   feed_dict=my_dict)

        self.writer.add_summary(summary, epoch)
        self.writer.flush()

    def on_epoch_end_cb(self):
        return LambdaCallback(on_epoch_end=lambda epoch, logs:
                                           self.on_epoch_end(epoch, logs))



from datetime import datetime

ts = datetime.now().strftime("%d-%m-%Y_%H:%M:%S")
log_path = "./summaries/" + MODEL_NAME + "_" + NORM + "_-_" + str(DegMax) + "deg_-_" + ts
tensorboard = TensorBoard(log_dir=log_path,
    write_graph=True, #This eats a lot of space. Enable with caution!
    #histogram_freq = 1,
    write_images=True,
    batch_size = 1,
    write_grads=True)

reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=25, verbose=1, mode='auto', cooldown=0, min_lr=0)

import keras
train_model.compile(loss=None, optimizer=keras.optimizers.Adam(lr=0.00001))
# loss_model_saver =  ModelCheckpoint(log_path + "/model.ckpt.min_loss.{epoch:04d}-{loss:.6f}.hdf5", monitor='loss', period=1, save_best_only=True)
loss_model_saver =  ModelCheckpoint(log_path + "/model.ckpt.min_loss.hdf5", monitor='loss', mode='min', period=1, save_best_only=True)
val_model_saver =  ModelCheckpoint(log_path + "/model.ckpt.min_val_loss.hdf5", monitor='val_loss', mode='min', period=1, save_best_only=True)
#load_metadata_from_facescrub('facescrub_db')
tboardkeras = TensorboardKeras(model=train_model, log_dir=log_path, GAval = GAval, GAtrain = GAtrain)
#on_epoch_begin or on_epoch_end
miscallbacks = [LambdaCallback(on_epoch_begin=lambda epoch, logs: tboardkeras.on_epoch_begin(epoch, logs),
                               on_epoch_end=lambda epoch, logs: tboardkeras.on_epoch_end(epoch, logs)),
                               tensorboard, TerminateOnNaN(), val_model_saver, loss_model_saver]#, reduce_lr]

Set_FirstThreadTouch(GAval,False)
Set_FirstThreadTouch(GAtrain,False)

if Debug:
    train_model.fit_generator(generator=affine_generator(GA=GAtrain,batch_num=2,ForceFast=True),
    validation_data=affine_generator(GA=GAval,batch_num=2,ForceFast=True), validation_steps=1,
    epochs=3, steps_per_epoch=2, callbacks = miscallbacks)
else:
    if Parallel:
        train_model.fit_generator(generator=affine_generator(GA=GAtrain,batch_num=batch_number,Force2Gen=True),
        validation_data=affine_generator(GA=GAval,batch_num=batch_number,Force2Gen=True), validation_steps=steps_epoch,
        epochs=N_epochs, steps_per_epoch=steps_epoch, callbacks = miscallbacks,
        max_queue_size=10,
        workers=8, use_multiprocessing=True)
    else:
        train_model.fit_generator(generator=affine_generator(GA=GAtrain,batch_num=batch_number,ForceFast=True),
        validation_data=affine_generator(GA=GAval,batch_num=batch_number,ForceFast=True), validation_steps=np.int32(steps_epoch/2),
        epochs=N_epochs, steps_per_epoch=steps_epoch, callbacks = miscallbacks)
