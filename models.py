#  VGG like network
from keras import layers
from keras.models import Model
import tensorflow as tf
from keras import backend as K


def create_model(input_shape, output_shape, model_name = 'DA_Pts_base', Norm='L2', resume = True, ResumeFile = ''):
    if model_name == 'AID_simCos_base':
        train_model, sim_type = AID_CreateModel(input_shape, desc_dim = 128, desc_between_0_1 = False, similarity = 'simCos')
        path2weights = 'model-data/'
    elif model_name == 'AID_simCos_128Desc_1FC':
        train_model, sim_type = AID_CreateModel(input_shape, desc_dim = 128, B5_FC1_neurons=0, desc_between_0_1 = False, similarity = 'simCos')
        path2weights = 'model-data/'
    elif model_name == 'AID_simCos_128Desc_1FC_dropout':
        train_model, sim_type = AID_CreateModel(input_shape, desc_dim = 128, B5_FC1_neurons=0, Spatial_Dropout=True, desc_between_0_1 = False, similarity = 'simCos')
        path2weights = 'model-data/'
    elif model_name == 'AID_simCos_BigDesc':
        train_model, sim_type = AID_CreateModel(input_shape, BigDesc = True, similarity = 'simCos')
        path2weights = 'model-data/model.AID_simCos_BigDesc.hdf5'
    elif model_name == 'AID_simCos_BigDesc_dropout':
        train_model, sim_type = AID_CreateModel(input_shape, BigDesc = True, Spatial_Dropout=True, similarity = 'simCos')
        path2weights = 'model-data/model.AID_simCos_BigDesc_dropout.hdf5'
    elif model_name == 'AID_simCos_between01':
        train_model, sim_type = AID_CreateModel(input_shape, desc_dim = 128, desc_between_0_1 = True, similarity = 'simCos')
        path2weights = 'model-data/'
    elif model_name == 'AID_simCos_2xdescdim_between01':
        train_model, sim_type = AID_CreateModel(input_shape, desc_dim = 256, desc_between_0_1 = True, similarity = 'simCos')
        path2weights = 'model-data/'
    elif model_name == 'AID_simCos_2xdescdim': # this one was wrong all the time
        train_model, sim_type = AID_CreateModel(input_shape, desc_dim = 256, desc_between_0_1 = False, similarity = 'simCos')
        path2weights = 'model-data/'
    elif model_name == 'AID_simFC_diff': # became nan to soon
        train_model, sim_type = AID_CreateModel(input_shape, desc_dim = 128, desc_between_0_1 = False, similarity = 'simFC_diff')
        path2weights = 'model-data/'
    elif model_name == 'AID_simFC_concat':
        train_model, sim_type = AID_CreateModel(input_shape, desc_dim = 128, desc_between_0_1 = False, similarity = 'simFC_concat')
        path2weights = 'model-data/'
    elif model_name == 'AID_simFC_concat_BigDesc':
        train_model, sim_type = AID_CreateModel(input_shape, BigDesc = True, similarity = 'simFC_concat_BigDesc')
        path2weights = 'model-data/'
    else:
        train_model = None
        print('Error: '+model_name+" does not exist !")
        resume = False

    if ResumeFile!='':
        path2weights = ResumeFile
    if resume:
        train_model.load_weights(path2weights)
        print(path2weights)
    if model_name[0:3] == 'AID':
        return train_model, sim_type
    else:
        return train_model

def AID_CreateModel(input_shape, alpha_hinge = 0.2, Spatial_Dropout = False, BN = True, B5_FC1_neurons = 1024, similarity = 'simCos', desc_dim = 128, desc_between_0_1 = False, BigDesc=False, verbose=True):

    # descriptor model
    in_desc = layers.Input(shape=input_shape, name='input_patches')

    x = layers.Conv2D(64, (3, 3),
    padding='same',
    name='block1_conv1')(in_desc)
    if BN:
        x = layers.BatchNormalization(name='block1_BN1')(x)
    x = layers.Activation('relu', name='block1_relu1')(x)

    x = layers.Conv2D(64, (3, 3),
    padding='same',
    name='block1_conv2')(x)
    if BN:
        x = layers.BatchNormalization(name='block1_BN2')(x)
    x = layers.Activation('relu', name='block1_relu2')(x)

    x = layers.MaxPooling2D((2, 2), strides=(2, 2), name='block1_pool')(x)


    # Block 2
    x = layers.Conv2D(64, (3, 3),
    padding='same',
    name='block2_conv1')(x)
    if BN:
        x = layers.BatchNormalization(name='block2_BN1')(x)
    x = layers.Activation('relu', name='block2_relu1')(x)

    x = layers.Conv2D(64, (3, 3),
    padding='same',
    name='block2_conv2')(x)
    if BN:
        x = layers.BatchNormalization(name='block2_BN2')(x)
    x = layers.Activation('relu', name='block2_relu2')(x)

    x = layers.MaxPooling2D((2, 2), strides=(2, 2), name='block2_pool')(x)


    # Block 3
    x = layers.Conv2D(128, (3, 3),
    padding='same',
    name='block3_conv1')(x)
    if BN:
        x = layers.BatchNormalization(name='block3_BN1')(x)
    x = layers.Activation('relu', name='block3_relu1')(x)

    x = layers.Conv2D(128, (3, 3),
    padding='same',
    name='block3_conv2')(x)
    if BN:
        x = layers.BatchNormalization(name='block3_BN2')(x)
    x = layers.Activation('relu', name='block3_relu2')(x)

    x = layers.MaxPooling2D((2, 2), strides=(2, 2), name='block3_pool')(x)


    # Block 4
    x = layers.Conv2D(128, (3, 3),
    padding='same',
    name='block4_conv1')(x)
    if BN:
        x = layers.BatchNormalization(name='block4_BN1')(x)
    x = layers.Activation('relu', name='block4_relu1')(x)

    x = layers.Conv2D(128, (3, 3),
    padding='same',
    name='block4_conv2')(x)


    if BigDesc==False and BN:
        x = layers.BatchNormalization(name='block4_BN2')(x)

    if Spatial_Dropout:
        x = layers.SpatialDropout2D(p= 0.5,name='block4_Dropout1')(x)

    if BigDesc==False:
        x = layers.Activation('relu', name='block4_relu2')(x)


    # Block 5
    x = layers.Flatten(name='block5_flatten1')(x)

    if BigDesc==False:
        if B5_FC1_neurons>0:
            x = layers.Dense(B5_FC1_neurons,activation='relu',name='block5_FC1')(x)

        if desc_between_0_1:
            x = layers.Dense(desc_dim,activation='sigmoid',name='block5_FC2')(x)
        else:
            x = layers.Dense(desc_dim,name='block5_FC2')(x)

    desc_model = Model(in_desc, x, name='aff_desc')


    # similarity model
    if similarity[0:5] == 'simFC':
        if similarity[5:] == '_concat' or similarity[5:] == '_concat_BigDesc':
            sim_type = 'concat'
            desc_dim = 2*desc_model.output_shape[1]
        elif similarity[5:] == '_diff':
            sim_type = 'diff'
        # 2 siamese network
        in_desc1 = layers.Input(shape=input_shape, name='input_patches1')
        in_desc2 = layers.Input(shape=input_shape, name='input_patches2')
        emb_1 = desc_model(in_desc1)
        emb_2 = desc_model(in_desc2)

        # Similarity model
        in_sim = layers.Input(shape=(desc_dim,), name='input_diff_desc')
        x = layers.Dense(64,activation='relu',name='block1_FC1')(in_sim)
        x = layers.Dense(32,activation='relu',name='block1_FC2')(x)
        x = layers.Dense(1,activation='sigmoid',name='block1_FC3')(x)
        sim_model = Model(in_sim, x, name='sim')

        if sim_type == 'concat':
            x = layers.Concatenate(name='Concat')([emb_1, emb_2])
        else:
            x = layers.Subtract(name='Subtract')([emb_1, emb_2])

        out_net = sim_model(x)

        # Groundtruth Model
        in_GT = layers.Input(shape=(1,),name='input_GroundTruth')
        GT_model = Model(in_GT, in_GT, name='GroundTruth')
        out_GT = GT_model(in_GT)

        class TopLossLayerClass(layers.Layer):
            def __init__(self, **kwargs):
                super(TopLossLayerClass, self).__init__(**kwargs)
            def call(self, inputs):
                #out_net,  out_GT = inputs
                s,  t = inputs # t=1 -> Positive class, t=0 -> Negative class
                loss =K.sum(  t*K.log(s) + (1-t)*K.log(1-s) )
                self.add_loss(loss)
                return loss
        TopLossLayer_obj = TopLossLayerClass(name='TopLossLayer')

        TopLossLayer = TopLossLayer_obj([out_net, out_GT ])
        train_model = Model([in_desc1, in_desc2, in_GT], TopLossLayer,name='TrainModel')
    elif similarity == 'simCos': # hinge loss
        # Similarity model
        desc_dim = desc_model.output_shape[1]
        in_sim1 = layers.Input(shape=(desc_dim,), name='input_desc1')
        in_sim2 = layers.Input(shape=(desc_dim,), name='input_desc2')
        x = layers.Dot(axes=1, normalize=True, name='CosineProximity')([in_sim1,in_sim2]) # cosine proximity
        sim_model = Model([in_sim1,in_sim2], x, name='sim')

        # 3 siamese networks
        in_desc1 = layers.Input(shape=input_shape, name='input_patches_anchor')
        in_desc2 = layers.Input(shape=input_shape, name='input_patches_positive')
        in_desc3 = layers.Input(shape=input_shape, name='input_patches_negative')
        emb_1 = desc_model(in_desc1)
        emb_2 = desc_model(in_desc2)
        emb_3 = desc_model(in_desc3)
        sim_type = 'inlist'
        out_net_positive = sim_model([emb_1, emb_2])
        out_net_negative = sim_model([emb_1, emb_3])

        class TopLossLayerClass(layers.Layer):
            def __init__(self, alpha = 0.2, **kwargs):
                self.alpha = alpha
                super(TopLossLayerClass, self).__init__(**kwargs)
            def call(self, inputs):
                out_net_positive, out_net_negative = inputs
                # Hinge loss computation
                loss = K.sum( K.maximum(out_net_negative - out_net_positive + self.alpha, 0) )#,axis=0)
                self.add_loss(loss)
                return loss
        TopLossLayer_obj = TopLossLayerClass(name='TopLossLayer', alpha = alpha_hinge)
        TopLossLayer = TopLossLayer_obj([out_net_positive, out_net_negative ])
        train_model = Model([in_desc1, in_desc2, in_desc3], TopLossLayer,name='TrainModel')


    if verbose:
        print('\n\n-------> The network architecture for the affine descriptor computation !')
        desc_model.summary()
        print('\n\n-------> The network architecture for the similarity computation !')
        sim_model.summary()
        print('\n\n-------> Train model connections')
        train_model.summary()
    return train_model, sim_type
