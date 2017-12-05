from __future__ import print_function
from __future__ import absolute_import
from __future__ import division

import warnings

from keras.models import Model
from keras.layers import Flatten, Dense, Input, Conv2D, MaxPooling2D, Dropout
from keras.layers import GlobalAveragePooling2D, GlobalMaxPooling2D, TimeDistributed
from keras.engine.topology import get_source_inputs
from keras.utils import layer_utils
from keras.utils.data_utils import get_file
from keras import backend as K
from keras_frcnn.RoiPoolingConv import RoiPoolingConv


"""
def get_weight_path():
    if K.image_dim_ordering() == 'th':
        print('pretrained weights not available for VGG with theano backend')
        return
    else:
        return 'vgg16_weights_tf_dim_ordering_tf_kernels.h5'
"""


def img_length_calc_function(C, width, height):
    def get_output_length(input_length):
        return input_length / C.rpn_stride

    return get_output_length(width), get_output_length(height)    


def nn_base(input_tensor=None, trainable=False):
    """
    # Determine proper input shape
    if K.image_dim_ordering() == 'th':
        input_shape = (3, None, None)
    else:
        input_shape = (None, None, 3)

    if input_tensor is None:
        img_input = Input(shape=input_shape)
    else:
        if not K.is_keras_tensor(input_tensor):
            img_input = Input(tensor=input_tensor, shape=input_shape)
        else:
            img_input = input_tensor

    if K.image_dim_ordering() == 'tf':
        bn_axis = 3
    else:
        bn_axis = 1

    # Block 1
    x = Conv2D(64, (3, 3), activation='relu', padding='same', name='block1_conv1')(img_input)
    x = Conv2D(64, (3, 3), activation='relu', padding='same', name='block1_conv2')(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block1_pool')(x)

    # Block 2
    x = Conv2D(128, (3, 3), activation='relu', padding='same', name='block2_conv1')(x)
    x = Conv2D(128, (3, 3), activation='relu', padding='same', name='block2_conv2')(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block2_pool')(x)

    # Block 3
    x = Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv1')(x)
    x = Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv2')(x)
    x = Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv3')(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block3_pool')(x)

    # Block 4
    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv1')(x)
    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv2')(x)
    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv3')(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block4_pool')(x)

    # Block 5
    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv1')(x)
    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv2')(x)
    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv3')(x)
    # x = MaxPooling2D((2, 2), strides=(2, 2), name='block5_pool')(x)

    return x
    """    
    
    if input_tensor is None:
        img_input = Input(shape=(None, None, 3))
    else:
        if not K.is_keras_tensor(input_tensor):
            img_input = Input(tensor=input_tensor, shape=(None, None, 3))
        else:
            img_input = input_tensor
            
    """ Layer 1: modified stides to 1  """
    conv1 = Conv2D(filters=96, kernel_size=(7, 7), strides=(1, 1),  padding='same', activation='relu')(img_input)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)
    #conv1 = Conv2D(filters=96, kernel_size=(7, 7), strides=(2, 2), padding='same', activation='relu')(img_input)
    #pool1 = MaxPooling2D(pool_size=(2, 2), strides=(2, 2))(conv1)

    """ Layer 2: modified stides to 1 """
    conv2 = Conv2D(filters=256, kernel_size=(5, 5), strides=(1, 1),  padding='same', activation='relu')(pool1)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)
    #pool2 = MaxPooling2D(pool_size=(2, 2), strides=(2, 2))(conv2)

    """ Layer 3 """
    conv3 = Conv2D(filters=384, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(pool2)

    """ Layer 4 """
    conv4 = Conv2D(filters=384, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(conv3)

    """ Layer 5: deleted the pooling layer """
    conv5 = Conv2D(filters=256, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(conv4)
    # pool5 = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='same')(conv5)
    #print("DEBUGGING 61: conv5 shape =", conv5.output_shape())
    x = conv5
    print("DEBUGGING: simple_net 45: x shape =", x.shape)
    return x


def rpn(base_layers, num_anchors):
    #x = Conv2D(512, (3, 3), padding='same', activation='relu', kernel_initializer='normal', name='rpn_conv1')(base_layers)
    x = Conv2D(256, (3, 3), padding='same', activation='relu', kernel_initializer='normal', name='rpn_conv1')(base_layers)
    x_class = Conv2D(num_anchors, (1, 1), activation='sigmoid', kernel_initializer='uniform', name='rpn_out_class')(x)
    x_regr = Conv2D(num_anchors * 4, (1, 1), activation='linear', kernel_initializer='zero', name='rpn_out_regress')(x)

    return [x_class, x_regr, base_layers]


def classifier(base_layers, input_rois, num_rois, nb_classes=44, trainable=False):
    '''
    # compile times on theano tend to be very high, so we use smaller ROI pooling regions to workaround
    if K.backend() == 'tensorflow':
        pooling_regions = 7
        input_shape = (num_rois,7,7,512)
    elif K.backend() == 'theano':
        pooling_regions = 7
        input_shape = (num_rois,512,7,7)
    '''
    pooling_regions = 7
    # input_shape = (num_rois, 7, 7, 512)
    out_roi_pool = RoiPoolingConv(pooling_regions, num_rois)([base_layers, input_rois])

    out = TimeDistributed(Flatten(name='flatten'))(out_roi_pool)
    out = TimeDistributed(Dense(4096, activation='relu', name='fc1'))(out)
    #out = TimeDistributed(Dropout(0.5))(out)           ######## modify to try dropout
    out = TimeDistributed(Dense(4096, activation='relu', name='fc2'))(out)
    #out = TimeDistributed(Dropout(0.5))(out)

    out_class = TimeDistributed(Dense(nb_classes, activation='softmax', kernel_initializer='zero'), name='dense_class_{}'.format(nb_classes))(out)
    # note: no regression target for bg class
    out_regr = TimeDistributed(Dense(4 * (nb_classes-1), activation='linear', kernel_initializer='zero'), name='dense_regress_{}'.format(nb_classes))(out)

    return [out_class, out_regr]


