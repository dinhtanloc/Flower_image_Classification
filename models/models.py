from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os

from keras_applications import get_submodules_from_kwargs
from keras_applications import imagenet_utils
from keras import layers
from tensorflow.keras.models import Model
from keras_applications.imagenet_utils import _obtain_input_shape, decode_predictions
from tensorflow.keras.layers import Flatten, Dense, Input, GlobalAveragePooling2D, \
    GlobalMaxPooling2D, Activation, Conv2D, MaxPooling2D, BatchNormalization, \
    AveragePooling2D, Reshape, Permute, multiply
from keras.layers import Flatten, Dropout, Dense
from keras.models import Model
from keras.regularizers import l2
from keras.applications import InceptionV3, ResNet50, VGG16

from keras.layers import GlobalAveragePooling2D, GlobalMaxPooling2D, Dense
from keras.layers import Flatten, Dropout, Concatenate, BatchNormalization, Input, Convolution2D, MaxPooling2D, concatenate, Activation

from keras.models import Model, Sequential, load_model

global weight_decay
weight_decay = 1e-4

""" *****************************************
MODELS
***************************************** """
torch_imagenet_mean = [0.485, 0.456, 0.406]
torch_imagenet_std = [0.229, 0.224, 0.225]
    
caffe_imagenet_mean = [103.939, 116.779, 123.68] # BGR
def caffe_preprocessing_input(x): # BGR
    x[..., 0] -= caffe_imagenet_mean[0]
    x[..., 1] -= caffe_imagenet_mean[1]
    x[..., 2] -= caffe_imagenet_mean[2]
    return x
# caffe_preprocessing_input

def torch_preprocessing_input(x): # BGR
    """
    torch: will scale pixels between 0 and 1 
    and then will normalize each channel with respect to the
    ImageNet dataset.
    """
    x = x[...,::-1] # BGR --> RGB
    x /= 255.
    x[..., 0] -= torch_imagenet_mean[0]
    x[..., 1] -= torch_imagenet_mean[1]
    x[..., 2] -= torch_imagenet_mean[2]

    x[..., 0] /= torch_imagenet_std[0]
    x[..., 1] /= torch_imagenet_std[1]
    x[..., 2] /= torch_imagenet_std[2]
    return x
# torch_preprocessing_input

def tf_preprocess_input(x):
    """
    Image RGB
    tf: will scale pixels between -1 and 1, sample-wise.
    """
    x = x[...,::-1] # BGR --> RGB
    x = x / 127.5
    x -= 1.0
    return x
# tf_preprocess_input

def vgg16_preprocessing_input(x):
    return caffe_preprocessing_input(x)
# vgg16_preprocessing_input

def resnet50_preprocessing_input(x):
    return caffe_preprocessing_input(x)
# resnet50_preprocessing_input

def inceptionv3_preprocessing_input(x):
    return tf_preprocess_input(x)
# inceptionv3_preprocessing_input

def build_common_model(weight_path = None, model_name = None, nb_classes = 14, fc=[2048, 0], dropout = [0.1, 0.1, 0.0], input_shape = None):
    # base model
    base_model = None
    if model_name=="imagenet_inception_v3":
        input_shape = (224, 224, 3) if input_shape is None else input_shape
        base_model = InceptionV3(include_top=False, weights='imagenet', input_tensor=None, input_shape=input_shape, pooling="avg", classes=14)
    elif model_name=="inception_v3":
        input_shape = (224, 224, 3) if input_shape is None else input_shape
        base_model = InceptionV3(include_top=False, weights=None, input_tensor=None, input_shape=input_shape, pooling="avg", classes=14)
    elif model_name=="imagenet_resnet50":
        input_shape = (224, 224, 3) if input_shape is None else input_shape
        base_model = ResNet50(include_top=False, weights='imagenet', input_tensor=None, input_shape=input_shape, pooling="avg", classes=14)
    elif model_name=="resnet50":
        input_shape = (224, 224, 3) if input_shape is None else input_shape
        base_model = ResNet50(include_top=False, weights=None, input_tensor=None, input_shape=input_shape, pooling="avg", classes=14)
    elif model_name=="imagenet_vgg16":
        input_shape = (224, 224, 3) if input_shape is None else input_shape
        base_model = VGG16(include_top=False, weights="imagenet", input_tensor=None, input_shape=input_shape, pooling="avg", classes=14)
    elif model_name=="vgg16":
        input_shape = (224, 224, 3) if input_shape is None else input_shape
        base_model = VGG16(include_top=False, weights=None, input_tensor=None, input_shape=input_shape, pooling="avg", classes=14)
    # if
    
    model = None
    if base_model is not None:
        x = base_model.output
        if dropout[0]>0: x = Dropout(dropout[0])(x)
        if fc[0]>0: x = Dense(fc[0], activation='relu')(x)
        if dropout[1] > 0: x = Dropout(dropout[1])(x)
        if fc[1]>0: x = Dense(fc[1], activation='relu')(x)
        if dropout[2] > 0: x = Dropout(dropout[2])(x)
        x = Dense(nb_classes,
                  activation='softmax',
                  name='predictions',
                  use_bias=False, trainable=True,
                  kernel_initializer='orthogonal',
                  kernel_regularizer=l2(weight_decay))(x)
        model = Model(inputs=base_model.input, outputs=x)
    # if
    
    if weight_path is not None: model.load_weights(weight_path, by_name = True)
    return model
# build_common_model





"""VGG16 model for Keras.

# Reference

- [Very Deep Convolutional Networks for Large-Scale Image Recognition](
    https://arxiv.org/abs/1409.1556) (ICLR 2015)

"""
preprocess_input = imagenet_utils.preprocess_input

WEIGHTS_PATH = ('https://github.com/fchollet/deep-learning-models/'
                'releases/download/v0.1/'
                'vgg16_weights_tf_dim_ordering_tf_kernels.h5')
WEIGHTS_PATH_NO_TOP = ('https://github.com/fchollet/deep-learning-models/'
                       'releases/download/v0.1/'
                       'vgg16_weights_tf_dim_ordering_tf_kernels_notop.h5')


def VGG16(include_top=True,
          weights_path = None,
          input_tensor=None,
          input_shape=None,
          pooling=None,
          classes=1000,
          **kwargs):
    """Instantiates the VGG16 architecture.

    Optionally loads weights pre-trained on ImageNet.
    Note that the data format convention used by the model is
    the one specified in your Keras config at `~/.keras/keras.json`.

    # Arguments
        include_top: whether to include the 3 fully-connected
            layers at the top of the network.
        weights: one of `None` (random initialization),
              'imagenet' (pre-training on ImageNet),
              or the path to the weights file to be loaded.
        input_tensor: optional Keras tensor
            (i.e. output of `Input()`)
            to use as image input for the model.
        input_shape: optional shape tuple, only to be specified
            if `include_top` is False (otherwise the input shape
            has to be `(224, 224, 3)`
            (with `channels_last` data format)
            or `(3, 224, 224)` (with `channels_first` data format).
            It should have exactly 3 input channels,
            and width and height should be no smaller than 32.
            E.g. `(200, 200, 3)` would be one valid value.
        pooling: Optional pooling mode for feature extraction
            when `include_top` is `False`.
            - `None` means that the output of the model will be
                the 4D tensor output of the
                last convolutional block.
            - `avg` means that global average pooling
                will be applied to the output of the
                last convolutional block, and thus
                the output of the model will be a 2D tensor.
            - `max` means that global max pooling will
                be applied.
        classes: optional number of classes to classify images
            into, only to be specified if `include_top` is True, and
            if no `weights` argument is specified.

    # Returns
        A Keras model instance.

    # Raises
        ValueError: in case of invalid argument for `weights`,
            or invalid input shape.
    """
    # backend, layers, models, keras_utils = get_submodules_from_kwargs(kwargs)
    # Determine proper input shape
    input_shape = _obtain_input_shape(input_shape,
                                      default_size=224,
                                      min_size=32,
                                      data_format='channels_last',
                                      require_flatten=include_top,
                                      )

    if input_tensor is None:
        img_input = Input(shape=input_shape)
    else:
            img_input = input_tensor
    # Block 1
    x = Conv2D(64, (3, 3),
                      activation='relu',
                      padding='same',
                      name='block1_conv1')(img_input)
    x = Conv2D(64, (3, 3),
                      activation='relu',
                      padding='same',
                      name='block1_conv2')(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block1_pool')(x)

    # Block 2
    x = Conv2D(128, (3, 3),
                      activation='relu',
                      padding='same',
                      name='block2_conv1')(x)
    x = Conv2D(128, (3, 3),
                      activation='relu',
                      padding='same',
                      name='block2_conv2')(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block2_pool')(x)

    # Block 3
    x = Conv2D(256, (3, 3),
                      activation='relu',
                      padding='same',
                      name='block3_conv1')(x)
    x = Conv2D(256, (3, 3),
                      activation='relu',
                      padding='same',
                      name='block3_conv2')(x)
    x = Conv2D(256, (3, 3),
                      activation='relu',
                      padding='same',
                      name='block3_conv3')(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block3_pool')(x)

    # Block 4
    x = Conv2D(512, (3, 3),
                      activation='relu',
                      padding='same',
                      name='block4_conv1')(x)
    x = Conv2D(512, (3, 3),
                      activation='relu',
                      padding='same',
                      name='block4_conv2')(x)
    x = Conv2D(512, (3, 3),
                      activation='relu',
                      padding='same',
                      name='block4_conv3')(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block4_pool')(x)

    # Block 5
    x = Conv2D(512, (3, 3),
                      activation='relu',
                      padding='same',
                      name='block5_conv1')(x)
    x = Conv2D(512, (3, 3),
                      activation='relu',
                      padding='same',
                      name='block5_conv2')(x)
    x = Conv2D(512, (3, 3),
                      activation='relu',
                      padding='same',
                      name='block5_conv3')(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block5_pool')(x)

    if include_top:
        # Classification block
        x = Flatten(name='flatten')(x)
        x = Dense(4096, activation='relu', name='fc1')(x)
        x = Dense(4096, activation='relu', name='fc2')(x)
        x = Dense(classes, activation='softmax', name='predictions')(x)
    else:
        if pooling == 'avg':
            x = GlobalAveragePooling2D()(x)
        elif pooling == 'max':
            x = GlobalMaxPooling2D()(x)

 
   
    # Create model.
    model = Model(img_input, x, name='vgg16')

   
    if weights_path is not None:
        model.load_weights(weights_path)

    return model


def resnet_conv_block(input_tensor, kernel_size, filters, stage, block,
                      strides=(2, 2), bias=False):
    filters1, filters2, filters3 = filters
    bn_axis=3

    conv1_reduce_name = 'conv' + str(stage) + "_" + str(block) + "_1x1_reduce"
    conv1_increase_name = 'conv' + str(stage) + "_" + str(
        block) + "_1x1_increase"
    conv1_proj_name = 'conv' + str(stage) + "_" + str(block) + "_1x1_proj"
    conv3_name = 'conv' + str(stage) + "_" + str(block) + "_3x3"

    x = Conv2D(filters1, (1, 1), strides=strides, use_bias=bias,
               name=conv1_reduce_name)(input_tensor)
    x = BatchNormalization(axis=bn_axis, name=conv1_reduce_name + "bn")(x)
    x = Activation('relu')(x)

    x = Conv2D(filters2, kernel_size, padding='same', use_bias=bias,
               name=conv3_name)(x)
    x = BatchNormalization(axis=bn_axis, name=conv3_name + "bn")(x)
    x = Activation('relu')(x)

    x = Conv2D(filters3, (1, 1), name=conv1_increase_name, use_bias=bias)(x)
    x = BatchNormalization(axis=bn_axis, name=conv1_increase_name + "bn")(x)

    shortcut = Conv2D(filters3, (1, 1), strides=strides, use_bias=bias,
                      name=conv1_proj_name)(input_tensor)
    shortcut = BatchNormalization(axis=bn_axis, name=conv1_proj_name + "bn")(
        shortcut)

    x = layers.add([x, shortcut])
    x = Activation('relu')(x)
    return x


def resnet_identity_block(input_tensor, kernel_size, filters, stage, block,
                          bias=False):
    filters1, filters2, filters3 = filters
    bn_axis=3

    conv1_reduce_name = 'conv' + str(stage) + "_" + str(block) + "_1x1_reduce"
    conv1_increase_name = 'conv' + str(stage) + "_" + str(
        block) + "_1x1_increase"
    conv3_name = 'conv' + str(stage) + "_" + str(block) + "_3x3"

    x = Conv2D(filters1, (1, 1), use_bias=bias, name=conv1_reduce_name)(
        input_tensor)
    x = BatchNormalization(axis=bn_axis, name=conv1_reduce_name + "bn")(x)
    x = Activation('relu')(x)

    x = Conv2D(filters2, kernel_size, use_bias=bias,
               padding='same', name=conv3_name)(x)
    x = BatchNormalization(axis=bn_axis, name=conv3_name + "bn")(x)
    x = Activation('relu')(x)

    x = Conv2D(filters3, (1, 1), use_bias=bias, name=conv1_increase_name)(x)
    x = BatchNormalization(axis=bn_axis, name=conv1_increase_name + "bn")(x)

    x = layers.add([x, input_tensor])
    x = Activation('relu')(x)
    return x


def RESNET50(include_top=True, weights_path = None,
             input_tensor=None, input_shape=None,
             pooling=None,
             classes=2):
    input_shape = _obtain_input_shape(input_shape,
                                      default_size=224,
                                      min_size=32,
                                      data_format='channel_last',
                                      require_flatten=include_top
                                        )
    bn_axis = 3
    if input_tensor is None:
        img_input = Input(shape=input_shape)
    else:
        img_input = input_tensor
    x = Conv2D(
        64, (7, 7), use_bias=False, strides=(2, 2), padding='same',
        name='conv1_7x7_s2')(img_input)
    x = BatchNormalization(axis=bn_axis, name='conv1_7x7_s2_bn')(x)
    x = Activation('relu')(x)
    x = MaxPooling2D((3, 3), strides=(2, 2))(x)

    x = resnet_conv_block(x, 3, [64, 64, 256], stage=2, block=1, strides=(1, 1))
    x = resnet_identity_block(x, 3, [64, 64, 256], stage=2, block=2)
    x = resnet_identity_block(x, 3, [64, 64, 256], stage=2, block=3)

    x = resnet_conv_block(x, 3, [128, 128, 512], stage=3, block=1)
    x = resnet_identity_block(x, 3, [128, 128, 512], stage=3, block=2)
    x = resnet_identity_block(x, 3, [128, 128, 512], stage=3, block=3)
    x = resnet_identity_block(x, 3, [128, 128, 512], stage=3, block=4)

    x = resnet_conv_block(x, 3, [256, 256, 1024], stage=4, block=1)
    x = resnet_identity_block(x, 3, [256, 256, 1024], stage=4, block=2)
    x = resnet_identity_block(x, 3, [256, 256, 1024], stage=4, block=3)
    x = resnet_identity_block(x, 3, [256, 256, 1024], stage=4, block=4)
    x = resnet_identity_block(x, 3, [256, 256, 1024], stage=4, block=5)
    x = resnet_identity_block(x, 3, [256, 256, 1024], stage=4, block=6)

    x = resnet_conv_block(x, 3, [512, 512, 2048], stage=5, block=1)
    x = resnet_identity_block(x, 3, [512, 512, 2048], stage=5, block=2)
    x = resnet_identity_block(x, 3, [512, 512, 2048], stage=5, block=3)

    x = AveragePooling2D((7, 7), name='avg_pool')(x)

    if include_top:
        x = Flatten()(x)
        x = Dense(1, activation='sigmoid', name='classifier')(x)
    else:
        if pooling == 'avg':
            x = GlobalAveragePooling2D()(x)
        elif pooling == 'max':
            x = GlobalMaxPooling2D()(x)
    # Create model.
    model = Model(img_input, x, name='vggface_resnet50')


    # load weights
   
    if weights_path is not None:
        model.load_weights(weights_path, by_name=True)

    return model
# resnet_model = RESNET50(include_top=False,input_shape=(224,224,3),pooling='avg')

