import time, cv2
import numpy as np
from math import ceil
from IPython import display
import matplotlib.pyplot as plt
import pandas as pd
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import cv2
# tensorflow.Keras
from tensorflow.keras.optimizers import SGD, Adam
from tensorflow.keras.losses import binary_crossentropy, categorical_crossentropy

from tensorflow.keras.layers import GlobalAveragePooling2D, GlobalMaxPooling2D, Dense
from tensorflow.keras.layers import Flatten, Dropout, Concatenate, BatchNormalization, Input, Convolution2D, MaxPooling2D, concatenate, Activation

from tensorflow.keras.models import Model, Sequential, load_model

from tensorflow.keras import backend as K

from tensorflow.keras.callbacks import CSVLogger, LambdaCallback, ModelCheckpoint, EarlyStopping, TensorBoard