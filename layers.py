import tensorflow as tf
from tensorflow import keras
import tensorflow_addons as tfa
from tensorflow.keras import layers, Model, regularizers, backend as K
from tensorflow.keras.activations import swish
from tensorflow.keras.layers import Activation, Conv2D, Input, GlobalAveragePooling2D, Concatenate, InputLayer, \
ReLU, Flatten, Dense, Dropout, BatchNormalization, MaxPooling2D, GlobalMaxPooling2D, Softmax, Lambda, LeakyReLU, Reshape, \
DepthwiseConv2D, Multiply, Add, LayerNormalization, Conv2DTranspose

# pip install -U git+https://github.com/leondgarse/keras_cv_attention_models -q

from keras_cv_attention_models.swin_transformer_v2.swin_transformer_v2 import swin_transformer_block

def msca(inputs):
    x = DepthwiseConv2D((5,5), padding="same")(inputs)
    x_7 = DepthwiseConv2D((1,7), padding="same")(x)
    x_7 = DepthwiseConv2D((7,1), padding="same")(x_7)
    x_11 = DepthwiseConv2D((1,11), padding="same")(x)
    x_11 = DepthwiseConv2D((11,1), padding="same")(x_11)
    x_21 = DepthwiseConv2D((1,21), padding="same")(x)
    x_21 = DepthwiseConv2D((21,1), padding="same")(x_21)
    x = Add()([x, x_7, x_11, x_21])
    x = Conv2D(inputs.shape[-1], 1, padding="same", activation="sigmoid")(x)
    return inputs * x 

def mscan(inputs):
    x = Conv2D(inputs.shape[-1], 1, padding="same")(inputs)
    x = Activation('gelu')(x)
    x = msca(x)
    x = Conv2D(inputs.shape[-1], 1, padding="same")(x)
    x_save = Add()([x, inputs])
    x = Conv2D(inputs.shape[-1], 1, padding="same")(x_save)
    x = DepthwiseConv2D((3,3), padding="same")(x)
    x = Activation('gelu')(x)
    x = Conv2D(inputs.shape[-1], 1, padding="same")(x)
    return Add()([x_save, x])

def convnext_block(inputs, filters, drop_rate=0, activation="gelu"):
    x = DepthwiseConv2D(kernel_size=7, padding="SAME", use_bias=True)(inputs)
    x = Dense(4 * filters)(x)
    x = Activation(activation=activation)(x)
    x = Dense(filters)(x)
    x = Dropout(drop_rate)(x)
    return Add()([inputs, x])

def extract_block(inputs, filters, drop_rate=0.01):
    x = convnext_block(inputs, filters, drop_rate=drop_rate)
    x = convnext_block(x, filters, drop_rate=drop_rate)
    return x

def local_emphasis(inputs, filters, resize_shape):
    x = extract_block(inputs, filters)
    x = tf.image.resize(x, resize_shape)
    return x