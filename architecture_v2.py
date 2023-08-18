import tensorflow as tf

from keras.layers import Input, ZeroPadding2D, Dense, Reshape, Flatten, Concatenate, Conv2D, Conv2DTranspose, GlobalAveragePooling2D, UpSampling2D, LeakyReLU, ReLU, Add, Multiply, Lambda, Dot, BatchNormalization, Activation, ZeroPadding2D, Cropping2D, Cropping1D
from keras.models import Model

from constants import *
from specnorm import ConvSN2D, ConvSN2DTranspose, DenseSN

init = tf.keras.initializers.he_uniform()

def conv2d(layer_input, filters, kernel_size, strides=2, padding='same', leaky=True, bnorm=True, sn=True):
    if leaky:
        Activ = LeakyReLU(alpha=0.2)
    else:
        Activ = ReLU()
    if sn:
        d = ConvSN2D(filters, kernel_size=kernel_size, strides=strides, padding=padding, kernel_initializer=init, use_bias=False)(layer_input)
    else:
        d = Conv2D(filters, kernel_size=kernel_size, strides=strides, padding=padding, kernel_initializer=init, use_bias=False)(layer_input)
    if bnorm:
        d = BatchNormalization()(d)
    d = Activ(d)
    return d

def deconv2d(layer_input, layer_res, filters, kernel_size, conc=True, scalev=False, bnorm=True, up=True, padding='same', strides=2):
    if up:
        u = UpSampling2D((1,2))(layer_input)
        u = ConvSN2D(filters, kernel_size, strides=(1,1), kernel_initializer=init, use_bias=False, padding=padding)(u)
    else:
        u = ConvSN2DTranspose(filters, kernel_size, strides=strides, kernel_initializer=init, use_bias=False, padding=padding)(layer_input)
    if bnorm:
        u = BatchNormalization()(u)
    u = LeakyReLU(alpha=0.2)(u)
    if conc:
        u = Concatenate()([u,layer_res])
    return u

""" U-NET style architecture """
def build_generator(input_shape):
    h,w,c = input_shape
    inputlayer = Input(shape=input_shape)
    #downscaling
    g0 = tf.keras.layers.ZeroPadding2D((0,1))(inputlayer)
    g1 = conv2d(g0, 256, kernel_size=(h,3), strides=1, padding='valid')
    g2 = conv2d(g1, 256, kernel_size=(1,9), strides=(1,2))
    g3 = conv2d(g2, 256, kernel_size=(1,7), strides=(1,2))
    #upscaling
    g4 = deconv2d(g3,g2, 256, kernel_size=(1,7), strides=(1,2))
    g5 = deconv2d(g4,g1, 256, kernel_size=(1,9), strides=(1,2), bnorm=False)
    g6 = ConvSN2DTranspose(1, kernel_size=(h,1), strides=(1,1), kernel_initializer=init, padding='valid', activation='tanh')(g5)
    return Model(inputlayer, g6, name='G')

""" Siamese Network """
def build_siamese(input_shape):
    h,w,c = input_shape
    inp = Input(shape=input_shape)
    g1 = conv2d(inp, 256, kernel_size=(h,3), strides=1, padding='valid', sn=False)
    g2 = conv2d(g1, 256, kernel_size=(1,9), strides=(1,2), sn=False)
    g3 = conv2d(g2, 256, kernel_size=(1,7), strides=(1,2), sn=False)
    g4 = Flatten()(g3)
    g5 = Dense(GL_VECLEN)(g4)
    return Model(inp, g5, name='S')

""" Discriminator (Critic) Network """
def build_critic(input_shape):
    h,w,c = input_shape
    inp = Input(shape=input_shape)
    g1 = conv2d(inp, 512, kernel_size=(h,3), strides=1, padding='valid', bnorm=False)
    g2 = conv2d(g1, 512, kernel_size=(1,9), strides=(1,2), bnorm=False)
    g3 = conv2d(g2, 512, kernel_size=(1,7), strides=(1,2), bnorm=False)
    g4 = Flatten()(g3)
    g4 = DenseSN(1, kernel_initializer=init)(g4) #spectral normalization layer
    return Model(inp, g4, name='C')

""" Load Saved Weights"""
def load(path):
    gen = build_generator((GL_HOP, GL_SHAPE, 1))
    siam = build_siamese((GL_HOP, GL_SHAPE, 1))
    critic = build_critic((GL_HOP, 3 * GL_SHAPE, 1))

    gen.load_weights(path + '/gen.h5')
    critic.load_weights(path + '/critic.h5')
    siam.load_weights(path + '/siam.h5')
    return gen, critic, siam

""" Build models """
def build():
    gen = build_generator((GL_HOP, GL_SHAPE, 1))
    siam = build_siamese((GL_HOP, GL_SHAPE, 1))
    critic = build_critic((GL_HOP, 3 * GL_SHAPE, 1))                                          #the discriminator accepts as input spectrograms of triple the width of those generated by the generator
    return gen, critic, siam

""" Extract function: splitting spectrograms """
def extract_image(im):
    im = tf.expand_dims(im, -1)
    im1 = Cropping2D(((0,0), (0, 2*(im.shape[2] // 3))))(im)
    im2 = Cropping2D(((0,0), (im.shape[2] // 3, im.shape[2] // 3)))(im)
    im3 = Cropping2D(((0,0), (2 * (im.shape[2] // 3), 0)))(im)

    return im1, im2, im3

""" Assemble function: concatenating spectrograms """
def assemble_image(lsim):
    im1, im2, im3 = lsim
    imh = Concatenate(2)([im1, im2, im3])
    imh = tf.squeeze(imh)
    return imh
