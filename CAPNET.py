#best1

from keras.models import Model
from keras.layers import Activation, Input
from keras.layers.convolutional import Conv2D, Conv2DTranspose, Conv3D, Conv3DTranspose
from keras.layers import Add, Concatenate, Subtract
from keras.layers import BatchNormalization
from keras.layers import Dropout
from keras.layers import UpSampling2D, UpSampling3D
from keras.layers import MaxPooling3D as mp3d
from keras.layers import AveragePooling3D as ap3d
from keras.optimizers import Adam
import keras.backend as K
from keras.utils import plot_model
from keras.layers.advanced_activations import LeakyReLU
from keras.optimizers import SGD
from keras.layers import Reshape
K.set_image_dim_ordering('th')
from capsulelayers import CapsuleLayer, PrimaryCap, Length, Mask
from keras.layers import Flatten, Dense


class Capsule(object):

    def __init__(self, img_shape, img_shape2):
        self.img_shape = img_shape
        self.img_shape2 = img_shape2

    def CapsuleNet(self, input_shape, input_shape2):

        #input_dim = Input([50, 256, 256, 1])
        #input_dim2 = Input([1, 256, 256, 1])    
        model = Conv3D(32, (3, 3, 3), strides = (1, 1, 1), padding = 'same', data_format = 'channels_last')(input_shape)
        model = ap3d(pool_size=(5, 3, 3), strides = (5, 1, 1), padding = 'same', data_format = 'channels_last')(model)
        model = Activation('relu')(model)
                
        model = Conv3D(16, (3, 3, 3), strides = 1, padding = 'same', data_format = 'channels_last')(model)
        model = ap3d(pool_size = (2, 3, 3), strides = (2, 1, 1), padding = 'same', data_format = 'channels_last')(model)        
        model = Activation('relu')(model)

        model = Conv3D(8, (3, 3, 3), strides = 1, padding = 'same', data_format = 'channels_last')(model)
        model = ap3d(pool_size = (5, 3, 3), strides = (5, 1, 1), padding = 'same', data_format = 'channels_last')(model)
        model = Activation('relu')(model)

        #model = Conv3D(4, (3, 3, 3), strides = 1, padding = 'same', data_format = 'channels_last')(model)
        #model = mp3d(pool_size = (3, 3, 3), strides = (5, 1, 1), padding = 'same', data_format = 'channels_last')(model)
        #model = Activation('relu')(model)
        

        bg = Conv3D(1, (1, 3, 3), padding = 'same', data_format = 'channels_last')(model)
        bg = Activation('relu')(bg)
        
        #result = Model(inputs = [input_dim, input_dim2], outputs = bg)
        #print(result.summary())

        bg_estim = Subtract()([input_shape2, bg])


        model = Conv3D(8, (1, 3, 3), padding = 'same', data_format = 'channels_last')(bg_estim)
        model2 = Conv3D(8, (1, 3, 3), padding = 'same', data_format = 'channels_last')(input_shape2)
        model = Concatenate(axis = -1)([ model, model2])
        model = Activation('relu')(model)
        
        model2 = Reshape(target_shape = (256, 256, 16))(model)
        
        model2 = Conv2D(filters = 64, kernel_size =3, strides = 1, padding = 'same', data_format = 'channels_last', activation = 'relu')(model2)
        
        primarycaps = PrimaryCap(model2, dim_capsule=8, n_channels=32, kernel_size=9, strides=2, padding='same', data_format = 'channels_last')
        
        digitcaps = CapsuleLayer(num_capsule = 2, dim_capsule=16, routings=3, name='digitcaps')(primarycaps)
        
        flatten = Flatten(data_format = 'channels_last')(digitcaps)
        
        dense = Dense(512)(flatten)
        dense2 = Dense(4096)(dense)
        
        
        reshape = Reshape(target_shape = (1, 64, 64, 1))(dense2)
        
        trans = UpSampling3D(size= (1, 2, 2), data_format = 'channels_last')(reshape)
        trans = Conv3DTranspose(1, (3, 3, 3), padding = 'same', data_format = 'channels_last')(trans)
        trans = UpSampling3D(size = (1, 2, 2), data_format = 'channels_last')(trans)
        trans = Conv3DTranspose(1, (3, 3, 3), padding = 'same', data_format = 'channels_last')(trans)

        
        #result = Model(inputs= [input_dim, input_dim2], outputs = trans)
        #print(result.summary())
        
        return trans

    def margin_loss(y_true, y_pred):
        """
        Margin loss for Eq.(4). When y_true[i, :] contains not just one `1`, this loss should work too. Not test it.
        :param y_true: [None, n_classes]
        :param y_pred: [None, num_capsule]
        :return: a scalar loss value.
        """
        L = y_true * K.square(K.maximum(0., 0.9 - y_pred)) + \
            0.5 * (1 - y_true) * K.square(K.maximum(0., y_pred - 0.1))
    
        return K.mean(K.sum(L, 1))

    def initModel_3D(self):

        depth, height, width = self.img_shape
        depth2, height2, width2 = self.img_shape2
        input_dimension = Input(shape = (depth, height, width, 1), name="main_input")
        input_dimension2 = Input(shape = (depth2, height2, width2, 1), name="aux_input")
        net_op = self.CapsuleNet(input_dimension, input_dimension2)
        net_model = Model(inputs = [input_dimension, input_dimension2], outputs = net_op)

        sgd = SGD(lr = 2e-04, decay = 0, nesterov = True) #for 50-50 scene = 1e-03,,,, for 50% 5e-04   #decay = 1e-05
        net_model.compile(optimizer = sgd, loss = 'binary_crossentropy', metrics = ['accuracy'])  #or loss = 'margin_loss'

        return net_model