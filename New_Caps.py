#NEW Capsnet

from keras import layers
from keras import Input
from keras import backend as K
from keras import Model
from keras.layers import Activation
from keras.layers import Add
K.set_image_data_format('channels_last')
from keras.optimizers import Adam
#from custom_losses import dice_hard, weighted_binary_crossentropy_loss, dice_loss


from capsule_layers import ConvCapsuleLayer, DeconvCapsuleLayer, Length

class NET(object):
    
    def __init__(self, img_shape):
        self.img_shape = img_shape

    def CapsNet(self, input_shape, n_class=2):
        
        input_dim = Input(shape = [256, 256, 1])
        
        # Layer 1: Just a conventional Conv2D layer
        conv1 = layers.Conv2D(filters=16, kernel_size=5, strides=1, padding='same', activation='relu', name='conv1')(input_dim)
    
        # Reshape layer to be 1 capsule x [filters] atoms
        _, H, W, C = conv1.get_shape()
        conv1_reshaped = layers.Reshape((H.value, W.value, 1, C.value))(conv1)
    
        # Layer 1: Primary Capsule: Conv cap with routing 1
        primary_caps = ConvCapsuleLayer(kernel_size=5, num_capsule=2, num_atoms=16, strides=2, padding='same',
                                        routings=1, name='primarycaps')(conv1_reshaped)
    
        # Layer 2: Convolutional Capsule
        conv_cap_2_1 = ConvCapsuleLayer(kernel_size=5, num_capsule=4, num_atoms=16, strides=1, padding='same',
                                        routings=3, name='conv_cap_2_1')(primary_caps)
        
        x_ip = ConvCapsuleLayer(kernel_size = 5, num_capsule = 4, num_atoms = 16, strides = 2, padding = 'same',
                                        routings = 3, name = 'x_ip')(primary_caps)
        
        y_ip = ConvCapsuleLayer(kernel_size = 5, num_capsule = 4, num_atoms = 16, strides = 4, padding = 'same',
                                        routings = 3, name = 'y_ip')(primary_caps)
        
        
        #Layer 3: Convolutional Capsule (X_IP)
        
        x_ip_1 = ConvCapsuleLayer(kernel_size = 5, num_capsule = 8, num_atoms = 32, strides = 1, padding = 'same',
                                        routings = 3, name = 'x_ip_1')(x_ip)
        
        x_ip_2 = ConvCapsuleLayer(kernel_size = 5, num_capsule = 8, num_atoms = 32, strides = 2, padding = 'same',
                                        routings = 3, name = 'x_ip_2')(x_ip_1)
        
        x_ip_3 = ConvCapsuleLayer(kernel_size = 5, num_capsule = 8, num_atoms = 32, strides = 1, padding = 'same',
                                        routings = 3, name = 'x_ip_3')(x_ip_2)
        
        
        #Layer 4: Convolutional Capsule (conv_cap_2_1)
        
        conv_cap_2_2 = ConvCapsuleLayer(kernel_size = 5, num_capsule = 8, num_atoms = 32, strides = 1, padding = 'same',
                                        routings = 3, name = 'conv_cap_2_2')(conv_cap_2_1)
        
        conv_cap_2_3 = ConvCapsuleLayer(kernel_size = 5, num_capsule = 8, num_atoms = 32, strides = 2, padding = 'same',
                                        routings = 3, name = 'conv_cap_2_3')(conv_cap_2_2)
        
        conv_cap_2_4 = ConvCapsuleLayer(kernel_size = 5, num_capsule = 8, num_atoms = 32, strides = 1, padding = 'same',
                                        routings = 3, name = 'conv_cap_2_4')(conv_cap_2_3)
        
        conv_cap_2_5 = ConvCapsuleLayer(kernel_size = 5, num_capsule = 8, num_atoms = 32, strides =2, padding = 'same',
                                        routings = 3, name = 'conv_cap_2_5')(conv_cap_2_4)
        
        conv_cap_2_6 = ConvCapsuleLayer(kernel_size = 5, num_capsule = 8, num_atoms = 32, strides = 1, padding = 'same',
                                        routings = 3, name = 'conv_cap_2_6')(conv_cap_2_5)
        
        #Layer 5: Convolutional Capsule (Y_IP)
        
        y_ip_1 = ConvCapsuleLayer(kernel_size = 5, num_capsule = 8, num_atoms = 32, strides = 1, padding = 'same',
                                        routings = 3, name = 'y_ip_1')(y_ip)
        
        #Layer 6: Concatenation ([X_IP_3, conv_cap_2_6, Y_IP_1])
        
        concat_1 = layers.Concatenate(axis = -1, name = 'concat_1')([x_ip_3, conv_cap_2_6, y_ip_1])

        Sum = ConvCapsuleLayer(kernel_size = 5, num_capsule = 8, num_atoms = 32, strides = 1, padding = 'same',
                                        routings = 3, name = 'sum')(concat_1)
        res = Sum
        
        Sum_1 = Activation('relu')(Sum)
        Sum_2 = ConvCapsuleLayer(kernel_size = 5, num_capsule = 8, num_atoms = 32, strides = 1, padding = 'same',
                                        routings = 3, name = 'Sum_2')(Sum_1)
        
        Sum_3 = Add()([Sum_2, res])
        res_1 = Sum_3
        
        Sum_4 = Activation('relu')(Sum_3)
        Sum_5 = ConvCapsuleLayer(kernel_size = 5, num_capsule = 8, num_atoms = 32, strides = 1, padding = 'same',
                                        routings = 3, name = 'Sum_5')(Sum_4)
        
        Sum_6 = Add()([Sum_5, res_1])
    
        # Layer 1 Up: Deconvolutional Capsule
        deconv_cap_1 = ConvCapsuleLayer(kernel_size=5, num_capsule=8, num_atoms=32, strides=1,
                                          padding='same', routings=3, name='deconv_cap_1')(Sum_6)
        
        deconv_cap_2 = ConvCapsuleLayer(kernel_size = 5, num_capsule = 8, num_atoms = 32, strides = 1,
                                          padding = 'same', routings = 3, name = 'deconv_cap_2')(Sum_6)
        
        deconv_cap_3 = ConvCapsuleLayer(kernel_size = 5, num_capsule = 8, num_atoms = 32, strides = 1,
                                          padding = 'same', routings = 3, name = 'deconv_cap_3')(Sum_6)
    
        # Layer 1.1 Up: Deconvolutional Capsule (DECONV_CAP_1):
        
        deconv_cap_1_1 = DeconvCapsuleLayer(kernel_size = 4, num_capsule = 4, num_atoms = 16, upsamp_type = 'deconv',
                                            scaling = 2, padding = 'same', routings = 3,
                                            name = 'deconv_cap_1_1')(deconv_cap_1)
        
        deconv_cap_1_2 = DeconvCapsuleLayer(kernel_size = 5, num_capsule = 4, num_atoms = 16, upsamp_type = 'deconv',
                                            scaling = 1, padding = 'same', routings = 3,
                                            name = 'deconv_cap_1_2')(deconv_cap_1_1)
        
        deconv_cap_1_3 = DeconvCapsuleLayer(kernel_size = 4, num_capsule = 4, num_atoms = 16, upsamp_type = 'deconv',
                                            scaling = 2, padding = 'same', routings = 3,
                                            name = 'deconv_cap_1_3')(deconv_cap_1_2)
        
        deconv_cap_1_4 = DeconvCapsuleLayer(kernel_size = 5, num_capsule = 4, num_atoms = 16, upsamp_type = 'deconv',
                                            scaling = 1, padding = 'same', routings = 3,
                                            name = 'deconv_cap_1_4')(deconv_cap_1_3)
        
        #Dimensions = [None, 128, 128, 4, 16]
        
        
        
        # Layer 1.2 Up: Deconvolutional Capsule (DECONV_CAP_2)
        
        deconv_cap_2_1 = DeconvCapsuleLayer(kernel_size=4, num_capsule=4, num_atoms=16, upsamp_type='deconv',
                                            scaling=2, padding='same', routings=3,
                                            name='deconv_cap_2_1')(deconv_cap_2)
        
        deconv_cap_2_2 = DeconvCapsuleLayer(kernel_size = 5, num_capsule = 4, num_atoms = 16, upsamp_type = 'deconv',
                                            scaling = 1, padding = 'same', routings = 3,
                                            name = 'deconv_cap_2_2')(deconv_cap_2_1)
        
        #Layers 1.3 Up: Deeconvolutional Capsule (DECONV_CAP_3)
      
        deconv_cap_3_1 = DeconvCapsuleLayer(kernel_size = 5, num_capsule = 4, num_atoms = 16, upsamp_type = 'deconv',
                                            scaling = 1, padding = 'same', routings = 3,
                                            name = 'deconv_cap_3_1')(deconv_cap_3)
        
        #Layers 2.2 Up: Deconvolutional Capsule  (DECONV_CAP_2.2)
        
        deconv_cap_2_2_1 = DeconvCapsuleLayer(kernel_size = 4, num_capsule = 4, num_atoms = 16, upsamp_type = 'deconv',
                                              scaling = 2, padding = 'same', routings = 3,
                                              name = 'deconv_cap_2.2_1')(deconv_cap_2_2)
        deconv_cap_2_2_2 = DeconvCapsuleLayer(kernel_size = 5, num_capsule = 4, num_atoms = 16, upsamp_type = 'deconv',
                                              scaling = 1, padding = 'same', routings = 3,
                                              name = 'deconv_cap_2.2_2')(deconv_cap_2_2_1)
        
        #Layers 2.3 Up: Deconvolutional Capsule  (DECONV_CAP_2.3)
        
        deconv_cap_2_3_1 = DeconvCapsuleLayer(kernel_size = 4, num_capsule = 4, num_atoms = 16, upsamp_type = 'deconv',
                                              scaling = 2, padding = 'same', routings = 3,
                                              name = 'deconv_cap_2.3_1')(deconv_cap_3_1)
        
        deconv_cap_2_3_2 = DeconvCapsuleLayer(kernel_size = 5, num_capsule = 4, num_atoms = 16, upsamp_type = 'deconv',
                                              scaling = 1, padding = 'same', routings = 3,
                                              name = 'deconv_cap_2.3_2')(deconv_cap_2_3_1)
        
        deconv_cap_2_3_3 = DeconvCapsuleLayer(kernel_size = 4, num_capsule = 4, num_atoms = 16, upsamp_type = 'deconv',
                                              scaling = 2, padding = 'same', routings = 3,
                                              name = 'deconv_cap_2.3_3')(deconv_cap_2_3_2)
        
        deconv_cap_2_3_4 = DeconvCapsuleLayer(kernel_size = 5, num_capsule = 4, num_atoms = 16, upsamp_type = 'deconv',
                                              scaling = 1, padding = 'same', routings = 3,
                                              name = 'deconv_cap_2.3_4')(deconv_cap_2_3_3)
        #Concatenate all Deconv Layers
        
        concat_2 = layers.Concatenate(axis = -1, name = 'concat_2')([deconv_cap_1_4, deconv_cap_2_2_2, deconv_cap_2_3_4])
        
        #Dimension = [None, 128, 128, 4, 48]
        
        #Final_Deconv:
        
        up_1 = DeconvCapsuleLayer(kernel_size = 5, num_capsule = 4, num_atoms = 16, upsamp_type = 'deconv',
                                  scaling = 1, padding = 'same', routings = 3,
                                  name = 'up_1')(concat_2)
        
        up_2 = DeconvCapsuleLayer(kernel_size = 4, num_capsule  =2, num_atoms = 16, upsamp_type = 'deconv',
                                  scaling = 2, padding = 'same', routings = 3,
                                  name = 'up_2')(up_1)
        
        seg_caps = ConvCapsuleLayer(kernel_size = 1, num_capsule = 1, num_atoms = 16, strides = 1, padding = 'same',
                                    routings = 3, name = 'seg_caps')(up_2)
        
        
        # Skip connection
        #up_2 = layers.Concatenate(axis=-2, name='up_2')([deconv_cap_2_1, conv_cap_2_1])
    
        # Layer 2 Up: Deconvolutional Capsule
        #deconv_cap_2_2 = ConvCapsuleLayer(kernel_size=5, num_capsule=4, num_atoms=16, strides=1,
                                          #padding='same', routings=3, name='deconv_cap_2_2')(up_2)
    
        # Layer 3 Up: Deconvolutional Capsule
        #deconv_cap_3_1 = DeconvCapsuleLayer(kernel_size=4, num_capsule=2, num_atoms=16, upsamp_type='deconv',
                                            #scaling=2, padding='same', routings=3,
                                            #name='deconv_cap_3_1')(deconv_cap_2_2)
    
        # Skip connection
        #up_3 = layers.Concatenate(axis=-2, name='up_3')([deconv_cap_3_1, conv1_reshaped])
    
        # Layer 4: Convolutional Capsule: 1x1
        #seg_caps = ConvCapsuleLayer(kernel_size=1, num_capsule=1, num_atoms=16, strides=1, padding='same',
                                    #routings=3, name='seg_caps')(up_3)
    
        # Layer 4: This is an auxiliary layer to replace each capsule with its length. Just to match the true label's shape.
        out_seg = Length(num_classes=2, seg=True, name='out_seg')(seg_caps)
    
        #result = Model(inputs = input_dim, outputs = out_seg)
        #print(result.summary())
        
        return out_seg
    
    
    """def margin_loss(y_true, y_pred):
        
        L = y_true * K.square(K.maximum(0., 0.9 - y_pred)) + \
            0.5 * (1 - y_true) * K.square(K.maximum(0., y_pred - 0.1))

        return K.mean(K.sum(L, 1))"""

    def initModel_2D(self):
        
        height, width, depth = self.img_shape
        input_dimension = Input(shape = (height, width, depth))
        net_op = self.CapsNet(input_dimension)
        net_model = Model(inputs = input_dimension, outputs = net_op)
        adam = Adam(lr = 1e-4)
        
        net_model.compile(optimizer = adam, loss = 'binary_crossentropy', metrics = ['accuracy'])
        
        return net_model
