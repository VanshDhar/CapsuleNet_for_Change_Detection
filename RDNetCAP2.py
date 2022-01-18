from keras import layers
from keras import Input
from keras import backend as K
from keras import Model
K.set_image_data_format('channels_last')
from keras.optimizers import Adam
#from keras.utils import plot_model

#from custom_losses import dice_hard, weighted_binary_crossentropy_loss, dice_loss


from capsule_layers import ConvCapsuleLayer, DeconvCapsuleLayer, Length

class NET(object):

    def __init__(self, img_shape):
        self.img_shape = img_shape

    def RDNetCAP(self, input_dim, n_classes ):

        input_dim = Input(shape = [512, 512, 3])
        n_classes = 32

        # Layer 1: Just a conventional Conv2D layer
        conv1 = layers.Conv2D(filters = 64, kernel_size=5, strides=1, padding='same', activation='relu', name='conv1')(input_dim)

        # Reshape layer to be 1 capsule x [filters] atoms
        _, H, W, C = conv1.get_shape()
        conv1_reshaped = layers.Reshape((H.value, W.value, 1, C.value))(conv1)

        # Layer 1: Primary Capsule: Conv cap with routing 1
        primary_caps = ConvCapsuleLayer(kernel_size=5, num_capsule=2, num_atoms = 64, strides=2, padding='same',
                                        routings=1, name='primarycaps')(conv1_reshaped)

        conv_cap_1 = ConvCapsuleLayer(kernel_size = 5, num_capsule = 2, num_atoms = 64, strides = 1, padding = 'same',
                                        routings = 3, name = 'conv_cap_1')(primary_caps)

        concat_1 = layers.Concatenate(axis = -2, name = 'concat_1')([primary_caps, conv_cap_1])

        conv_cap_1_2 = ConvCapsuleLayer(kernel_size = 5, num_capsule = 2, num_atoms = 64, strides = 1, padding = 'same',
                                        routings = 1, name = 'conv_cap_1_2')(concat_1)

        concat_1_2 = layers.Concatenate(axis = -2, name = 'concat_1_2')([primary_caps, conv_cap_1, conv_cap_1_2])

        conv_cap_1_3 = ConvCapsuleLayer(kernel_size = 5, num_capsule = 2, num_atoms = 64, strides = 1, padding = 'same',
                                        routings = 1, name = 'conv_cap_1_3')(concat_1_2)

        add_1 = layers.Add(name = 'add_1')([conv_cap_1_3, primary_caps])


        #Layer-2:
        conv_cap_2 = ConvCapsuleLayer(kernel_size = 5, num_capsule = 4, num_atoms = 32, strides = 2, padding = 'same',
                                        routings = 3, name = 'conv_cap_2')(add_1)

        conv_cap_2_2 = ConvCapsuleLayer(kernel_size = 5, num_capsule = 4, num_atoms = 32, strides = 1, padding = 'same',
                                        routings = 1, name = 'conv_cap_2_2')(conv_cap_2)

        concat_2 = layers.Concatenate(axis = -2, name = 'concat_2')([conv_cap_2_2, conv_cap_2])

        conv_cap_2_3 = ConvCapsuleLayer(kernel_size = 5, num_capsule = 4, num_atoms = 32, strides = 1, padding = 'same',
                                        routings = 1, name = 'conv_cap_2_3')(concat_2)

        concat_2_2 = layers.Concatenate(axis = -2, name = 'concat_2_2')([conv_cap_2_3, conv_cap_2_2, conv_cap_2])

        conv_cap_2_4 = ConvCapsuleLayer(kernel_size = 5, num_capsule = 4, num_atoms = 32, strides = 1, padding = 'same',
                                        routings = 1, name = 'conv_cap_2_4')(concat_2_2)

        add_2 = layers.Add(name = 'add_2')([conv_cap_2_4, conv_cap_2])

        #Layer-3:
        conv_cap_3 = ConvCapsuleLayer(kernel_size = 5, num_capsule = 8, num_atoms = 32, strides = 2, padding = 'same',
                                         routings = 3, name = 'conv_cap_3')(add_2)

        conv_cap_3_2 = ConvCapsuleLayer(kernel_size = 5, num_capsule = 8, num_atoms = 32, strides = 1, padding = 'same',
                                        routings = 1, name = 'conv_cap_3_2')(conv_cap_3)

        concat_3 = layers.Concatenate(axis = -2, name = 'concat_3')([conv_cap_3_2, conv_cap_3])

        conv_cap_3_3 = ConvCapsuleLayer(kernel_size = 5, num_capsule = 8, num_atoms = 32, strides = 1, padding = 'same',
                                        routings = 1, name = 'conv_cap_3_3')(concat_3)

        concat_3_2 = layers.Concatenate(axis = -2, name = 'concat_3_2')([conv_cap_3_3, conv_cap_3_2, conv_cap_3])

        conv_cap_3_4 = ConvCapsuleLayer(kernel_size = 5, num_capsule = 8, num_atoms = 32, strides = 1, padding = 'same',
                                        routings = 1, name = 'conv_cap_3_4')(concat_3_2)

        add_3 = layers.Add(name = 'add_3')([conv_cap_3_4, conv_cap_3])

        #Layer-4:
        conv_cap_4 = ConvCapsuleLayer(kernel_size = 5, num_capsule = 8, num_atoms = 16, strides = 2, padding = 'same',
                                        routings = 3, name = 'conv_cap_4')(add_3)

        conv_cap_4_2 = ConvCapsuleLayer(kernel_size = 5, num_capsule = 8, num_atoms = 16, strides = 1, padding = 'same',
                                        routings = 1, name = 'conv_cap_4_2')(conv_cap_4)

        concat_4 = layers.Concatenate(axis = -2, name = 'concat_4')([conv_cap_4, conv_cap_4_2])

        conv_cap_4_3 = ConvCapsuleLayer(kernel_size = 5, num_capsule = 8, num_atoms = 16, strides = 1, padding = 'same',
                                        routings = 1, name = 'conv_cap_4_3')(concat_4)

        concat_4_2 = layers.Concatenate(axis = -2, name = 'concat_4_2')([conv_cap_4, conv_cap_4_2, conv_cap_4_3])

        conv_cap_4_4 = ConvCapsuleLayer(kernel_size = 5, num_capsule = 8, num_atoms = 16, strides = 1, padding = 'same',
                                        routings = 1, name = 'conv_cap_4_4')(concat_4_2)

        add_4 = layers.Add(name = 'add_4')([conv_cap_4_4, conv_cap_4])

        #////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

        conv_cap = ConvCapsuleLayer(kernel_size = 5, num_capsule = 8, num_atoms = 16, strides = 1, padding = 'same',
                                        routings = 1, name = 'conv_cap')(add_4)

        #////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

        #Deconvolution-Layer
        deconv_cap_1 = DeconvCapsuleLayer(kernel_size = 4, num_capsule = 4, num_atoms = 16, upsamp_type = 'deconv',
                                          scaling = 2, padding = 'same', routings = 3,
                                          name = 'deconv_cap_1')(conv_cap)

        deconv_cap_2 = ConvCapsuleLayer(kernel_size = 5, num_capsule = 4, num_atoms = 16, strides = 1, padding = 'same',
                                        routings = 1, name = 'deconv_cap_2')(deconv_cap_1)

        deconv_cap_3 = DeconvCapsuleLayer(kernel_size = 4, num_capsule = 2, num_atoms = 32, upsamp_type = 'deconv',
                                          scaling = 2, padding = 'same', routings = 3,
                                          name = 'deconv_cap_3')(deconv_cap_2)

        deconv_cap_4 = ConvCapsuleLayer(kernel_size = 5, num_capsule = 4, num_atoms = 32, padding = 'same',
                                        routings = 1, name = 'deconv_cap_4')(deconv_cap_3)

        deconv_cap_5 = DeconvCapsuleLayer(kernel_size = 4, num_capsule = 2, num_atoms = 32, upsamp_type = 'deconv',
                                          scaling = 2, padding = 'same', routings = 3,
                                          name = 'deconv_cap_5')(deconv_cap_4)

        deconv_cap_6 = ConvCapsuleLayer(kernel_size = 5, num_capsule = 4, num_atoms = 32, strides = 1, padding = 'same',
                                        routings = 1, name = 'deconv_cap_6')(deconv_cap_5)

        deconv_cap_7 = DeconvCapsuleLayer(kernel_size = 4, num_capsule = 2, num_atoms = 64, upsamp_type = 'deconv',
                                          scaling = 2, padding = 'same', routings = 3,
                                          name = 'deconv_cap_7')(deconv_cap_6)

        final_cap = ConvCapsuleLayer(kernel_size = 1, num_capsule = 1, num_atoms = n_classes, strides = 1, padding = 'same',
                                         routings = 3, name = 'final_cap')(deconv_cap_7)

        out_seg = Length(n_classes, seg=True, name='out_seg')(final_cap)
        #plot_model(deconv_cap_5, to_file='rdcap.png')


        #result = Model(inputs = input_dim, outputs = out_seg)
        #print(result.summary())

        return out_seg


    def initModel_2D(self):

        height, width, depth = self.img_shape
        input_dimension = Input(shape = (height, width, depth))
        net_op = self.RDNetCAP(input_dimension)
        net_model = Model(inputs = input_dimension, outputs = net_op)
        adam = Adam(lr = 1e-4)

        net_model.compile(optimizer = adam, loss = 'binary_crossentropy', metrics = ['accuracy'])

        return net_model
