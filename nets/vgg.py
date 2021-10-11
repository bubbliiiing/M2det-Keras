from keras import Model
from keras.layers import Conv2D, MaxPooling2D

def VGG16(inputs):
    net = {} 
    #------------------------#
    #   输入默认为320,320,3
    #------------------------#
    net['input'] = inputs

    #------------------------------------------------#
    #   第一个卷积部分  320,320,3 -> 160,160,64
    #------------------------------------------------#
    net['conv1_1'] = Conv2D(64, kernel_size=(3,3),
                                   activation='relu',
                                   padding='same',
                                   name='conv1_1')(net['input'])
    net['conv1_2'] = Conv2D(64, kernel_size=(3,3),
                                   activation='relu',
                                   padding='same',
                                   name='conv1_2')(net['conv1_1'])
    net['pool1'] = MaxPooling2D((2, 2), strides=(2, 2), padding='same',
                                name='pool1')(net['conv1_2'])
    
    #------------------------------------------------#
    #   第二个卷积部分  160,160,64 -> 80,80,128
    #------------------------------------------------#
    net['conv2_1'] = Conv2D(128, kernel_size=(3,3),
                                   activation='relu',
                                   padding='same',
                                   name='conv2_1')(net['pool1'])
    net['conv2_2'] = Conv2D(128, kernel_size=(3,3),
                                   activation='relu',
                                   padding='same',
                                   name='conv2_2')(net['conv2_1'])
    net['pool2'] = MaxPooling2D((2, 2), strides=(2, 2), padding='same',
                                name='pool2')(net['conv2_2'])
    y0 = net['pool2']

    #------------------------------------------------#
    #   第三个卷积部分  80,80,128 -> 40,40,256
    #------------------------------------------------#
    net['conv3_1'] = Conv2D(256, kernel_size=(3,3),
                                   activation='relu',
                                   padding='same',
                                   name='conv3_1')(net['pool2'])
    net['conv3_2'] = Conv2D(256, kernel_size=(3,3),
                                   activation='relu',
                                   padding='same',
                                   name='conv3_2')(net['conv3_1'])
    net['conv3_3'] = Conv2D(256, kernel_size=(3,3),
                                   activation='relu',
                                   padding='same',
                                   name='conv3_3')(net['conv3_2'])
    net['pool3'] = MaxPooling2D((2, 2), strides=(2, 2), padding='same',
                                name='pool3')(net['conv3_3'])
    y1 = net['pool3']

    #------------------------------------------------#
    #   第四个卷积部分  40,40,256 -> 40,40,512
    #------------------------------------------------#
    net['conv4_1'] = Conv2D(512, kernel_size=(3,3),
                                   activation='relu',
                                   padding='same',
                                   name='conv4_1')(net['pool3'])
    net['conv4_2'] = Conv2D(512, kernel_size=(3,3),
                                   activation='relu',
                                   padding='same',
                                   name='conv4_2')(net['conv4_1'])
    net['conv4_3'] = Conv2D(512, kernel_size=(3,3),
                                   activation='relu',
                                   padding='same',
                                   name='conv4_3')(net['conv4_2'])
    # net['pool4'] = MaxPooling2D((2, 2), strides=(2, 2), padding='same',
    #                             name='block4_pool')(net['conv4_3'])
    y2 = net['conv4_3']
    
    #------------------------------------------------#
    #   第五个卷积部分  40,40,512 -> 20,20,1024
    #------------------------------------------------#
    net['conv5_1'] = Conv2D(1024, kernel_size=(3,3),
                                   activation='relu',
                                   padding='same',
                                   name='conv5_1')(net['conv4_3'])
    net['conv5_2'] = Conv2D(1024, kernel_size=(3,3),
                                   activation='relu',
                                   padding='same',
                                   name='conv5_2')(net['conv5_1'])
    net['conv5_3'] = Conv2D(1024, kernel_size=(3,3),
                                   activation='relu',
                                   padding='same',
                                   name='conv5_3')(net['conv5_2'])
    net['pool5'] = MaxPooling2D((3, 3), strides=(2, 2), padding='same',
                                name='pool5')(net['conv5_3'])
    y3 = net['pool5']

    model = Model(inputs, [y0,y1,y2,y3], name='vgg16')
    return model
