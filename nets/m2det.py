import keras
import tensorflow as tf

from nets.vgg import VGG16


def conv2d(inputs, filters, kernel_size, strides, padding, name='conv'):
    conv    = keras.layers.Conv2D(filters=filters, kernel_size=kernel_size, strides=strides, padding=padding, name=name+'_conv')(inputs)
    bn      = keras.layers.BatchNormalization(name=name+'_BN')(conv)
    relu    = keras.layers.Activation('relu',name=name)(bn)
    return relu

def FFMv1(C4, C5, feature_size_1=256, feature_size_2=512, name='FFMv1'):
    #------------------------------------------------#
    #   C4特征层      40,40,512
    #   C5特征层      20,20,1024
    #------------------------------------------------#

    # 40,40,512 -> 40,40,256
    F4 = conv2d(C4, filters=feature_size_1, kernel_size=(3, 3), strides=(1, 1), padding='same', name='F4')

    # 20,20,1024 -> 20,20,512
    F5 = conv2d(C5, filters=feature_size_2, kernel_size=(1, 1), strides=(1, 1), padding='same', name='F5')
    # 20,20,512 -> 40,40,512
    F5 = keras.layers.UpSampling2D(size=(2, 2), name='F5_Up')(F5)

    # 40,40,256 + 40,40,512 -> 40,40,768
    outputs = keras.layers.Concatenate(name=name)([F4, F5])
    return outputs

def FFMv2(stage, base, tum, base_size=(40,40,768), tum_size=(40,40,128), feature_size=128, name='FFMv2'):
    # 40,40,128
    outputs = conv2d(base, filters=feature_size, kernel_size=(1, 1), strides=(1, 1), padding='same', name=name+"_"+str(stage) + '_base_feature')
    outputs = keras.layers.Concatenate(name=name+"_"+str(stage))([outputs, tum])
    # 40,40,256
    return outputs

def TUM(stage, inputs, feature_size=256, name="TUM"):
    #---------------------------------#
    #   进行下采样的部分
    #---------------------------------#
    # 40,40,256
    f1 = inputs
    # 40,40,256 -> 20,20,256
    f2 = conv2d(f1, filters=feature_size, kernel_size=(3, 3), strides=(2, 2), padding='same',name=name + "_" + str(stage) + '_f2')
    # 20,20,256 -> 10,10,256
    f3 = conv2d(f2, filters=feature_size, kernel_size=(3, 3), strides=(2, 2), padding='same',name=name + "_" + str(stage) + '_f3')
    # 10,10,256 -> 5,5,256   
    f4 = conv2d(f3, filters=feature_size, kernel_size=(3, 3), strides=(2, 2), padding='same',name=name + "_" + str(stage) + '_f4')
    # 5,5,256 -> 3,3,256
    f5 = conv2d(f4, filters=feature_size, kernel_size=(3, 3), strides=(2, 2), padding='same',name=name + "_" + str(stage) + '_f5')
    # 3,3,256 -> 1,1,256
    f6 = conv2d(f5, filters=feature_size, kernel_size=(3, 3), strides=(2, 2), padding='valid',name=name + "_" + str(stage) + '_f6')

    size_buffer = []
    # 40,40
    size_buffer.append([int(f1.shape[1]), int(f1.shape[2])])
    # 20,20
    size_buffer.append([int(f2.shape[1]), int(f2.shape[2])])
    # 10,10
    size_buffer.append([int(f3.shape[1]), int(f3.shape[2])])
    # 5,5
    size_buffer.append([int(f4.shape[1]), int(f4.shape[2])])
    # 3,3
    size_buffer.append([int(f5.shape[1]), int(f5.shape[2])])
    
    #---------------------------------#
    #   进行上采样与特征融合的部分
    #---------------------------------#
    c6 = f6
    # 1,1,256 -> 1,1,256
    c5 = conv2d(c6, filters=feature_size, kernel_size=(3, 3), strides=(1, 1), padding='same',name=name + "_" + str(stage) + '_c5')
    # 1,1,256 -> 3,3,256
    c5 = keras.layers.Lambda(lambda x: tf.image.resize_bilinear(x, size=size_buffer[4]), name=name + "_" + str(stage) + '_upsample_add5')(c5)
    c5 = keras.layers.Add()([c5, f5])
 
    # 3,3,256 -> 3,3,256
    c4 = conv2d(c5, filters=feature_size, kernel_size=(3, 3), strides=(1, 1), padding='same', name=name + "_" + str(stage) + '_c4')
    # 3,3,256 -> 5,5,256
    c4 = keras.layers.Lambda(lambda x: tf.image.resize_bilinear(x, size=size_buffer[3]), name=name + "_" + str(stage) + '_upsample_add4')(c4)
    c4 = keras.layers.Add()([c4, f4])

    # 5,5,256 -> 5,5,256
    c3 = conv2d(c4, filters=feature_size, kernel_size=(3, 3), strides=(1, 1), padding='same', name=name + "_" + str(stage) + '_c3')
    # 5,5,256 -> 10,10,256
    c3 = keras.layers.Lambda(lambda x: tf.image.resize_bilinear(x, size=size_buffer[2]), name=name + "_" + str(stage) + '_upsample_add3')(c3)
    c3 = keras.layers.Add()([c3, f3])

    # 10,10,256 -> 10,10,256
    c2 = conv2d(c3, filters=feature_size, kernel_size=(3, 3), strides=(1, 1), padding='same', name=name + "_" + str(stage) + '_c2')
    # 10,10,256 -> 20,20,256
    c2 = keras.layers.Lambda(lambda x: tf.image.resize_bilinear(x, size=size_buffer[1]), name=name + "_" + str(stage) + '_upsample_add2')(c2)
    c2 = keras.layers.Add()([c2, f2])

    # 20,20,256 -> 20,20,256
    c1 = conv2d(c2, filters=feature_size, kernel_size=(3, 3), strides=(1, 1), padding='same', name=name + "_" + str(stage) + '_c1')
    # 20,20,256 -> 40,40,256
    c1 = keras.layers.Lambda(lambda x: tf.image.resize_bilinear(x, size=size_buffer[0]), name=name + "_" + str(stage) + '_upsample_add1')(c1)
    c1 = keras.layers.Add()([c1, f1])

    #---------------------------------#
    #   利用1x1卷积调整通道数后输出
    #---------------------------------#
    output_features = feature_size // 2
    # 40,40,256 -> 40,40,128 
    o1 = conv2d(c1, filters=output_features, kernel_size=(1, 1), strides=(1, 1), padding='valid',name=name + "_" + str(stage) + '_o1')
    # 20,20,256 -> 20,20,128
    o2 = conv2d(c2, filters=output_features, kernel_size=(1, 1), strides=(1, 1), padding='valid',name=name + "_" + str(stage) + '_o2')
    # 10,10,256 -> 10,10,128
    o3 = conv2d(c3, filters=output_features, kernel_size=(1, 1), strides=(1, 1), padding='valid',name=name + "_" + str(stage) + '_o3')
    # 5,5,256 -> 5,5,128
    o4 = conv2d(c4, filters=output_features, kernel_size=(1, 1), strides=(1, 1), padding='valid',name=name + "_" + str(stage) + '_o4')
    # 3,3,256 -> 3,3,128
    o5 = conv2d(c5, filters=output_features, kernel_size=(1, 1), strides=(1, 1), padding='valid',name=name + "_" + str(stage) + '_o5')
    # 1,1,256 -> 1,1,128
    o6 = conv2d(c6, filters=output_features, kernel_size=(1, 1), strides=(1, 1), padding='valid',name=name + "_" + str(stage) + '_o6')

    outputs = [o1, o2, o3, o4, o5, o6]

    return outputs

def _create_feature_pyramid(base_feature, stage=8):
    features = [[], [], [], [], [], []]
    #-------------------------------------------------#
    #   利用卷积对输入进来的特征层进行通道数的调整
    #   40,40,768 -> 40,40,256
    #-------------------------------------------------#
    TUM_input_feature = keras.layers.Conv2D(filters=256, kernel_size=1, strides=1, padding='same')(base_feature)
    #-------------------------------------------------#
    #   第一个TUM模块，可以获得六个有效特征层，分别是
    #   o1  40,40,128
    #   o2  20,20,128
    #   o3  10,10,128
    #   o4  5,5,128
    #   o5  3,3,128
    #   o6  1,1,128
    #-------------------------------------------------#
    outputs = TUM(1, TUM_input_feature)

    max_output = outputs[0]
    for j in range(len(features)):
        features[j].append(outputs[j])

    #-------------------------------------------------------------------------------------------#
    #   构建第2,3,4个TUM模块，需要将上一个Tum模块输出的40x40x128的内容，传入到下一个Tum模块中
    #-------------------------------------------------------------------------------------------#
    for i in range(2, stage+1):
        #------------------------------------------------------#
        #   将基础特征层和Tum模块的输出传入到FFmv2层当中
        #   输入为base_feature 40x40x768，max_output 40x40x128
        #   输出的TUM_input_feature为40x40x256的特征层
        #------------------------------------------------------#
        TUM_input_feature = FFMv2(i - 1, base_feature, max_output)
        #-------------------------------------------------#
        #   TUM可以获得六个有效特征层，分别是
        #   o1  40,40,128
        #   o2  20,20,128
        #   o3  10,10,128
        #   o4  5,5,128
        #   o5  3,3,128
        #   o6  1,1,128
        #-------------------------------------------------#
        outputs = TUM(i, TUM_input_feature)

        max_output = outputs[0]
        for j in range(len(features)):
            features[j].append(outputs[j])

    #-------------------------------------------------#
    #   进行了4次TUM
    #   将获得的同样大小的特征层堆叠到一起
    #-------------------------------------------------#
    concatenate_features = []
    for feature in features:
        concat = keras.layers.Concatenate()([f for f in feature])
        concatenate_features.append(concat)
    return concatenate_features

#-------------------------------------------------#
#   注意力机制
#-------------------------------------------------#
def SE_block(inputs, compress_ratio=16, name='SE_block'):
    num_filters = int(inputs.shape[3])

    pool = keras.layers.GlobalAveragePooling2D()(inputs)
    reshape = keras.layers.Reshape((1, 1, -1))(pool)

    fc1 = keras.layers.Conv2D(filters=num_filters // compress_ratio, kernel_size=1, strides=1, padding='valid',
                              activation='relu', name=name+'_fc1')(reshape)
    fc2 = keras.layers.Conv2D(filters=num_filters, kernel_size=1, strides=1, padding='valid', activation='sigmoid',
                              name=name+'_fc2')(fc1)

    reweight = keras.layers.Multiply(name=name+'_reweight')([inputs, fc2])
    return reweight

#-------------------------------------------------#
#   给合并后的特征层添加上注意力机制
#-------------------------------------------------#
def SFAM(feature_pyramid, compress_ratio=16):
    outputs = []
    for i in range(len(feature_pyramid)):
        _output = SE_block(feature_pyramid[i], compress_ratio=compress_ratio, name='SE_block_' + str(i))
        outputs.append(_output)
    return outputs

def m2det(input_shape, num_classes=21, num_anchors = 6):
    inputs = keras.layers.Input(shape=input_shape)
    #------------------------------------------------#
    #   利用主干特征提取网络获得两个有效特征层
    #   分别是C4      40,40,512
    #   分别是C5      20,20,1024
    #------------------------------------------------#
    C4, C5 = VGG16(inputs).outputs[2:]

    # base_feature的shape为40,40,768
    base_feature = FFMv1(C4, C5, feature_size_1=256, feature_size_2=512)

    #---------------------------------------------------------------------------------------------------#
    #   在_create_feature_pyramid函数里，我们会使用TUM模块对输入进来的特征层进行特征提取
    #   最终输出的特征层有六个，由于进行了四次的TUM模块，所以六个有效特征层由4次TUM模块的输出堆叠而成
    #   o1      40,40,128*4         40,40,512
    #   o2      20,20,128*4         20,20,512
    #   o3      10,10,128*4         10,10,512
    #   o4      5,5,128*4           5,5,512
    #   o5      3,3,128*4           3,3,512
    #   o6      1,1,128*4           1,1,512
    #---------------------------------------------------------------------------------------------------#
    feature_pyramid = _create_feature_pyramid(base_feature, stage=4)

    #-------------------------------------------------#
    #   给合并后的特征层添加上注意力机制
    #-------------------------------------------------#
    outputs = SFAM(feature_pyramid)

    #-------------------------------------------------#
    #   将有效特征层转换成输出结果
    #-------------------------------------------------#
    classifications = []
    regressions = []
    for feature in outputs:
        classification = keras.layers.Conv2D(filters = num_anchors * num_classes, kernel_size=3, strides=1, padding='same')(feature)
        classification = keras.layers.Reshape((-1, num_classes))(classification)
        classification = keras.layers.Activation('softmax')(classification)

        regression = keras.layers.Conv2D(filters = num_anchors * 4, kernel_size=3, strides=1, padding='same')(feature)
        regression = keras.layers.Reshape((-1, 4))(regression)

        classifications.append(classification)
        regressions.append(regression)
    
    classifications = keras.layers.Concatenate(axis=1, name="classification")(classifications)
    regressions     = keras.layers.Concatenate(axis=1, name="regression")(regressions)

    pyramids        = keras.layers.Concatenate(axis=-1, name="out")([regressions, classifications])
    return keras.models.Model(inputs=inputs, outputs=pyramids)