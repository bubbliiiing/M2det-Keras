import keras
import numpy as np
from keras.callbacks import (EarlyStopping, ModelCheckpoint, ReduceLROnPlateau,
                             TensorBoard)

import nets.M2det as M2det
from nets.M2det_training import Generator, LossHistory, conf_loss, smooth_l1
from utils.anchors import get_anchors
from utils.utils import BBoxUtility

#----------------------------------------------------#
#   检测精度mAP和pr曲线计算参考视频
#   https://www.bilibili.com/video/BV1zE411u7Vw
#----------------------------------------------------#
if __name__ == "__main__":
    annotation_path = '2007_train.txt'
    #----------------------------------------------------#
    #   训练之前一定要修改NUM_CLASSES
    #   修改成所需要区分的类的个数+1。
    #----------------------------------------------------#
    NUM_CLASSES = 21
    #----------------------------------------------------#
    #   输入图像大小
    #----------------------------------------------------#
    input_shape = (320, 320, 3)
    #----------------------------------------------------#
    #   获得先验框
    #----------------------------------------------------#
    anchors_size = [0.08, 0.15, 0.33, 0.51, 0.69, 0.87, 1.05]
    priors = get_anchors((input_shape[0],input_shape[1]), anchors_size)
    bbox_util = BBoxUtility(NUM_CLASSES, priors)

    model = M2det.m2det(NUM_CLASSES,input_shape)
    #------------------------------------------------------#
    #   权值文件请看README，百度网盘下载
    #   训练自己的数据集时提示维度不匹配正常
    #   预测的东西都不一样了自然维度不匹配
    #------------------------------------------------------#
    model.load_weights("model_data/M2det_weights.h5", by_name=True, skip_mismatch=True)

    #-------------------------------------------------------------------------------#
    #   训练参数的设置
    #   logging表示tensorboard的保存地址
    #   checkpoint用于设置权值保存的细节，period用于修改多少epoch保存一次
    #   reduce_lr用于设置学习率下降的方式
    #   early_stopping用于设定早停，val_loss多次不下降自动结束训练，表示模型基本收敛
    #-------------------------------------------------------------------------------#
    logging = TensorBoard(log_dir='logs/')
    checkpoint = ModelCheckpoint('logs/ep{epoch:03d}-loss{loss:.3f}-val_loss{val_loss:.3f}.h5',
        monitor='val_loss', save_weights_only=True, save_best_only=False, period=1)
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, verbose=1)
    early_stopping = EarlyStopping(monitor='val_loss', min_delta=0, patience=10, verbose=1)
    loss_history = LossHistory('logs/')

    #----------------------------------------------------------------------#
    #   验证集的划分在train.py代码里面进行
    #   2007_test.txt和2007_val.txt里面没有内容是正常的。训练不会使用到。
    #   当前划分方式下，验证集和训练集的比例为1:9
    #----------------------------------------------------------------------#
    val_split = 0.1
    with open(annotation_path) as f: 
        lines = f.readlines()
    np.random.seed(10101)
    np.random.shuffle(lines)
    np.random.seed(None)
    num_val = int(len(lines)*val_split)
    num_train = len(lines) - num_val

    #------------------------------------------------------#
    #   主干特征提取网络特征通用，冻结训练可以加快训练速度
    #   也可以在训练初期防止权值被破坏。
    #   Init_Epoch为起始世代
    #   Freeze_Epoch为冻结训练的世代
    #   Epoch总训练世代
    #   提示OOM或者显存不足请调小Batch_size
    #------------------------------------------------------#
    for i in range(14):
        model.layers[i].trainable = False
    if True:
        Init_epoch          = 0
        Freeze_epoch        = 50
        Batch_size          = 8
        learning_rate_base  = 5e-4

        gen                 = Generator(bbox_util, Batch_size, lines[:num_train], lines[num_train:],
                                (input_shape[0], input_shape[1]),NUM_CLASSES)

        epoch_size          = num_train // Batch_size
        epoch_size_val      = num_val // Batch_size

        if epoch_size == 0 or epoch_size_val == 0:
            raise ValueError("数据集过小，无法进行训练，请扩充数据集。")

        model.compile(loss={
                    'regression'    : smooth_l1(),
                    'classification': conf_loss()
                },optimizer=keras.optimizers.Adam(lr=learning_rate_base)
        )

        model.fit_generator(gen.generate(True), 
                steps_per_epoch=epoch_size,
                validation_data=gen.generate(False),
                validation_steps=epoch_size_val,
                epochs=Freeze_epoch, 
                initial_epoch=Init_epoch,
                callbacks=[logging, checkpoint, reduce_lr, early_stopping, loss_history])

    for i in range(14):
        model.layers[i].trainable = True
    if True:
        Freeze_epoch        = 50
        Epoch               = 100
        Batch_size          = 4
        learning_rate_base  = 1e-4
        
        gen                 = Generator(bbox_util, Batch_size, lines[:num_train], lines[num_train:],
                                (input_shape[0], input_shape[1]),NUM_CLASSES)

        epoch_size          = num_train // Batch_size
        epoch_size_val      = num_val // Batch_size

        if epoch_size == 0 or epoch_size_val == 0:
            raise ValueError("数据集过小，无法进行训练，请扩充数据集。")

        model.compile(loss={
                    'regression'    : smooth_l1(),
                    'classification': conf_loss()
                },optimizer=keras.optimizers.Adam(lr=learning_rate_base)
        )

        model.fit_generator(gen.generate(True), 
                steps_per_epoch=epoch_size,
                validation_data=gen.generate(False),
                validation_steps=epoch_size_val,
                epochs=Epoch, 
                initial_epoch=Freeze_epoch,
                callbacks=[logging, checkpoint, reduce_lr, early_stopping, loss_history])
