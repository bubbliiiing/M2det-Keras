import nets.M2det as M2det
import numpy as np
import keras
from keras.optimizers import Adam
from nets.M2det_training import Generator
from nets.M2det_training import conf_loss,smooth_l1 
from keras.callbacks import TensorBoard, ModelCheckpoint, ReduceLROnPlateau, EarlyStopping
from utils.utils import BBoxUtility
from utils.anchors import get_anchors

if __name__ == "__main__":
    NUM_CLASSES = 21
    input_shape = (320, 320, 3)
    annotation_path = '2007_train.txt'

    inputs = keras.layers.Input(shape=input_shape)
    model = M2det.m2det(NUM_CLASSES,inputs)
    
    priors = get_anchors((input_shape[0],input_shape[1]))
    bbox_util = BBoxUtility(NUM_CLASSES, priors)

    model.load_weights("model_data\M2det_weights.h5",by_name=True,skip_mismatch=True)
    model.summary()
    # 0.1用于验证，0.9用于训练
    val_split = 0.1
    with open(annotation_path) as f: 
        lines = f.readlines()
    np.random.seed(10101)
    np.random.shuffle(lines)
    np.random.seed(None)
    num_val = int(len(lines)*val_split)
    num_train = len(lines) - num_val

    # 训练参数设置
    logging = TensorBoard(log_dir="logs")
    checkpoint = ModelCheckpoint('logs/ep{epoch:03d}-loss{loss:.3f}-val_loss{val_loss:.3f}.h5',
        monitor='val_loss', save_weights_only=True, save_best_only=False, period=1)
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=2, verbose=1)
    early_stopping = EarlyStopping(monitor='val_loss', min_delta=0, patience=6, verbose=1)

    BATCH_SIZE = 8
    gen = Generator(bbox_util, BATCH_SIZE, lines[:num_train], lines[num_train:],
                    (input_shape[0], input_shape[1]),NUM_CLASSES)

    for i in range(14):
        model.layers[i].trainable = False

    model.compile(loss={
                'regression'    : smooth_l1(),
                'classification': conf_loss()
            },optimizer=keras.optimizers.Adam(lr=1e-4,clipnorm=0.001)
    )

    model.fit_generator(    gen.generate(True), 
            steps_per_epoch=num_train//BATCH_SIZE,
            validation_data=gen.generate(False),
            validation_steps=num_val//BATCH_SIZE,
            epochs=60, 
            verbose=1,
            initial_epoch=20,
            callbacks=[logging, checkpoint, reduce_lr])

    for i in range(14):
        model.layers[i].trainable = True

    model.compile(loss={
                'regression'    : smooth_l1(),
                'classification': conf_loss()
            },optimizer=keras.optimizers.Adam(lr=1e-5,clipnorm=0.001)
    )
    model.fit_generator(    gen.generate(True), 
            steps_per_epoch=num_train//BATCH_SIZE,
            validation_data=gen.generate(False),
            validation_steps=num_val//BATCH_SIZE,
            epochs=100, 
            verbose=1,
            initial_epoch=13,
            callbacks=[logging, checkpoint, reduce_lr, early_stopping])
