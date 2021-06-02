import colorsys
import os
import time

import numpy as np
from keras import backend as K
from keras.applications.imagenet_utils import preprocess_input
from PIL import Image, ImageDraw, ImageFont

import nets.M2det as M2det
from utils.anchors import get_anchors
from utils.utils import BBoxUtility, letterbox_image, m2det_correct_boxes


#--------------------------------------------#
#   使用自己训练好的模型预测需要修改2个参数
#   model_path和classes_path都需要修改！
#   如果出现shape不匹配，一定要注意
#   训练时的model_path和classes_path参数的修改
#--------------------------------------------#
class M2DET(object):
    _defaults = {
        "model_path"        : 'model_data/M2det_weights.h5',
        "classes_path"      : 'model_data/voc_classes.txt',
        "model_image_size"  : (320, 320, 3),
        "nms_iou"           : 0.45,
        "confidence"        : 0.5,
        "anchors_size"      : [0.08, 0.15, 0.33, 0.51, 0.69, 0.87, 1.05],
        #---------------------------------------------------------------------#
        #   该变量用于控制是否使用letterbox_image对输入图像进行不失真的resize，
        #   在多次测试后，发现关闭letterbox_image直接resize的效果更好
        #---------------------------------------------------------------------#
        "letterbox_image"   : False,

    }

    @classmethod
    def get_defaults(cls, n):
        if n in cls._defaults:
            return cls._defaults[n]
        else:
            return "Unrecognized attribute name '" + n + "'"

    #---------------------------------------------------#
    #   初始化M2det
    #---------------------------------------------------#
    def __init__(self, **kwargs):
        self.__dict__.update(self._defaults)
        self.class_names = self._get_class()
        self.sess = K.get_session()
        self.generate()
        self.prior = self._get_prior()
        self.bbox_util = BBoxUtility(self.num_classes)

    #---------------------------------------------------#
    #   获得所有的分类
    #---------------------------------------------------#
    def _get_class(self):
        classes_path = os.path.expanduser(self.classes_path)
        with open(classes_path) as f:
            class_names = f.readlines()
        class_names = [c.strip() for c in class_names]
        return class_names

    def _get_prior(self):
        data = get_anchors((self.model_image_size[0], self.model_image_size[1]), anchors_size=self.anchors_size)
        return data

    #---------------------------------------------------#
    #   载入模型
    #---------------------------------------------------#
    def generate(self):
        model_path = os.path.expanduser(self.model_path)
        assert model_path.endswith('.h5'), 'Keras model or weights must be a .h5 file.'
        
        #-------------------------------#
        #   计算总的类的数量
        #-------------------------------#
        self.num_classes = len(self.class_names)+1

        #-------------------------------#
        #   载入模型与权值
        #-------------------------------#
        self.m2det = M2det.m2det(self.num_classes, self.model_image_size)
        self.m2det.load_weights(self.model_path)
        print('{} model, anchors, and classes loaded.'.format(model_path))

        # 画框设置不同的颜色
        hsv_tuples = [(x / len(self.class_names), 1., 1.)
                      for x in range(len(self.class_names))]
        self.colors = list(map(lambda x: colorsys.hsv_to_rgb(*x), hsv_tuples))
        self.colors = list(
            map(lambda x: (int(x[0] * 255), int(x[1] * 255), int(x[2] * 255)),
                self.colors))

    #---------------------------------------------------#
    #   检测图片
    #---------------------------------------------------#
    def detect_image(self, image):
        image_shape = np.array(np.shape(image)[0:2])
        #---------------------------------------------------------#
        #   给图像增加灰条，实现不失真的resize
        #   也可以直接resize进行识别
        #---------------------------------------------------------#
        if self.letterbox_image:
            crop_img = np.array(letterbox_image(image, (self.model_image_size[1],self.model_image_size[0])))
        else:
            crop_img = image.convert('RGB')
            crop_img = crop_img.resize((self.model_image_size[1],self.model_image_size[0]), Image.BICUBIC)
        photo = np.array(crop_img,dtype = np.float64)
        
        #-----------------------------------------------------------#
        #   图片预处理，归一化。
        #-----------------------------------------------------------#
        photo = preprocess_input(np.reshape(photo,[1,self.model_image_size[0],self.model_image_size[1],self.model_image_size[2]]))

        preds = self.m2det.predict(photo)
        #-----------------------------------------------------------#
        #   将预测结果进行解码
        #-----------------------------------------------------------#
        results = self.bbox_util.detection_out(preds,self.prior,confidence_threshold=self.confidence)

        #--------------------------------------#
        #   如果没有检测到物体，则返回原图
        #--------------------------------------#
        if len(results[0])<=0:
            return image

        #-----------------------------------------------------------#
        #   筛选出其中得分高于confidence的框 
        #-----------------------------------------------------------#
        det_label = results[0][:, 0]
        det_conf = results[0][:, 1]
        det_xmin, det_ymin, det_xmax, det_ymax = results[0][:, 2], results[0][:, 3], results[0][:, 4], results[0][:, 5]
        top_indices = [i for i, conf in enumerate(det_conf) if conf >= self.confidence]
        top_conf = det_conf[top_indices]
        top_label_indices = det_label[top_indices].tolist()
        top_xmin, top_ymin, top_xmax, top_ymax = np.expand_dims(det_xmin[top_indices],-1),np.expand_dims(det_ymin[top_indices],-1),np.expand_dims(det_xmax[top_indices],-1),np.expand_dims(det_ymax[top_indices],-1)
        
        #-----------------------------------------------------------#
        #   去掉灰条部分
        #-----------------------------------------------------------#
        if self.letterbox_image:
            boxes = m2det_correct_boxes(top_ymin,top_xmin,top_ymax,top_xmax,np.array([self.model_image_size[0],self.model_image_size[1]]),image_shape)
        else:
            top_xmin = top_xmin * image_shape[1]
            top_ymin = top_ymin * image_shape[0]
            top_xmax = top_xmax * image_shape[1]
            top_ymax = top_ymax * image_shape[0]
            boxes = np.concatenate([top_ymin,top_xmin,top_ymax,top_xmax], axis=-1)
            
        font = ImageFont.truetype(font='model_data/simhei.ttf',size=np.floor(3e-2 * np.shape(image)[1] + 0.5).astype('int32'))

        thickness = max((np.shape(image)[0] + np.shape(image)[1]) // self.model_image_size[0], 1)

        for i, c in enumerate(top_label_indices):
            predicted_class = self.class_names[int(c-1)]
            score = top_conf[i]

            top, left, bottom, right = boxes[i]
            top = top - 5
            left = left - 5
            bottom = bottom + 5
            right = right + 5

            top = max(0, np.floor(top + 0.5).astype('int32'))
            left = max(0, np.floor(left + 0.5).astype('int32'))
            bottom = min(np.shape(image)[0], np.floor(bottom + 0.5).astype('int32'))
            right = min(np.shape(image)[1], np.floor(right + 0.5).astype('int32'))

            # 画框框
            label = '{} {:.2f}'.format(predicted_class, score)
            draw = ImageDraw.Draw(image)
            label_size = draw.textsize(label, font)
            label = label.encode('utf-8')
            print(label, top, left, bottom, right)
            
            if top - label_size[1] >= 0:
                text_origin = np.array([left, top - label_size[1]])
            else:
                text_origin = np.array([left, top + 1])

            for i in range(thickness):
                draw.rectangle(
                    [left + i, top + i, right - i, bottom - i],
                    outline=self.colors[int(c-1)])
            draw.rectangle(
                [tuple(text_origin), tuple(text_origin + label_size)],
                fill=self.colors[int(c-1)])
            draw.text(text_origin, str(label,'UTF-8'), fill=(0, 0, 0), font=font)
            del draw
        return image

    def get_FPS(self, image, test_interval):
        image_shape = np.array(np.shape(image)[0:2])
        #---------------------------------------------------------#
        #   给图像增加灰条，实现不失真的resize
        #   也可以直接resize进行识别
        #---------------------------------------------------------#
        if self.letterbox_image:
            crop_img = np.array(letterbox_image(image, (self.model_image_size[1],self.model_image_size[0])))
        else:
            crop_img = image.convert('RGB')
            crop_img = crop_img.resize((self.model_image_size[1],self.model_image_size[0]), Image.BICUBIC)
        photo = np.array(crop_img,dtype = np.float64)
        photo = preprocess_input(np.reshape(photo,[1,self.model_image_size[0],self.model_image_size[1],self.model_image_size[2]]))

        preds = self.m2det.predict(photo)
        results = self.bbox_util.detection_out(preds,self.prior,confidence_threshold=self.confidence)

        if len(results[0])>0:
            det_label = results[0][:, 0]
            det_conf = results[0][:, 1]
            det_xmin, det_ymin, det_xmax, det_ymax = results[0][:, 2], results[0][:, 3], results[0][:, 4], results[0][:, 5]
            top_indices = [i for i, conf in enumerate(det_conf) if conf >= self.confidence]
            top_conf = det_conf[top_indices]
            top_label_indices = det_label[top_indices].tolist()
            top_xmin, top_ymin, top_xmax, top_ymax = np.expand_dims(det_xmin[top_indices],-1),np.expand_dims(det_ymin[top_indices],-1),np.expand_dims(det_xmax[top_indices],-1),np.expand_dims(det_ymax[top_indices],-1)
            #-----------------------------------------------------------#
            #   去掉灰条部分
            #-----------------------------------------------------------#
            if self.letterbox_image:
                boxes = m2det_correct_boxes(top_ymin,top_xmin,top_ymax,top_xmax,np.array([self.model_image_size[0],self.model_image_size[1]]),image_shape)
            else:
                top_xmin = top_xmin * image_shape[1]
                top_ymin = top_ymin * image_shape[0]
                top_xmax = top_xmax * image_shape[1]
                top_ymax = top_ymax * image_shape[0]
                boxes = np.concatenate([top_ymin,top_xmin,top_ymax,top_xmax], axis=-1)

        t1 = time.time()
        for _ in range(test_interval):
            preds = self.m2det.predict(photo)
            results = self.bbox_util.detection_out(preds,self.prior,confidence_threshold=self.confidence)

            if len(results[0])>0:
                det_label = results[0][:, 0]
                det_conf = results[0][:, 1]
                det_xmin, det_ymin, det_xmax, det_ymax = results[0][:, 2], results[0][:, 3], results[0][:, 4], results[0][:, 5]
                top_indices = [i for i, conf in enumerate(det_conf) if conf >= self.confidence]
                top_conf = det_conf[top_indices]
                top_label_indices = det_label[top_indices].tolist()
                top_xmin, top_ymin, top_xmax, top_ymax = np.expand_dims(det_xmin[top_indices],-1),np.expand_dims(det_ymin[top_indices],-1),np.expand_dims(det_xmax[top_indices],-1),np.expand_dims(det_ymax[top_indices],-1)
                #-----------------------------------------------------------#
                #   去掉灰条部分
                #-----------------------------------------------------------#
                if self.letterbox_image:
                    boxes = m2det_correct_boxes(top_ymin,top_xmin,top_ymax,top_xmax,np.array([self.model_image_size[0],self.model_image_size[1]]),image_shape)
                    
                else:
                    top_xmin = top_xmin * image_shape[1]
                    top_ymin = top_ymin * image_shape[0]
                    top_xmax = top_xmax * image_shape[1]
                    top_ymax = top_ymax * image_shape[0]
                    boxes = np.concatenate([top_ymin,top_xmin,top_ymax,top_xmax], axis=-1)

        t2 = time.time()
        tact_time = (t2 - t1) / test_interval
        return tact_time

    def close_session(self):
        self.sess.close()
