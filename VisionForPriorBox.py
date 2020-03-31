import numpy as np
import pickle
import matplotlib.pyplot as plt

class PriorBox():
    def __init__(self, img_size, min_size, max_size=None, aspect_ratios=None,
                 flip=True, variances=[0.1], clip=True, **kwargs):

        self.waxis = 1
        self.haxis = 0

        self.img_size = img_size
        if min_size <= 0:
            raise Exception('min_size must be positive.')

        self.min_size = min_size
        self.max_size = max_size
        self.aspect_ratios = [1.0]
        if max_size:
            if max_size < min_size:
                raise Exception('max_size must be greater than min_size.')
            self.aspect_ratios.append(1.0)
        if aspect_ratios:
            for ar in aspect_ratios:
                if ar in self.aspect_ratios:
                    continue
                self.aspect_ratios.append(ar)
                if flip:
                    self.aspect_ratios.append(1.0 / ar)
        self.variances = np.array(variances)
        self.clip = True

    def compute_output_shape(self, input_shape):
        num_priors_ = len(self.aspect_ratios)
        layer_width = input_shape[self.waxis]
        layer_height = input_shape[self.haxis]
        num_boxes = num_priors_ * layer_width * layer_height
        return (input_shape[0], num_boxes, 8)

    def call(self, input_shape, mask=None):

        # 获取输入进来的特征层的宽与高
        # 3x3
        layer_width = input_shape[self.waxis]
        layer_height = input_shape[self.haxis]

        # 获取输入进来的图片的宽和高
        # 300x300
        img_width = self.img_size[0]
        img_height = self.img_size[1]

        # 获得先验框的宽和高
        box_widths = []
        box_heights = []
        for ar in self.aspect_ratios:
            if ar == 1 and len(box_widths) == 0:
                box_widths.append(self.min_size)
                box_heights.append(self.min_size)
            elif ar == 1 and len(box_widths) > 0:
                box_widths.append(np.sqrt(self.min_size * self.max_size))
                box_heights.append(np.sqrt(self.min_size * self.max_size))
            elif ar != 1:
                box_widths.append(self.min_size * np.sqrt(ar))
                box_heights.append(self.min_size / np.sqrt(ar))

        print("box_widths:",box_widths)
        print("box_heights:",box_heights)

        box_widths = 0.5 * np.array(box_widths)
        box_heights = 0.5 * np.array(box_heights)

        step_x = img_width / layer_width
        step_y = img_height / layer_height

        linx = np.linspace(0.5 * step_x, img_width - 0.5 * step_x,
                           layer_width)
        liny = np.linspace(0.5 * step_y, img_height - 0.5 * step_y,
                           layer_height)


        print("linx:",linx)
        print("liny:",liny)
        centers_x, centers_y = np.meshgrid(linx, liny)
        # 计算网格中心
        centers_x = centers_x.reshape(-1, 1)
        centers_y = centers_y.reshape(-1, 1)

                
        fig = plt.figure()
        ax = fig.add_subplot(111)
        plt.ylim(0,320)
        plt.xlim(0,320)
        plt.scatter(centers_x,centers_y)

        num_priors_ = len(self.aspect_ratios)
        # 每一个先验框需要两个(centers_x, centers_y)，前一个用来计算左上角，后一个计算右下角
        prior_boxes = np.concatenate((centers_x, centers_y), axis=1)
        prior_boxes = np.tile(prior_boxes, (1, 2 * num_priors_))
        
        # 获得先验框的左上角和右下角
        prior_boxes[:, ::4] -= box_widths
        prior_boxes[:, 1::4] -= box_heights
        prior_boxes[:, 2::4] += box_widths
        prior_boxes[:, 3::4] += box_heights

        rect1 = plt.Rectangle([prior_boxes[4, 0],prior_boxes[4, 1]],box_widths[0]*2,box_heights[0]*2,color="r",fill=False)
        rect2 = plt.Rectangle([prior_boxes[4, 4],prior_boxes[4, 5]],box_widths[1]*2,box_heights[1]*2,color="r",fill=False)
        rect3 = plt.Rectangle([prior_boxes[4, 8],prior_boxes[4, 9]],box_widths[2]*2,box_heights[2]*2,color="r",fill=False)
        rect4 = plt.Rectangle([prior_boxes[4, 12],prior_boxes[4, 13]],box_widths[3]*2,box_heights[3]*2,color="r",fill=False)
        
        ax.add_patch(rect1)
        ax.add_patch(rect2)
        ax.add_patch(rect3)
        ax.add_patch(rect4)

        plt.show()
        # 变成小数的形式
        prior_boxes[:, ::2] /= img_width
        prior_boxes[:, 1::2] /= img_height
        prior_boxes = prior_boxes.reshape(-1, 4)

        prior_boxes = np.minimum(np.maximum(prior_boxes, 0.0), 1.0)

        num_boxes = len(prior_boxes)
        
        if len(self.variances) == 1:
            variances = np.ones((num_boxes, 4)) * self.variances[0]
        elif len(self.variances) == 4:
            variances = np.tile(self.variances, (num_boxes, 1))
        else:
            raise Exception('Must provide one or four variances.')

        prior_boxes = np.concatenate((prior_boxes, variances), axis=1)
        return prior_boxes


if __name__ == '__main__':
    net = {} 
    #-----------------------将提取到的主干特征进行处理---------------------------#
    img_size = (320,320)
    length1 = img_size[0]*0.08
    length2 = img_size[0]*0.15
    length3 = img_size[0]*0.33
    length4 = img_size[0]*0.51
    length5 = img_size[0]*0.69
    length6 = img_size[0]*0.87
    length7 = img_size[0]*1.05
    # priorbox = PriorBox(img_size, length1,max_size = length2, aspect_ratios=[2, 3],
    #                     variances=[0.1, 0.1, 0.2, 0.2],
    #                     name='conv4_3_norm_mbox_priorbox')
    # net['conv4_3_norm_mbox_priorbox'] = priorbox.call([40,40])


    # priorbox = PriorBox(img_size, length2, max_size=length3, aspect_ratios=[2, 3],
    #                     variances=[0.1, 0.1, 0.2, 0.2],
    #                     name='fc7_mbox_priorbox')
    # net['fc7_mbox_priorbox'] = priorbox.call([20,20])


    # priorbox = PriorBox(img_size, length3, max_size=length4, aspect_ratios=[2, 3],
    #                     variances=[0.1, 0.1, 0.2, 0.2],
    #                     name='conv6_2_mbox_priorbox')
    # net['conv6_2_mbox_priorbox'] = priorbox.call([10,10])


    # priorbox = PriorBox(img_size, length4, max_size=length5, aspect_ratios=[2, 3],
    #                     variances=[0.1, 0.1, 0.2, 0.2],
    #                     name='conv7_2_mbox_priorbox')
    # net['conv7_2_mbox_priorbox'] = priorbox.call([5,5])


    priorbox = PriorBox(img_size, length5, max_size=length6, aspect_ratios=[2, 3],
                        variances=[0.1, 0.1, 0.2, 0.2],
                        name='conv8_2_mbox_priorbox')
    net['conv8_2_mbox_priorbox'] = priorbox.call([3,3])

    # priorbox = PriorBox(img_size, length6, max_size=length7, aspect_ratios=[2, 3],
    #                     variances=[0.1, 0.1, 0.2, 0.2],
    #                     name='pool6_mbox_priorbox')
                        
    # net['pool6_mbox_priorbox'] = priorbox.call([1,1])

    # net['mbox_priorbox'] = np.concatenate([net['conv4_3_norm_mbox_priorbox'],
    #                                 net['fc7_mbox_priorbox'],
    #                                 net['conv6_2_mbox_priorbox'],
    #                                 net['conv7_2_mbox_priorbox'],
    #                                 net['conv8_2_mbox_priorbox'],
    #                                 net['pool6_mbox_priorbox']],
    #                                 axis=0)

    # print(np.shape(net['mbox_priorbox']))
    # f = open('model_data/prior_boxes_rfb300.pkl', 'wb')
    # pickle.dump(net['mbox_priorbox'],f)
    # f.close()

    # f = open('model_data/prior_boxes_rfb300.pkl', 'rb')
    # e=pickle.load(f)