#--------------------------------------------#
#   该部分代码用于看网络结构
#--------------------------------------------#
from nets.m2det import m2det
from utils.utils import net_flops

if __name__ == "__main__":
    input_shape = [320, 320]
    num_classes = 21

    model = m2det([input_shape[0], input_shape[1], 3], num_classes)
    #--------------------------------------------#
    #   查看网络结构网络结构
    #--------------------------------------------#
    model.summary()
    #--------------------------------------------#
    #   计算网络的FLOPS
    #--------------------------------------------#
    net_flops(model, table=False)

    # for i,layer in enumerate(model.layers):
    #     print(i,layer.name)
