#--------------------------------------------#
#   该部分代码用于看网络结构
#--------------------------------------------#
from nets.m2det import m2det

if __name__ == "__main__":
    input_shape = [320, 320, 3]
    num_classes = 21

    model = m2det(input_shape, num_classes)
    model.summary()

    # for i,layer in enumerate(model.layers):
    #     print(i,layer.name)
