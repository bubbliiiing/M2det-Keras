#--------------------------------------------#
#   该部分代码只用于看网络结构，并非测试代码
#   map测试请看get_dr_txt.py、get_gt_txt.py
#   和get_map.py
#--------------------------------------------#
from nets.M2det import m2det

if __name__ == "__main__":
    input_shape = [512, 512, 3]
    model = m2det(21, input_shape, name='m2det')
    model.summary()
    for i in range(len(model.layers)):
        print(i,model.layers[i].name)
