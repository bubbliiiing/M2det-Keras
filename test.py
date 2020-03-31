from nets.M2det import m2det
model = m2det(21,None, name='m2det')
model.summary()
for i in range(len(model.layers)):
    print(i,model.layers[i].name)