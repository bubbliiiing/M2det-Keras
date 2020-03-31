from keras.layers import Input
from m2det import M2DET
from PIL import Image

m2det = M2DET()

while True:
    img = input('Input image filename:')
    try:
        image = Image.open(img)
    except:
        print('Open Error! Try again!')
        continue
    else:
        r_image = m2det.detect_image(image)
        r_image.show()
m2det.close_session()
    