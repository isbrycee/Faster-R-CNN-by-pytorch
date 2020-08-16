#-------------------------------------#
#       对单张图片进行预测
#-------------------------------------#
from frcnn import FRCNN
from PIL import Image

frcnn = FRCNN()

while True:
    img = input('Input image filename:')
    try:
        image = Image.open(img)
    except:
        print('Open Error! Try again!')
        continue
    else:
        r_image = frcnn.detect_image(image)
        r_image.save('img/street_result.jpg')
        r_image.show()
