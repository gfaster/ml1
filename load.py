import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import tensorflow as tf
import numpy as np
from PIL import Image, ImageOps
import random
import sys

config = tf.compat.v1.ConfigProto()
config.gpu_options.allow_growth = True
session = tf.compat.v1.Session(config=config)

import_path = "C:\\Users\\gfast\\Desktop\\lil projects\\ml1\\build"
image_path = ''

if len(sys.argv) == 2:
	image_path = sys.argv[1]
else:
	exit('Must specify image')


raw_image = Image.open(image_path)



# from https://note.nkmk.me/en/python-pillow-add-margin-expand-canvas/
# this makes sure the image is the right size
def expand2square(pil_img):
    width, height = pil_img.size
    if width == height:
        return pil_img
    elif width > height:
        result = Image.new('L', (width, width))
        result.paste(pil_img, (0, (width - height) // 2))
        return result
    else:
        result = Image.new('L', (height, height))
        result.paste(pil_img, ((height - width) // 2, 0))
        return result

raw_image = raw_image.convert('RGBA')
datas = raw_image.getdata()
newData = []
for item in datas:
    if item[3] < 10:
        newData.append((255, 255, 255, 255))
    else:
        newData.append(item)

raw_image.putdata(newData)
raw_image = raw_image.convert('L')

raw_image.thumbnail((28,28), Image.ANTIALIAS)


final_image = expand2square(ImageOps.invert(raw_image))

final_image.show()
image_data = np.array(np.asarray(final_image)/255,ndmin=3)
# print(image_data)

model = tf.keras.models.load_model(import_path)

out_tensor = model(image_data)
guess = int(tf.math.argmax(out_tensor[-1]))
# print(model(image_data).numpy()[-1])
print(f'I think it is: {guess} with {round(float(out_tensor[-1][guess]) * 100,3)}% confidence!')


