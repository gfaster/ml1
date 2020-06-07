import tensorflow as tf 
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import random

config = tf.compat.v1.ConfigProto()
config.gpu_options.allow_growth = True
session = tf.compat.v1.Session(config=config)


mnist = tf.keras.datasets.mnist # handwriting database

(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0 # converting the samples to floating point

# something of note, basically all variables are stored as matricies

model = tf.keras.models.Sequential([
	tf.keras.layers.Flatten(input_shape=(28,28)), # serializing the images
	tf.keras.layers.Dense(256, activation='relu'), # first layer, not sure about activation
	tf.keras.layers.Dropout(0.1), # not sure what this does
	tf.keras.layers.Dense(64, activation='relu'),
	tf.keras.layers.Dropout(0.1),
	tf.keras.layers.Dense(10) # output layer
	])

predictions = model(x_train[:1]).numpy() # logit scores, inputs to a logistic funciton

# print(tf.nn.softmax(predictions).numpy()) # converts the logits to probabilites

loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True) #gives a loss value (think error) from logits

print(loss_fn(y_train[:1], predictions).numpy())

model.compile(optimizer='adam',
              loss=loss_fn,
              metrics=['accuracy'])

model.fit(x_train, y_train, epochs=15)

model.evaluate(x_test,  y_test, verbose=2)

probability_model = tf.keras.Sequential([ # writing it out converting the logits to probability values
  model,
  tf.keras.layers.Softmax()
])


# testvar = random.randint(1, 100)

# print('I think it is: ' + str(int(tf.math.argmax(probability_model(x_test[testvar:testvar+1])[0]))))

# # show the tested image
# image_to_display = x_test[testvar]
# image_to_display = np.array(image_to_display * 255, dtype=np.uint8)
# pixels = image_to_display.reshape((28, 28))
# img = Image.fromarray(pixels, "L")
# img.show()

export_path = "C:\\Users\\gfast\\Desktop\\lil projects\\ml1\\build\\"


# print(model(x_train[:1]))
probability_model.save(export_path)
# loaded_model = tf.keras.models.load_model(export_path)
# print(loaded_model(x_train[:1]))