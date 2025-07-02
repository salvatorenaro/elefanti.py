import tensorflow as tf
from tensorflow.keras.applications import VGG16
import numpy as np
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.vgg16 import preprocess_input, decode_predictions
import matplotlib.pyplot as plt
import cv2


model = VGG16(weights='imagenet')
img_path = 'C:\\Users\\salva\\OneDrive\\Desktop\\machine-learning\\elefantiafricani.jpg'
img = image.load_img(img_path, target_size=(224, 224))
x = image.img_to_array(img)
x = np.expand_dims(x, axis=0)
x = preprocess_input(x)


predizione = model.predict(x)
decode = decode_predictions(predizione, top=3)[0]
print(decode)
print(np.argmax(predizione[0]))

african_elephant_index = 386  


last_conv_layer = model.get_layer('block5_conv3')
grad_model = tf.keras.models.Model(
    [model.inputs],
    [last_conv_layer.output, model.output]
)

with tf.GradientTape() as tape:
    inputs = tf.cast(x, tf.float32)
    conv_outputs, predictions = grad_model(inputs)
    loss = predictions[:, african_elephant_index]


grads = tape.gradient(loss, conv_outputs)
pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

conv_outputs = conv_outputs[0].numpy()
pooled_grads = pooled_grads.numpy()

for i in range(pooled_grads.shape[-1]):
    conv_outputs[:, :, i] *= pooled_grads[i]

heatmap = np.mean(conv_outputs, axis=-1)
heatmap = np.maximum(heatmap, 0)
heatmap /= np.max(heatmap)

plt.matshow(heatmap)
plt.show()

img = cv2.imread(img_path)
heathmap = cv2.resize(heatmap,(img.shape[1],img.shape[0]))
heatmap = np.uint8(255*heathmap)
heatmap = cv2.applyColorMap(heatmap,cv2.COLORMAP_JET)
sup = heatmap *0.4+img
cv2.imwrite('C:\\Users\\salva\\OneDrive\\Desktop\\machine-learning\\elefantiafricani.jpg',sup)