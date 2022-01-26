import tensorflow as tf
import os
import random

def prep_image(img_path, dimension):
    img = tf.keras.utils.load_img(img_path, 
        target_size=(dimension, dimension))
    return tf.keras.utils.img_to_array(img)

model_path = 'checkpoints/EfficientNetB3_1642522858'
model = tf.keras.models.load_model(model_path)

tensors = []
pred_dir = 'data/cats_v_dogs/validation/cats/'
img_paths = os.listdir(pred_dir)
img_paths = random.sample(img_paths, 10)
for path in img_paths:
    path = os.path.join(pred_dir, path)
    tensors.append(prep_image(path, 256))
tensors = tf.stack(tensors)
pred = model.predict(tensors)
print('\n\n')
print('Prediction:')
print(pred)

