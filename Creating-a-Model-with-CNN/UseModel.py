import tensorflow as tf
from tensorflow.keras.preprocessing import image
import numpy as np

# Modeli yükleme
model = tf.keras.models.load_model('tiger_eagle_classifier_final.h5')

# Görüntüyü ön işleme
def preprocess_image(img_path):
    img = image.load_img(img_path, target_size=(224, 224))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array /= 255.0
    return img_array

# Görüntüleri sınıflandırma
def classify_image(img_path):
    img = preprocess_image(img_path)
    prediction = model.predict(img)
    if prediction < 0.5:
        print(f"{img_path} {prediction} kaplan resmi.")
    else:
        print(f"{img_path} {prediction} kartal resmi")

# Örnek görüntülerin sınıflandırılması
image_paths = [
    'PATH',
    'PATH',
    'PATH',
]

for img_path in image_paths:
    classify_image(img_path)
