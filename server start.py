import sys
import tensorflow as tf
import numpy as np
from PIL import Image

# Load model once
model = tf.keras.models.load_model("model.h5")

def predict(image_path):
    img = Image.open(image_path).resize((224, 224))
    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    prediction = model.predict(img_array)
    return int(np.argmax(prediction))

if __name__ == "__main__":
    result = predict(sys.argv[1])
    print(result)
