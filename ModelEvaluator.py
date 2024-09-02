import tensorflow as tf
import os
import cv2
import random
import numpy as np

from matplotlib import pyplot as plt
from tensorflow.keras.models import load_model

model_dir = "models"
abspath_model_dir = os.path.abspath(model_dir)
model = load_model(os.path.join(abspath_model_dir, "CategoricalClassifier.keras"))

data_dir = "data"
abspath_happy_data_dir = os.path.join(os.path.abspath(data_dir), "Happy")
happy_image_files = [file for file in os.listdir(abspath_happy_data_dir) if os.path.isfile(os.path.join(abspath_happy_data_dir, file))]
abspath_sad_data_dir = os.path.join(os.path.abspath(data_dir), "Sad")
sad_image_files = [file for file in os.listdir(abspath_sad_data_dir) if os.path.isfile(os.path.join(abspath_sad_data_dir, file))]
abspath_angry_data_dir = os.path.join(os.path.abspath(data_dir), "Angry")
angry_image_files = [file for file in os.listdir(abspath_angry_data_dir) if os.path.isfile(os.path.join(abspath_angry_data_dir, file))]
image_files = happy_image_files + sad_image_files + angry_image_files

image_chosen = random.choice(image_files)

if image_chosen in happy_image_files:
    abspath_image_chosen = os.path.join(abspath_happy_data_dir, image_chosen)
    state = "Happy"
elif image_chosen in sad_image_files:
    abspath_image_chosen = os.path.join(abspath_sad_data_dir, image_chosen)
    state = "Sad"
else:
    abspath_image_chosen = os.path.join(abspath_sad_data_dir, image_chosen)
    state = "Angry"

image = cv2.imread(abspath_image_chosen)

image = tf.image.resize(image, (256,256))

predicted_state_value = model.predict(np.expand_dims(image/255,0))
predicted_state_value = np.argmax(predicted_state_value)

if predicted_state_value == 2:
    predicted_state = "Sad"
elif predicted_state_value == 1:
    predicted_state = "Happy"
else:
    predicted_state = "Angry"

print("Actual State: ", state)
print("Predicted State: ", predicted_state)

try:
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    plt.imshow(image.numpy().astype(int))
    plt.show()
except:
    plt.imshow(image.numpy().astype(int))
    plt.show()