import os

import numpy as np
import pandas as pd
from PIL import Image
from keras.models import load_model
from keras.utils import to_categorical
from sklearn.model_selection import train_test_split

from models.create_model import create_cnn_model
from plots.plot_accuracy import plot_accuracy

data = []
labels = []
classes = 43
cur_path = os.getcwd()

# Dataset Can be downloaded at https://www.kaggle.com/datasets/meowmeowmeowmeowmeow/gtsrb-german-traffic-sign?select=Train
# Retrieving the images and their labels
for i in range(classes):
    path = os.path.join(cur_path, 'train', str(i))
    images = os.listdir(path)

    for each_image in images:
        try:
            image = Image.open(path + '/' + each_image)
            image = image.resize((30, 30))
            image = np.array(image)
            # sim = Image.fromarray(image)
            data.append(image)
            labels.append(i)
        except Exception as e:
            print("Error loading image", e)

# Converting lists into numpy arrays
data = np.array(data)
labels = np.array(labels)

# Splitting training and testing dataset
X_train, X_test, y_train, y_test = train_test_split(data, labels, test_size=0.2, random_state=42)

# Converting the labels into one hot encoding
y_train = to_categorical(y_train, 43)
y_test = to_categorical(y_test, 43)

history = create_cnn_model(X_train, X_test, y_train, y_test)

plot_accuracy(history)

# testing accuracy on test dataset
test_data = pd.read_csv('Test.csv')

labels = test_data["ClassId"].values
imgs = test_data["Path"].values

data = []

for img in imgs:
    image = Image.open(img)
    image = image.resize((30, 30))
    data.append(np.array(image))

X_test = np.array(data)
loaded_model = load_model("traffic_signal_keras_model.h5")
y_predict = np.argmax(loaded_model.predict(X_test), axis=-1)
# Accuracy with the test data
from sklearn.metrics import accuracy_score

print(accuracy_score(labels, y_predict))
