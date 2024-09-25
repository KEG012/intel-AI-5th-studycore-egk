#%%
import numpy as np # forlinear algebra
import matplotlib.pyplot as plt #for plotting things
import os
import PIL
from PIL import Image # for reading images
import pickle
import cv2

# Keras Libraries <- CNN
import tensorflow as tf
import tensorflow_datasets as tfds
from tensorflow.keras import datasets, layers, models, Model, Sequential
from tensorflow.keras.layers import Conv2D, MaxPool2D, AveragePooling2D, Flatten, Dense, Input, BatchNormalization, Concatenate, Dropout, GlobalAveragePooling2D
from tensorflow.keras.preprocessing.image import ImageDataGenerator, load_img
#from sklearn.metrics import classification_report, confusion_matrix # <- define evaluation metrics


(train_ds, val_ds, test_ds), metadata = tfds.load(
    'tf_flowers',
    split=['train[:80%]', 'train[80%:90%]', 'train[90%:]'],
    batch_size = 32,
    with_info=True,
    as_supervised=True,
)

num_classes = metadata.features['label'].num_classes
get_label_name = metadata.features['label'].int2str
train_ds.cache()
train_ds.shuffle(buffer_size=1000)
val_ds.cache()
val_ds.shuffle(buffer_size=1000)
image, label = next(iter(train_ds))

print(label)
print(np.array(image).shape)

plt.figure(figsize=(5,5))
for i in range(10):
    plt.subplot(3,4,i+1)
    plt.imshow(image[i])
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    plt.title(get_label_name(label[i]))
plt.show()

input_shape = (80, 80, 3)

base_model = tf.keras.applications.MobileNetV3Small(
    input_shape=input_shape,
    include_top=False,
    # 전이 학습을 위해 FC 층을 제거.
    weights='imagenet')

preprocess_input = tf.keras.applications.mobilenet_v3.preprocess_input


inputs = tf.keras.Input(shape=input_shape)
x = preprocess_input(inputs)
# Mobilenet CNN model
x = layers.RandomFlip("horizontal_and_vertical")(x)
x = layers.RandomRotation(0.2)(x)
x = base_model(x, training = False)
x = GlobalAveragePooling2D()(x)
# 직접 FC층을 작성 해야 함.
x = tf.keras.layers.Dropout(0.2)(x)
output = tf.keras.layers.Dense(num_classes, activation='softmax')(x)
model = tf.keras.Model(inputs, output)
model.summary()

model.compile(optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy'])

history = model.fit(train_ds,
    validation_data=val_ds,
    batch_size = 32,
    # steps_per_epoch=20,
    epochs=10)

with open('Flower_data_CNN', 'wb') as file_pi:
    pickle.dump(history.history, file_pi)

model.save('Flower_data_CNN.h5')
