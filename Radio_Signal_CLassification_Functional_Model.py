# -*- coding: utf-8 -*-
"""
Created on Tue Sep  7 20:15:41 2021

@author: ahmed
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

import random
import os
import datetime
from tqdm import tqdm

import cv2
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score

import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv2D, Dense, Flatten, MaxPooling2D, BatchNormalization, Dropout
from tensorflow.keras.callbacks import EarlyStopping, CSVLogger, ModelCheckpoint, TensorBoard
from tensorflow.keras.preprocessing import image_dataset_from_directory

train_path = "E:/Software/professional practice projects/In progress/primary_small/train"
val_path = "E:/Software/professional practice projects/In progress/primary_small/valid"
test_path = "E:/Software/professional practice projects/In progress/primary_small/test"

classes_list = os.listdir("E:/Software/professional practice projects/In progress/primary_small/train")

batch_size= 128
train_ds = image_dataset_from_directory(
    train_path,
    batch_size = batch_size)

plt.figure(figsize=(50, 40))
for images, labels in train_ds.take(1):
    for i in range(1, 36):
        plt.subplot(5, 7, i)
        plt.imshow(images[i].numpy().astype("uint8"))
        plt.title(classes_list[labels[i]])
        plt.xlabel(images[i].numpy().astype("uint8").shape)
        plt.ylabel(images[i].numpy().astype("uint8").shape)
        # plt.axis('off')

plt.show()
plt.close()


val_ds = image_dataset_from_directory(
    val_path, 
    batch_size=batch_size)


plt.figure(figsize=(50, 40))
for images, labels in train_ds.take(1):
    for i in range(1, 36):
        plt.subplot(5, 7, i)
        plt.imshow(images[i].numpy().astype("uint8"))
        plt.title(classes_list[labels[i]])
        plt.xlabel(images[i].numpy().astype("uint8").shape)
        plt.ylabel(images[i].numpy().astype("uint8").shape)
        # plt.axis('off')

plt.show()
plt.close()


for img_batch, label_batch in train_ds.take(1):
    print(img_batch.shape)
    print(label_batch.shape)


for img_batch, label_batch in val_ds.take(1):
    print(img_batch.shape)
    print(label_batch.shape)


img_width = 256
img_height = 256

normalizing_layer = tf.keras.layers.experimental.preprocessing.Rescaling(1./255)
resizing_layer = tf.keras.layers.experimental.preprocessing.Resizing(img_width, img_height)

train_ds_norm = train_ds.map(lambda x, y: (resizing_layer(x), y))
train_ds_norm = train_ds.map(lambda x, y: (normalizing_layer(x), y))

val_ds_norm = val_ds.map(lambda x, y: (resizing_layer(x), y))
val_ds_norm = val_ds.map(lambda x, y: (normalizing_layer(x), y))

img_batch, label_batch = next(iter(train_ds_norm))
img = img_batch[0]
label = label_batch[0]

print(np.min(img), ' --> ', np.max(img))
print('shape: ', img.shape)
print('label: ', label)




AutoTune = tf.data.experimental.AUTOTUNE

train_ds = train_ds.cache().prefetch(buffer_size=AutoTune)
val_ds = train_ds.cache().prefetch(buffer_size=AutoTune)



num_classes = len(classes_list)

inputs = Input(shape=(img_width, img_height, 3))

x = Conv2D(128, kernel_size=(3, 3), activation='relu', padding='same')(inputs)
x = MaxPooling2D(pool_size=(5, 5))(x)
x = Conv2D(128, kernel_size=(3, 3), activation='relu', padding='same')(x)
x = MaxPooling2D(pool_size=(5, 5))(x)
x = Dropout(0.15)(x)

x = Conv2D(128, kernel_size=(3, 3), activation='relu', padding='same')(x)
x = Conv2D(64, kernel_size=(3, 3), activation='relu', padding='same')(x)
x = MaxPooling2D(pool_size=(2, 2))(x)
x = Dropout(0.15)(x)

x = Conv2D(64, kernel_size=(3, 3), activation='relu', padding='same')(x)
x = MaxPooling2D(pool_size=(2, 2))(x)
x = Dropout(0.15)(x)

x = Flatten()(x)

x = Dense(1024, activation='relu')(x)
x = Dropout(0.4)(x)

classifier = Dense(num_classes, activation='softmax')(x)

model = Model(inputs=inputs, outputs=classifier)


model.compile(optimizer=tf.keras.optimizers.Adam(), loss='sparse_categorical_crossentropy', metrics=['accuracy'])


epochs = 30
# log_dir = "logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
model.fit(train_ds,
          batch_size=batch_size,
          epochs=epochs,
          verbose=1,
          validation_data=val_ds,
          callbacks=[EarlyStopping(monitor='val_loss', patience=5, verbose=1, restore_best_weights=True),
                     CSVLogger("train.csv")])






