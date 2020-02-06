from __future__ import absolute_import, division, print_function, unicode_literals
import tensorflow as tf

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Flatten, Dropout, MaxPooling2D
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
import os
import numpy as np
import matplotlib.pyplot as plt

def create_VGG_16():
	model = Sequential()
	model.add(Conv2D(input_shape=(150,150,3),filters=64,kernel_size=(3,3),padding="same", activation="relu"))

	model.add(Conv2D(filters=64,kernel_size=(3,3),padding="same", activation="relu"))
	model.add(MaxPooling2D(pool_size=(2,2),strides=(2,2)))
	model.add(Conv2D(filters=128, kernel_size=(3,3), padding="same", activation="relu"))
	model.add(Conv2D(filters=128, kernel_size=(3,3), padding="same", activation="relu"))
	model.add(MaxPooling2D(pool_size=(2,2),strides=(2,2)))
	model.add(Conv2D(filters=256, kernel_size=(3,3), padding="same", activation="relu"))
	model.add(Conv2D(filters=256, kernel_size=(3,3), padding="same", activation="relu"))
	model.add(Conv2D(filters=256, kernel_size=(3,3), padding="same", activation="relu"))

	model.add(MaxPooling2D(pool_size=(2,2),strides=(2,2)))
	model.add(Conv2D(filters=512, kernel_size=(3,3), padding="same", activation="relu"))
	model.add(Conv2D(filters=512, kernel_size=(3,3), padding="same", activation="relu"))
	model.add(Conv2D(filters=512, kernel_size=(3,3), padding="same", activation="relu"))
	model.add(MaxPooling2D(pool_size=(2,2),strides=(2,2)))

	model.add(Conv2D(filters=512, kernel_size=(3,3), padding="same", activation="relu"))
	model.add(Conv2D(filters=512, kernel_size=(3,3), padding="same", activation="relu"))
	model.add(Conv2D(filters=512, kernel_size=(3,3), padding="same", activation="relu"))
	model.add(MaxPooling2D(pool_size=(2,2),strides=(2,2)))

	model.add(Flatten())
	model.add(Dense(units=4096,activation="relu"))
	model.add(Dense(units=4096,activation="relu"))
	model.add(Dense(units=1, activation="softmax"))
	model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy'])

	model.summary()


	return model

def test_model_vgg_16():
	accuracies = []
	val_accuracies = []
	losses = []
	val_losses =[]

	image_gen_train = ImageDataGenerator(
                    rescale=1./255,
                    rotation_range=45,
                    width_shift_range=.15,
                    height_shift_range=.15,
                    horizontal_flip=True,
                    zoom_range=0.25
                    )

	train_data_gen = image_gen_train.flow_from_directory(batch_size=16,
                                                     directory=train_dir,
                                                     shuffle=True,
                                                     target_size=(150, 150),
                                                     class_mode='binary')

	image_gen_val = ImageDataGenerator(rescale=1./255)

	val_data_gen = image_gen_val.flow_from_directory(batch_size=16,
                                                 directory=validation_dir,
                                                 target_size=(150, 150),

                                                 class_mode='binary')
	model_new = create_VGG_16()
	

	checkpoint = ModelCheckpoint("vgg16_1.h5", monitor='val_acc', verbose=1, save_best_only=True, save_weights_only=False, mode='auto', period=1)
	early = EarlyStopping(monitor='val_acc', min_delta=0, patience=20, verbose=1, mode='auto')
	
	history = model_new.fit_generator(
    	train_data_gen,
    	steps_per_epoch=100,
    	epochs=100,
    	validation_data=val_data_gen,
    	validation_steps=100
	)
	acc = history.history['acc']
	val_acc = history.history['val_acc']

	loss = history.history['loss']
	val_loss = history.history['val_loss']

	epochs_range = range(epochs)

	plt.figure(figsize=(8, 8))
	plt.subplot(1, 2, 1)
	plt.plot(epochs_range, acc, label='Training Accuracy')
	plt.plot(epochs_range, val_acc, label='Validation Accuracy')
	plt.legend(loc='lower right')
	plt.title('Training and Validation Accuracy')

	plt.subplot(1, 2, 2)
	plt.plot(epochs_range, loss, label='Training Loss')
	plt.plot(epochs_range, val_loss, label='Validation Loss')
	plt.legend(loc='upper right')
	plt.title('Training and Validation Loss VGG 16')
	plt.show()
_URL = 'https://storage.googleapis.com/mledu-datasets/cats_and_dogs_filtered.zip'
path_to_zip = tf.keras.utils.get_file('cats_and_dogs.zip', origin=_URL, extract=True)

PATH = os.path.join(os.path.dirname(path_to_zip), 'cats_and_dogs_filtered')

train_dir = os.path.join(PATH, 'train')
validation_dir = os.path.join(PATH, 'validation')

train_cats_dir = os.path.join(train_dir, 'cats')  # directory with our training cat pictures
train_dogs_dir = os.path.join(train_dir, 'dogs')  # directory with our training dog pictures
validation_cats_dir = os.path.join(validation_dir, 'cats')  # directory with our validation cat pictures
validation_dogs_dir = os.path.join(validation_dir, 'dogs')  # directory with our validation dog pictures

num_cats_tr = len(os.listdir(train_cats_dir))
num_dogs_tr = len(os.listdir(train_dogs_dir))

num_cats_val = len(os.listdir(validation_cats_dir))
num_dogs_val = len(os.listdir(validation_dogs_dir))

total_train = num_cats_tr + num_dogs_tr
total_val = num_cats_val + num_dogs_val

test_model_vgg_16()
