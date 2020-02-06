# Code taken from https://www.tensorflow.org/tutorials/images/classification

from __future__ import absolute_import, division, print_function, unicode_literals
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Flatten, Dropout, MaxPooling2D
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
import os
import numpy as np
from vgg_16 import create_VGG_16
import matplotlib.pyplot as plt

def generate_model(dropout,HIDDEN_UNITS,OPTIMIZER):
	model_new = Sequential([
    Conv2D(16, 3, padding='same', activation='relu', 
           input_shape=(IMG_HEIGHT, IMG_WIDTH ,3)),
    MaxPooling2D(),
    Dropout(dropout),
    Conv2D(32, 3, padding='same', activation='relu'),
    MaxPooling2D(),
    Conv2D(64, 3, padding='same', activation='relu'),
    MaxPooling2D(),
    Dropout(dropout),
    Flatten(),
    Dense(HIDDEN_UNITS, activation='relu'),
    Dense(1, activation='sigmoid')
	])

	model_new.compile(optimizer=OPTIMIZER,
              loss='binary_crossentropy',
              metrics=['accuracy'])

	model_new.summary()
	return model_new

def test_model_augmentation_cnn(image_gen_train,train_data_gen,image_gen_val,val_data_gen):
	accuracies = []
	val_accuracies = []
	losses = []
	val_losses =[]

	for idx in range (1,5):
		model_new = generate_model(idx/10,HIDDEN_UNITS,OPTIMIZER)

		history = model_new.fit_generator(
	    	train_data_gen,
	    	steps_per_epoch=total_train // batch_size,
	    	epochs=epochs,
	    	validation_data=val_data_gen,
	    	validation_steps=total_val // batch_size
		)

		accuracies.append(history.history['acc'])
		val_accuracies.append(history.history['val_acc'])

		losses.append(history.history['loss'])
		val_losses.append(history.history['val_loss'])


	epochs_range = range(epochs)

	for idx in range (0,4):

		plt.figure(figsize=(8, 8))
		plt.subplot(1, 2, 1)


		plt.plot(epochs_range, accuracies[idx], label='Training Accuracy with dropout '+str((idx+1)/10))
		plt.plot(epochs_range, val_accuracies[idx], label='Validation Accuracy with dropout '+str((idx+1)/10))

		plt.legend(loc='lower right')
		plt.title('Training and Validation Accuracy')

		plt.subplot(1, 2, 2)
		plt.plot(epochs_range, losses[idx], label='Training Loss with dropout '+str((idx+1)/10))
		plt.plot(epochs_range, val_losses[idx], label='Validation Loss with dropout '+str((idx+1)/10))
		plt.legend(loc='upper right')
		plt.title('Training and Validation Loss')
		plt.show()


	
#Configurations for the model

batch_size = 128
epochs = 15
IMG_HEIGHT = 150
IMG_WIDTH = 150
HIDDEN_UNITS =512
OPTIMIZER = 'adam'

# Load the data 
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

# Load and rescale the images

# train_image_generator = ImageDataGenerator(rescale=1./255) # Generator for our training data
# validation_image_generator = ImageDataGenerator(rescale=1./255) # Generator for our validation data

# train_data_gen = train_image_generator.flow_from_directory(batch_size=batch_size,
#                                                            directory=train_dir,
#                                                            shuffle=True,
#                                                            target_size=(IMG_HEIGHT, IMG_WIDTH),
#                                                            class_mode='binary')

# val_data_gen = validation_image_generator.flow_from_directory(batch_size=batch_size,
#                                                               directory=validation_dir,
#                                                               target_size=(IMG_HEIGHT, IMG_WIDTH),
#                                                               class_mode='binary')
# test_model_augmentation_cnn(train_image_generator,train_data_gen,validation_image_generator,val_data_gen)

image_gen_train = ImageDataGenerator(
                    rescale=1./255,
                    rotation_range=45,
                    width_shift_range=.15,
                    height_shift_range=.15,
                    horizontal_flip=True,
                    zoom_range=0.25
                    )

train_data_gen = image_gen_train.flow_from_directory(batch_size=batch_size,
                                                     directory=train_dir,
                                                     shuffle=True,
                                                     target_size=(IMG_HEIGHT, IMG_WIDTH),
                                                     class_mode='binary')

image_gen_val = ImageDataGenerator(rescale=1./255)

val_data_gen = image_gen_val.flow_from_directory(batch_size=batch_size,
                                                 directory=validation_dir,
                                                 target_size=(IMG_HEIGHT, IMG_WIDTH),

                                                 class_mode='binary')
# test_model_augmentation_cnn(image_gen_train,train_data_gen,image_gen_val,val_data_gen)