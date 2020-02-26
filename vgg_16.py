from __future__ import absolute_import, division, print_function, unicode_literals
import tensorflow as tf

from tensorflow.keras.models import Sequential 
from tensorflow.keras import Model
from tensorflow.keras.layers import Dense, Conv2D, Flatten, Dropout, MaxPooling2D
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.applications import VGG16
import os
import numpy as np
import matplotlib.pyplot as plt

def create_VGG_16():
	vgg16 = VGG16(weights='imagenet', include_top=True)

	#Add a layer where input is the output of the  second last layer 
	x = Dense(1, activation='softmax', name='predictions')(vgg16.layers[-2].output)

	#Then create the corresponding model 
	my_model = Model(inputs=vgg16.input, outputs=x)
	sgd = SGD(lr=0.1, decay=1e-6, momentum=0.9, nesterov=True)
	my_model.compile(optimizer=sgd, loss='binary_crossentropy')
		
	return my_model

def test_model_vgg_16(total_train,total_val):
	accuracies = []
	val_accuracies = []
	losses = []
	val_losses =[]

	image_gen_train = ImageDataGenerator(rescale=1.0/255.0,
		 			rotation_range=45,
                    width_shift_range=.15,
                    height_shift_range=.15,
                    horizontal_flip=True,
                    zoom_range=0.25)
                                       

	train_data_gen = image_gen_train.flow_from_directory(batch_size=16,
                                                     directory=train_dir,
                                                     shuffle=True,
                                                     target_size=(224, 224),
                                                     class_mode='binary')

	image_gen_val = image_gen_train = ImageDataGenerator(rescale=1.0/255.0)

	val_data_gen = image_gen_val.flow_from_directory(batch_size=16,
                                                 directory=validation_dir,
                                                 target_size=(224, 224),
                                                 class_mode='binary')
	model_new = create_VGG_16()
	

	checkpoint = ModelCheckpoint("vgg16_1.h5", monitor='val_acc', verbose=1, save_best_only=True, save_weights_only=False, mode='auto', period=1)
	early = EarlyStopping(monitor='val_acc', min_delta=0, patience=20, verbose=1, mode='auto')
	
	history = model_new.fit_generator(
    	train_data_gen,
    	steps_per_epoch= total_train //16,
    	epochs=100,
    	validation_data=val_data_gen,
    	validation_steps=total_val //16,
    	callbacks=[checkpoint,early])
	acc = history.history['acc']
	val_acc = history.history['val_acc']

	loss = history.history['loss']
	val_loss = history.history['val_loss']

	epochs_range = 100

	plt.figure(figsize=(8, 8))
	plt.subplot(1, 2, 1)
	plt.plot(acc, label='Training Accuracy')
	plt.plot(val_acc, label='Validation Accuracy')
	plt.legend(loc='lower right')
	plt.title('Training and Validation Accuracy')

	plt.subplot(1, 2, 2)
	plt.plot(loss, label='Training Loss')
	plt.plot(val_loss, label='Validation Loss')
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

test_model_vgg_16(total_train,total_val)
