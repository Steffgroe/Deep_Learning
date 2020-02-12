#Code taken from https://www.tensorflow.org/tutorials/images/classification

from __future__ import absolute_import, division, print_function, unicode_literals
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Flatten, Dropout, MaxPooling2D, LeakyReLU
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from tensorflow.keras.preprocessing.image import ImageDataGenerator

import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from tensorflow.keras.datasets import cifar10

def generate_model(dropout,HIDDEN_UNITS,activation,optimizer,x_train):
	if activation == "leaky":
		model_new = Sequential([
    	Conv2D(32, (3, 3), padding='same',
                 input_shape=x_train.shape[1:]),
	    MaxPooling2D(),
	    Dropout(dropout),
	    Conv2D(32, 3, padding='same', activation='relu'),
	    MaxPooling2D(),
	    Conv2D(64, 3, padding='same', activation='relu'),
	    MaxPooling2D(),
	    Dropout(dropout),
	    Flatten(),
	   	Dense(HIDDEN_UNITS),
	   	LeakyReLU(alpha=0.3),
	   	Dense(10, activation='sigmoid')
		])
	else:
		model_new = Sequential([
	    Conv2D(32, (3, 3), padding='same',
                 input_shape=x_train.shape[1:]),
	    MaxPooling2D(),
	    Dropout(dropout),
	    Conv2D(32, 3, padding='same', activation='relu'),
	    MaxPooling2D(),
	    Conv2D(64, 3, padding='same', activation='relu'),
	    MaxPooling2D(),
	    Dropout(dropout),
	    Flatten(),
	   	Dense(HIDDEN_UNITS,activation =activation),
	   	Dense(10, activation='sigmoid')	])

	model_new.compile(optimizer=optimizer,
              loss='categorical_crossentropy',
              metrics=['accuracy'])

	model_new.summary()
	return model_new

def get_augmentation(n):
	if n == 0:
		print('0 reached')
		image_gen_train = ImageDataGenerator(
                    rescale=1./255,
                    )
	elif n == 1:
		image_gen_train = ImageDataGenerator(
                    rescale=1./255,
                    rotation_range=45,
                    )
	elif n == 2:
		image_gen_train = ImageDataGenerator(
                    rescale=1./255,
                    rotation_range=45,
                     width_shift_range=.15,
                    )
	elif n ==3:
		image_gen_train = ImageDataGenerator(
                    rescale=1./255,
                    rotation_range=45,
                    width_shift_range=.15,
                    height_shift_range=.15,
                    )
	elif n == 4:
		image_gen_train = ImageDataGenerator(
                    rescale=1./255,
                    rotation_range=45,
                    width_shift_range=.15,
                    height_shift_range=.15,
                    horizontal_flip=True,
                    
                    )
	else:
		image_gen_train = ImageDataGenerator(
                    rescale=1./255,
                    rotation_range=45,
                    width_shift_range=.15,
                    height_shift_range=.15,
                    horizontal_flip=True,
                    zoom_range=0.25
                    )

	return image_gen_train

batch_size = 32
num_classes = 10
epochs = 100
data_augmentation = True
num_predictions = 20
save_dir = os.path.join(os.getcwd(), 'saved_models')
model_name = 'keras_cifar10_trained_model.h5'

# The data, split between train and test sets:
(x_train, y_train), (x_test, y_test) = cifar10.load_data()

x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train /= 255
x_test /= 255
y_train = tf.keras.utils.to_categorical(y_train, num_classes)
y_test = tf.keras.utils.to_categorical(y_test, num_classes)


# accuracies = []
# val_accuracies = []
# losses = []
# val_losses =[]

# for idx in range (0,6):
# 	model = generate_model(0.2,512,'relu','adam',x_train)
# 	datagen = get_augmentation(idx)
# 	datagen.fit(x_train)
	
# 	checkpoint = ModelCheckpoint("cnn.h5", monitor='val_acc', verbose=1, save_best_only=True, save_weights_only=False, mode='auto', period=1)
# 	early = EarlyStopping(monitor='val_acc', min_delta=0, patience=20, verbose=1, mode='auto')
	

# 	history = model.fit_generator(datagen.flow(x_train, y_train,
#                                      batch_size=batch_size),
#                       				 epochs=epochs,
#                         			validation_data=(x_test, y_test),
#                         			callbacks=[checkpoint,early])

# 	accuracies.append(history.history['acc'])
# 	val_accuracies.append(history.history['val_acc'])

# 	losses.append(history.history['loss'])
# 	val_losses.append(history.history['val_loss'])


# epochs_range = range(epochs)

# accuracies_df = pd.DataFrame([array for array in accuracies] )
# accuracies_df.to_csv("augmentation/accuracies.csv")

# val_accuracies_df = pd.DataFrame([array for array in val_accuracies] )
# val_accuracies_df.to_csv("augmentation/val_accuracies.csv")

# losses_df = pd.DataFrame([array for array in losses] )
# losses_df.to_csv("augmentation/losses.csv")

# val_losses_df = pd.DataFrame([array for array in val_losses] ) 
# val_losses_df.to_csv("augmentation/val_losses.csv")
# datagen.fit(x_train)
activation =['leaky','relu','tanh','sigmoid','softsign']
optimizers = [ 'adam','sgd','rmsprop','adagrad','adadelta']

accuracies = []
val_accuracies = []
losses = []
val_losses =[]
datagen = get_augmentation(0)
datagen.fit(x_train)
for idx in range (0,len(activation)):
	model = generate_model(0.2,512,activation[idx],'adam',x_train)

	checkpoint = ModelCheckpoint("cnn.h5", monitor='val_acc', verbose=1, save_best_only=True, save_weights_only=False, mode='auto', period=1)
	early = EarlyStopping(monitor='val_acc', min_delta=0, patience=20, verbose=1, mode='auto')
	

	history = model.fit_generator(datagen.flow(x_train, y_train,
                                     batch_size=batch_size),
                      				 epochs=epochs,
                        			validation_data=(x_test, y_test),
                        			callbacks=[checkpoint,early])

	accuracies.append(history.history['acc'])
	val_accuracies.append(history.history['val_acc'])

	losses.append(history.history['loss'])
	val_losses.append(history.history['val_loss'])


epochs_range = range(epochs)

accuracies_df = pd.DataFrame([array for array in accuracies] )
accuracies_df.to_csv("activation/accuracies.csv")

val_accuracies_df = pd.DataFrame([array for array in val_accuracies] )
val_accuracies_df.to_csv("activation/val_accuracies.csv")

losses_df = pd.DataFrame([array for array in losses] )
losses_df.to_csv("activation/losses.csv")

val_losses_df = pd.DataFrame([array for array in val_losses] ) 
val_losses_df.to_csv("activation/val_losses.csv")