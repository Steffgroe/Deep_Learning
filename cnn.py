# Code taken from https://www.tensorflow.org/tutorials/images/classification

from __future__ import absolute_import, division, print_function, unicode_literals
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Flatten, Dropout, MaxPooling2D
from tensorflow.keras.preprocessing.image import ImageDataGenerator

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from tensorflow.keras.datasets import cifar10

batch_size = 32
num_classes = 10
epochs = 100
data_augmentation = True
num_predictions = 20
save_dir = os.path.join(os.getcwd(), 'saved_models')
model_name = 'keras_cifar10_trained_model.h5'

# The data, split between train and test sets:
(x_train, y_train), (x_test, y_test) = cifar10.load_data()

def generate_model(dropout,HIDDEN_UNITS,OPTIMIZER,x_train):
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
    Dense(HIDDEN_UNITS, activation='relu'),
    Dense(10, activation='sigmoid')
	])

	model_new.compile(optimizer=OPTIMIZER,
              loss='categorical_crossentropy',
              metrics=['accuracy'])

	model_new.summary()
	return model_new

x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train /= 255
x_test /= 255
y_train = tf.keras.utils.to_categorical(y_train, num_classes)
y_test = tf.keras.utils.to_categorical(y_test, num_classes)
print("Loaded data")
if not data_augmentation:
    print('Not using data augmentation.')
    model.fit(x_train, y_train,
              batch_size=batch_size,
              epochs=epochs,
              validation_data=(x_test, y_test),
              shuffle=True)
else:
    print('Using real-time data augmentation.')
    # This will do preprocessing and realtime data augmentation:
    datagen = ImageDataGenerator(
        featurewise_center=False,  # set input mean to 0 over the dataset
        samplewise_center=False,  # set each sample mean to 0
        featurewise_std_normalization=False,  # divide inputs by std of the dataset
        samplewise_std_normalization=False,  # divide each input by its std
        zca_whitening=False,  # apply ZCA whitening
        zca_epsilon=1e-06,  # epsilon for ZCA whitening
        rotation_range=0,  # randomly rotate images in the range (degrees, 0 to 180)
        # randomly shift images horizontally (fraction of total width)
        width_shift_range=0.1,
        # randomly shift images vertically (fraction of total height)
        height_shift_range=0.1,
        shear_range=0.,  # set range for random shear
        zoom_range=0.,  # set range for random zoom
        channel_shift_range=0.,  # set range for random channel shifts
        # set mode for filling points outside the input boundaries
        fill_mode='nearest',
        cval=0.,  # value used for fill_mode = "constant"
        horizontal_flip=True,  # randomly flip images
        vertical_flip=False,  # randomly flip images
        # set rescaling factor (applied before any other transformation)
        rescale=None,
        # set function that will be applied on each input
        preprocessing_function=None,
        # image data format, either "channels_first" or "channels_last"
        data_format=None,
        # fraction of images reserved for validation (strictly between 0 and 1)
        validation_split=0.0)

    # Compute quantities required for feature-wise normalization
    # (std, mean, and principal components if ZCA whitening is applied).
print("test")
datagen.fit(x_train)
optimizers = [ 'adam','sgd','rmsprop','adagrad','adadelta']

accuracies = []
val_accuracies = []
losses = []
val_losses =[]

for idx in range (0,5):
	model = generate_model(0.2,512,optimizers[idx],x_train)

	history = model.fit_generator(datagen.flow(x_train, y_train,
                                     batch_size=batch_size),
                        epochs=epochs,
                        validation_data=(x_test, y_test))

	accuracies.append(history.history['acc'])
	val_accuracies.append(history.history['val_acc'])

	losses.append(history.history['loss'])
	val_losses.append(history.history['val_loss'])


epochs_range = range(epochs)

accuracies_df = pd.DataFrame([array for array in accuracies] )
accuracies_df.to_csv("accuracies.csv")

val_accuracies_df = pd.DataFrame([array for array in val_accuracies] )
val_accuracies_df.to_csv("val_accuracies.csv")

losses_df = pd.DataFrame([array for array in losses] )
losses_df.to_csv("losses.csv")

val_losses_df = pd.DataFrame([array for array in val_losses] ) 
val_losses_df.to_csv("val_losses.csv")

# plt.figure(figsize=(8, 8))
# plt.subplot(1, 2, 1)
# for idx in range (0,5):
# 	plt.plot(epochs_range, accuracies[idx], label='Training Accuracy with dropout '+str(idx/10))
# 	plt.plot(epochs_range, val_accuracies, label='Validation Accuracy with dropout '+str(idx/10))

# plt.legend(loc='lower right')
# plt.title('Training and Validation Accuracy')

# plt.subplot(1, 2, 2)
# for idx in range (0,5):
# 	plt.plot(epochs_range, lossses[idx], label='Training Loss with dropout '+str(idx/10))
# 	plt.plot(epochs_range, val_losses[idx], label='Validation Loss with dropout '+str(idx/10))
# plt.legend(loc='upper right')
# plt.title('Training and Validation Loss')
# plt.show()
