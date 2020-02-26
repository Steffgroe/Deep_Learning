#!/usr/bin/env python
# coding: utf-8

# In[1]:


import tensorflow as tf
import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Flatten, Dropout, MaxPooling2D, LeakyReLU
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.datasets import cifar10

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# In[2]:


# The data, split between train and test sets:
(x_train, y_train), (x_test, y_test) = cifar10.load_data()


# In[3]:


batch_size = 32
num_classes = 10
epochs = 100
data_augmentation = True
num_predictions = 20
save_dir = os.path.join(os.getcwd(), 'saved_models')
model_name = 'keras_cifar10_trained_model.h5'


x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train /= 255
x_test /= 255
y_train = tf.keras.utils.to_categorical(y_train, num_classes)
y_test = tf.keras.utils.to_categorical(y_test, num_classes)


# In[5]:


#Code taken from https://www.tensorflow.org/tutorials/images/classification
import tensorflow
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Flatten, Dropout, MaxPooling2D, LeakyReLU, BatchNormalization
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from tensorflow.keras.preprocessing.image import ImageDataGenerator

import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from tensorflow.keras.datasets import cifar10

def generate_model(dropout,HIDDEN_UNITS,activation,optimizer,x_train):
    if optimizer is 'sgd':
        optimizer = tensorflow.keras.optimizers.SGD(lr=0.1 ,decay=1e-6, momentum=0.9, nesterov=True)
    if optimizer is 'adadelta':
        optimizer = tensorflow.keras.optimizers.Adadelta(learning_rate=0.1, rho=0.95)
	
    if activation == "leaky":
        model_new = Sequential([
    	Conv2D(32, (3, 3), padding='same',
                 input_shape=x_train.shape[1:]),
	    BatchNormalization(),
        MaxPooling2D(),
	    Dropout(dropout),
	    Conv2D(32, 3, padding='same', activation='relu'),
	    BatchNormalization(),
        MaxPooling2D(),
	    Conv2D(64, 3, padding='same', activation='relu'),
	    BatchNormalization(),
        MaxPooling2D(),
	    Dropout(dropout),
	    Flatten(),
	   	Dense(HIDDEN_UNITS),
	   	LeakyReLU(alpha=0.3),
        BatchNormalization(),
	   	Dense(10, activation='softmax')
		])
    else:
        model_new = Sequential([
	    Conv2D(32, (3, 3), padding='same',
                 input_shape=x_train.shape[1:]),
        BatchNormalization(),
	    MaxPooling2D(),
	    Dropout(dropout),
	    Conv2D(32, 3, padding='same', activation='relu'),
	    BatchNormalization(),
        MaxPooling2D(),
	    Conv2D(64, 3, padding='same', activation='relu'),
	    BatchNormalization(),
        MaxPooling2D(),
	    Dropout(dropout),
	    Flatten(),
	   	Dense(HIDDEN_UNITS,activation =activation),
        BatchNormalization(),
	   	Dense(10, activation='softmax')	])

    print(optimizer)
    model_new.compile(optimizer=optimizer,
              loss='categorical_crossentropy',
              metrics=['accuracy'])

    model_new.summary()
    return model_new


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

# activation =['leaky','relu','tanh','sigmoid','softsign']
activation = ['relu']
optimizers = [ 'adam','sgd','rmsprop','adagrad','adadelta']
test_accuracies = []
accuracies = []
val_accuracies = []
losses = []
val_losses =[]
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
    validation_split=0.33)


datagen.fit(x_train)

test_accuracies = []
accuracies = []
val_accuracies = []
losses = []
val_losses =[]

for idx in range (0,len(optimizers)):
	name = 'batch2/cnn ' + str(optimizers[idx] +'.h5')
	model = generate_model(0.2,512,'relu',optimizers[idx],x_train)

	checkpoint = ModelCheckpoint(name, monitor='val_acc', verbose=1, save_best_only=True, save_weights_only=False, mode='auto', period=1)
	early = EarlyStopping(monitor='val_acc', min_delta=0, patience=10, verbose=1, mode='auto')
	

	history = model.fit_generator(datagen.flow(x_train, y_train,
                                     batch_size=batch_size,subset ='training'),
                      				 epochs=epochs,
                      				 validation_data= datagen.flow(x_train, y_train,
                                     batch_size=batch_size,subset ='validation'),
                        			callbacks=[checkpoint,early])
	model.load_weights(name)
	score = model.evaluate(x_test, y_test, verbose=0)

	accuracies.append(history.history['acc'])
	val_accuracies.append(history.history['val_acc'])

	losses.append(history.history['loss'])
	val_losses.append(history.history['val_loss'])
	test_accuracies.append(score)

epochs_range = range(epochs)

accuracies_df = pd.DataFrame([array for array in accuracies] )
accuracies_df.to_csv("batch2/accuracies.csv")

val_accuracies_df = pd.DataFrame([array for array in val_accuracies] )
val_accuracies_df.to_csv("batch2/val_accuracies.csv")

test_accuracies_df = pd.DataFrame(test_accuracies )
test_accuracies_df.to_csv("batch2/test_accuracies.csv")

losses_df = pd.DataFrame([array for array in losses] )
losses_df.to_csv("batch2/losses.csv")

val_losses_df = pd.DataFrame([array for array in val_losses] ) 
val_losses_df.to_csv("batch2/val_losses.csv")


# In[35]:



accuracies_df = pd.DataFrame([array for array in accuracies] )
accuracies_df.to_csv("optimizers/accuracies.csv")

val_accuracies_df = pd.DataFrame([array for array in val_accuracies] )
val_accuracies_df.to_csv("optimizers/val_accuracies.csv")

test_accuracies_df = pd.DataFrame(test_accuracies )
test_accuracies_df.to_csv("optimizers/test_accuracies.csv")

losses_df = pd.DataFrame([array for array in losses] )
losses_df.to_csv("optimizers/losses.csv")

val_losses_df = pd.DataFrame([array for array in val_losses] ) 
val_losses_df.to_csv("optimizers/val_losses.csv")


# In[48]:


accuracies = np.genfromtxt('accuracies.csv',delimiter=',')
val_accuracies = np.genfromtxt('val_accuracies.csv',delimiter=',')
losses = np.genfromtxt('losses.csv',delimiter=',')
val_losses = np.genfromtxt("val_losses.csv",delimiter =',')
epochs_range = 15
optimizers = [ 'adam','sgd','rmsprop','adamax','adadelta']
fig = plt.figure(figsize=(8, 8))
plt.style.use('ggplot')

plt.subplot(2, 2, 1)
for idx in range (0,5):
	temp = (accuracies[idx+1])
	temp = temp[1:]
	plt.plot([1,2,3,4,5,6,7,8,9,10,11,12,13,14,15],temp,label='Training accuracy with ' + str(optimizers[idx]))
	# plt.errorbar([1,2,3,4,5,6,7,8,9,10,11,12,13,14,15],temp, yerr=np.std(temp))
plt.title('Training Accuracy')

plt.subplot(2, 2, 2)

for idx in range (0,5):
	temp = (losses[idx+1])
	temp = temp[1:]
	plt.ylim(top=1,bottom=0.5)
	plt.plot([1,2,3,4,5,6,7,8,9,10,11,12,13,14,15],temp,label='Training loss with ' + str(optimizers[idx]))
	
plt.title('Training loss')
plt.subplot(2, 2, 3)
for idx in range (0,5):
	temp = (val_accuracies[idx+1])
	temp = temp[1:]
	plt.plot([1,2,3,4,5,6,7,8,9,10,11,12,13,14,15],temp, label='Validation accuracy with ' + str(optimizers[idx]))
	

plt.title('Validation Accuracy')

plt.subplot(2, 2, 4)
for idx in range (0,5):
	temp = (val_losses[idx+1])
	temp = temp[1:]

	plt.plot([1,2,3,4,5,6,7,8,9,10,11,12,13,14,15],temp, label='Validation Loss with dropout '+ str(optimizers[idx]))

plt.title('Validation Loss')
fig.legend(optimizers,loc='lower center',ncol=5)
plt.show()


# In[61]:


batch_size = 128
epochs = 15
IMG_HEIGHT = 150
IMG_WIDTH = 150
HIDDEN_UNITS =512
data_augmentation = False

def generate_model(dropout,HIDDEN_UNITS):
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

	model_new.compile(optimizer="Adam",
              loss='binary_crossentropy',
              metrics=['accuracy'])

	model_new.summary()
	return model_new

#Configurations for the model
accuracies = []
val_accuracies = []
losses = []
val_losses =[]

for idx in np.arange(0.0, 1.0, 0.1):
	model = generate_model(idx,512)

	history = model.fit_generator(
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

accuracies_df = pd.DataFrame([array for array in accuracies] )
accuracies_df.to_csv("accuracies.csv")

val_accuracies_df = pd.DataFrame([array for array in val_accuracies] )
val_accuracies_df.to_csv("val_accuracies.csv")

losses_df = pd.DataFrame([array for array in losses] )
losses_df.to_csv("losses.csv")

val_losses_df = pd.DataFrame([array for array in val_losses] ) 
val_losses_df.to_csv("val_losses.csv")


# In[7]:


accuracies = np.genfromtxt('accuracies.csv',delimiter=',')
val_accuracies = np.genfromtxt('val_accuracies.csv',delimiter=',')
losses = np.genfromtxt('losses.csv',delimiter=',')
val_losses = np.genfromtxt("val_losses.csv",delimiter =',')
epochs_range = 15
fig = plt.figure(figsize=(8, 8))
plt.style.use('ggplot')



plt.subplot(2, 2, 1)
for idx in np.arange(1, 10, 1):
	temp = (accuracies[idx])[1:]
	plt.plot([1,2,3,4,5,6,7,8,9,10,11,12,13,14,15],temp,label='Training accuracy with ' + str(idx))
	# plt.errorbar([1,2,3,4,5,6,7,8,9,10,11,12,13,14,15],temp, yerr=np.std(temp))
plt.title('Training Accuracy')

plt.subplot(2, 2, 2)

for idx in np.arange(1, 10, 1):
	temp = (losses[idx])[1:]
	plt.ylim(top=1,bottom=0.5)
	plt.plot([1,2,3,4,5,6,7,8,9,10,11,12,13,14,15],temp,label='Training loss with ' + str(idx))
	
plt.title('Training loss')
plt.subplot(2, 2, 3)
for idx in np.arange(1, 10, 1):
	temp = (val_accuracies[idx])[1:]
	plt.plot([1,2,3,4,5,6,7,8,9,10,11,12,13,14,15],temp, label='Validation accuracy with ' + str(idx))
	

plt.title('Validation Accuracy')
plt.subplot(2, 2, 4)
for idx in np.arange(1, 10, 1):
	temp = (val_losses[idx])[1:]
	plt.plot([1,2,3,4,5,6,7,8,9,10,11,12,13,14,15],temp, label='Validation Loss with dropout '+ str(idx))

plt.title('Validation Loss')
test = np.linspace(0, 1, num = 10, endpoint = True)
test = np.around(test,1)
fig.legend(test,loc='lower center',ncol=5)
plt.show()


# In[85]:



batch_size = 128
epochs = 15
IMG_HEIGHT = 150
IMG_WIDTH = 150
HIDDEN_UNITS =512
data_augmentation = False

def generate_model(REG):
    model_new = Sequential([
        Conv2D(16, 3, padding='same', activation='relu', 
           input_shape=(IMG_HEIGHT, IMG_WIDTH ,3)),
        MaxPooling2D(),
        Dropout(0.2),
        Conv2D(32, 3, padding='same', activation='relu'),
        MaxPooling2D(),
        Conv2D(64, 3, padding='same', activation='relu'),
        MaxPooling2D(),
        Dropout(0.2),
        Flatten(),
        Dense(512, activation='relu', kernel_regularizer=keras.regularizers.l2(REG)),
        Dense(1, activation='sigmoid')
])

    model_new.compile(optimizer="Adam",
              loss='binary_crossentropy',
              metrics=['accuracy'])

    model_new.summary()
    return model_new

#Configurations for the model
accuracies = []
val_accuracies = []
losses = []
val_losses =[]

for REG in [.1, .01, .001, .0001, .00001]:
    model = generate_model(REG)

    history = model.fit_generator(
    train_data_gen,
    steps_per_epoch=total_train // batch_size,
    epochs=epochs,
    validation_data=val_data_gen,
    validation_steps=total_val // batch_size)


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


# In[91]:


accuracies = np.genfromtxt('accuracies.csv',delimiter=',')
val_accuracies = np.genfromtxt('val_accuracies.csv',delimiter=',')
losses = np.genfromtxt('losses.csv',delimiter=',')
val_losses = np.genfromtxt("val_losses.csv",delimiter =',')
epochs_range = 15
fig = plt.figure(figsize=(8, 8))
plt.style.use('ggplot')



plt.subplot(2, 2, 1)
for idx in [1,2,3,4]:
	temp = (accuracies[idx])[1:]
	plt.plot([1,2,3,4,5,6,7,8,9,10,11,12,13,14,15],temp,label='Training accuracy with ' + str(idx))
	# plt.errorbar([1,2,3,4,5,6,7,8,9,10,11,12,13,14,15],temp, yerr=np.std(temp))
plt.title('Training Accuracy')

plt.subplot(2, 2, 2)

for idx in [1,2,3,4]:
	temp = (losses[idx])[1:]
	plt.ylim(top=1,bottom=0.5)
	plt.plot([1,2,3,4,5,6,7,8,9,10,11,12,13,14,15],temp,label='Training loss with ' + str(idx))
	
plt.title('Training loss')
plt.subplot(2, 2, 3)
for idx in [1,2,3,4]:
	temp = (val_accuracies[idx])[1:]
	plt.plot([1,2,3,4,5,6,7,8,9,10,11,12,13,14,15],temp, label='Validation accuracy with ' + str(idx))
	

plt.title('Validation Accuracy')
plt.subplot(2, 2, 4)
for idx in [1,2,3,4]:
	temp = (val_losses[idx])[1:]
	plt.plot([1,2,3,4,5,6,7,8,9,10,11,12,13,14,15],temp, label='Validation Loss with dropout '+ str(idx))

plt.title('Validation Loss')
test = [.1, .01, .001, .0001, .00001]
fig.legend(test,loc='lower center',ncol=5)
plt.show()


# In[100]:



def generate_model():
    model_new = Sequential([
        Conv2D(16, 3, padding='same', activation='relu', 
           input_shape=(IMG_HEIGHT, IMG_WIDTH ,3)),
        MaxPooling2D(),
        Dropout(0.2),
        Conv2D(32, 3, padding='same', activation='relu'),
        MaxPooling2D(),
        Conv2D(64, 3, padding='same', activation='relu'),
        MaxPooling2D(),
        Dropout(0.2),
        Flatten(),
        Dense(512, activation='relu', kernel_regularizer=keras.regularizers.l2(.001)),
        Dense(1, activation='sigmoid')
])

    model_new.compile(optimizer="Adam",
              loss='binary_crossentropy',
              metrics=['accuracy'])

    model_new.summary()
    return model_new

#Configurations for the model
accuracies = []
val_accuracies = []
losses = []
val_losses =[]

for batch_size in [64, 128, 256, 512]:
    epochs = 15
    IMG_HEIGHT = 150
    IMG_WIDTH = 150
    HIDDEN_UNITS =512
    data_augmentation = False

    model = generate_model()

    history = model.fit_generator(
    train_data_gen,
    steps_per_epoch=total_train // batch_size,
    epochs=epochs,
    validation_data=val_data_gen,
    validation_steps=total_val // batch_size)


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


# In[101]:



epochs_range = range(epochs)

accuracies_df = pd.DataFrame([array for array in accuracies] )
accuracies_df.to_csv("accuracies.csv")

val_accuracies_df = pd.DataFrame([array for array in val_accuracies] )
val_accuracies_df.to_csv("val_accuracies.csv")

losses_df = pd.DataFrame([array for array in losses] )
losses_df.to_csv("losses.csv")

val_losses_df = pd.DataFrame([array for array in val_losses] ) 
val_losses_df.to_csv("val_losses.csv")


# In[103]:



plt.subplot(2, 2, 1)
for idx in [1,2,3,4]:
	temp = (accuracies[idx])[1:]
	plt.plot([1,2,3,4,5,6,7,8,9,10,11,12,13,14,15],temp,label='Training accuracy with ' + str(idx))
	# plt.errorbar([1,2,3,4,5,6,7,8,9,10,11,12,13,14,15],temp, yerr=np.std(temp))
plt.title('Training Accuracy')

plt.subplot(2, 2, 2)

for idx in [1,2,3,4]:
	temp = (losses[idx])[1:]
	plt.ylim(top=1,bottom=0.5)
	plt.plot([1,2,3,4,5,6,7,8,9,10,11,12,13,14,15],temp,label='Training loss with ' + str(idx))
	
plt.title('Training loss')
plt.subplot(2, 2, 3)
for idx in [1,2,3,4]:
	temp = (val_accuracies[idx])[1:]
	plt.plot([1,2,3,4,5,6,7,8,9,10,11,12,13,14,15],temp, label='Validation accuracy with ' + str(idx))
	

plt.title('Validation Accuracy')
plt.subplot(2, 2, 4)
for idx in [1,2,3,4]:
	temp = (val_losses[idx])[1:]
	plt.plot([1,2,3,4,5,6,7,8,9,10,11,12,13,14,15],temp, label='Validation Loss with dropout '+ str(idx))

plt.title('Validation Loss')
test = [64, 128, 256, 512]
fig.legend(test,loc='lower center',ncol=5)
plt.show()


# In[60]:


os.getcwd()

activation_val_accuracies = np.genfromtxt('activation_val_accuracies.csv',delimiter=',')
augmentation_val_accuracies = np.genfromtxt('augmentation_val_accuracies.csv',delimiter=',')
batch_val_accuracies = np.genfromtxt('batch_val_accuracies.csv',delimiter=',')
optimizers_val_accuracies = np.genfromtxt('optimizers_val_accuracies.csv',delimiter=',')

fig = plt.figure(figsize=(8, 8))
plt.style.use('ggplot')

plt.subplot(2, 2, 1)
for idx in [1,2,3,4,5]:
	temp = (activation_val_accuracies[idx])[1:]
	plt.plot(range(0,len(temp)),temp,label='Training accuracy with ' + str(idx))
plt.title('Activations')

plt.subplot(2, 2, 2)

for idx in [1,2,3,4,5]:
	temp = (augmentation_val_accuracies[idx])[1:]
	plt.ylim(top=1,bottom=0.5)
	plt.plot(range(0,len(temp)),temp,label='Training loss with ' + str(idx))
	
plt.title('Augmentation')

plt.subplot(2, 2, 3)
for idx in [1,2,3,4,5]:
	temp = (batch_val_accuracies[idx])[1:]
	plt.plot(range(0,len(temp)),temp, label='Validation accuracy with ' + str(idx))
	
plt.title('Batch normalization')

plt.subplot(2, 2, 4)
for idx in [1,2,3,4,5]:
	temp = (optimizers_val_accuracies[idx])[1:]
	plt.plot(range(0,len(temp)),temp, label='Validation Loss with dropout '+ str(idx))
plt.title('Optimizers')

test = ["Leaky ReLU", "ReLU", "tanh", "sigmoid", "softsign"]
fig.legend(test,loc='lower center',ncol=5)
plt.show()

