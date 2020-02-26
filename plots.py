import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

accuracies = np.genfromtxt('activation/accuracies.csv',delimiter=',')
val_accuracies = np.genfromtxt('activation/val_accuracies.csv',delimiter=',')
losses = np.genfromtxt('activation/losses.csv',delimiter=',')
val_losses = np.genfromtxt('activation/val_losses.csv' ,delimiter =',')
epochs_range = 15
optimizers = ['only rescale','with rotation','with width shift','with height shift','with horizontal flip','with zoom']
fig = plt.figure(figsize=(8, 8))
plt.style.use('ggplot')

plt.subplot(2, 2, 1)
for idx in range (0,len(optimizers)):
	temp = (accuracies[idx+1])
	temp = temp[1:]
	plt.plot(temp,label='Training accuracy with ' + str(optimizers[idx]))
	# plt.errorbar([1,2,3,4,5,6,7,8,9,10,11,12,13,14,15],temp, yerr=np.std(temp))
plt.title('Training Accuracy')

plt.subplot(2, 2, 2)

for idx in range (0,len(optimizers)-1):
	temp = (losses[idx+1])
	temp = temp[1:]
	plt.ylim(top=1,bottom=0.5)
	plt.plot(temp,label='Training loss with ' + str(optimizers[idx]))
	
plt.title('Training loss')
plt.subplot(2, 2, 3)
for idx in range (0,len(optimizers)):
	temp = (val_accuracies[idx+1])
	temp = temp[1:]
	plt.plot(temp, label='Validation accuracy with ' + str(optimizers[idx]))
	

plt.title('Validation Accuracy')

plt.subplot(2, 2, 4)
for idx in range (0,len(optimizers)):
	temp = (val_losses[idx+1])
	temp = temp[1:]

	plt.plot(temp, label='Validation Loss with dropout '+ str(optimizers[idx]))
	plt.ylim(top=1,bottom=0.5)
plt.title('Validation Loss')
fig.legend(optimizers,loc='lower center',ncol=3)
plt.show()
