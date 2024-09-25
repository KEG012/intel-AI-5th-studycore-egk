#%%
import matplotlib.pyplot as plt
import numpy as np
import pickle

history_CNN_relu = pickle.load(open('./history_Medical_CNN', 'rb'))

tra_acc_CNN_relu = history_CNN_relu["accuracy"]
tra_loss_CNN_relu = history_CNN_relu["loss"]
val_acc_CNN_relu = history_CNN_relu["val_accuracy"]
val_loss_CNN_relu = history_CNN_relu["val_loss"]

plt.subplot(1,2,1)
plt.title("Model Accuracy")
plt.plot(range(len(tra_acc_CNN_relu)), tra_acc_CNN_relu, label = 'tra_accuracy')
plt.plot(range(len(val_acc_CNN_relu)), val_acc_CNN_relu, label = 'val_accuracy')
plt.legend()
plt.subplot(1,2,2)
plt.title("Model Loss")
plt.plot(range(len(tra_loss_CNN_relu)), tra_loss_CNN_relu, label = 'tra_loss')
plt.plot(range(len(val_loss_CNN_relu)), val_loss_CNN_relu, label = 'val_loss')
plt.legend()
plt.show()