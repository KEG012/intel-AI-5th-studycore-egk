#%%
import matplotlib.pyplot as plt
import numpy as np
import pickle

CNN_batch_sig = pickle.load(open('./history_CNN_batch_sigmoid', 'rb'))
CNN_nobatch_sig = pickle.load(open('./history_CNN_nobatch_sigmoid', 'rb'))
CNN_relu = pickle.load(open('./history_CNN_relu', 'rb'))

val_acc_batch_sig = CNN_batch_sig["val_accuracy"]
val_loss_batch_sig = CNN_batch_sig["val_loss"]
val_acc_nobatch_sig = CNN_nobatch_sig["val_accuracy"]
val_loss_nobatch_sig = CNN_nobatch_sig["val_loss"]
val_acc_relu = CNN_relu["val_accuracy"]
val_loss_relu = CNN_relu["val_loss"]

plt.subplot(1,2,1)
plt.title('Validation Accuracy')
plt.plot(range(len(val_acc_batch_sig)), val_acc_batch_sig, label = 'val_acc_b')
plt.plot(range(len(val_acc_nobatch_sig)), val_acc_nobatch_sig, label = 'val_acc_nb')
plt.plot(range(len(val_acc_relu)), val_acc_relu, label = 'val_acc_r')
plt.legend()
plt.subplot(1,2,2)
plt.title('Validation Loss')
plt.plot(range(len(val_loss_batch_sig)), val_loss_batch_sig, label = 'val_loss_b')
plt.plot(range(len(val_loss_nobatch_sig)), val_loss_nobatch_sig, label = 'val_loss_nb')
plt.plot(range(len(val_loss_relu)), val_loss_relu, label = 'val_loss_r')
plt.legend()
plt.savefig("Summary.png")
plt.show()