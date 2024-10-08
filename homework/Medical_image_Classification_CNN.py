#%%
import numpy as np # forlinear algebra
import matplotlib.pyplot as plt #for plotting things
import os
from PIL import Image # for reading images
import pickle

# Keras Libraries <- CNN
import tensorflow as tf
from tensorflow.keras import datasets, layers, models, Model, Sequential
from tensorflow.keras.layers import Conv2D, MaxPool2D, AveragePooling2D, Flatten, Dense, Input, BatchNormalization, Concatenate, Dropout
from tensorflow.keras.preprocessing.image import ImageDataGenerator, load_img
#from sklearn.metrics import classification_report, confusion_matrix # <- define evaluation metrics

# data directory set
mainDIR = os.listdir('./archive/chest_xray')    # os.listdir 지정한 디렉토리 내의 모든 파일과 디렉토리 리스트를 리턴
print(mainDIR)
train_folder= './archive/chest_xray/train/'
val_folder = './archive/chest_xray/val/'
test_folder = './archive/chest_xray/test/'

# train data
os.listdir(train_folder)
train_n = train_folder+'NORMAL/'        # Normal lung file directory
train_p = train_folder+'PNEUMONIA/'     # Pneumonia lung file directory

print("what is train_n: ", train_n)
print("what is train_p: ", train_p)

#Normal pic
print(len(os.listdir(train_n)))
rand_norm= np.random.randint(0,len(os.listdir(train_n)))
norm_pic = os.listdir(train_n)[rand_norm]
print('normal picture title: ',norm_pic)
norm_pic_address = train_n+norm_pic
print('norm picture address: ', norm_pic_address)

#Pneumonia
rand_p = np.random.randint(0,len(os.listdir(train_p)))
sic_pic = os.listdir(train_p)[rand_norm]
sic_address = train_p+sic_pic
print('pneumonia picture title:', sic_pic)
print('pneumonia picture address: ', sic_address)

# Load the images
norm_load = Image.open(norm_pic_address)
sic_load = Image.open(sic_address)

#Let's plt these images
f = plt.figure(figsize= (10,6))
a1 = f.add_subplot(1,2,1)
img_plot = plt.imshow(norm_load)
a1.set_title('Normal')
a2 = f.add_subplot(1, 2, 2)
img_plot = plt.imshow(sic_load)
a2.set_title('Pneumonia')
plt.show()
 
# let's build the CNN model
# model = Sequential()
# Convolution
model_in = Input(shape = (64, 64, 3))
model = Conv2D(32, 3, activation = "relu")(model_in)
model = MaxPool2D(pool_size = (2,2))(model)
model = Conv2D(32, 3, activation = "relu")(model)
model = MaxPool2D(pool_size = (2,2))(model)
model = Flatten()(model)
# Fully Connected Layers
model = Dense(activation = 'relu', units = 128)(model)
model = Dense(activation = 'sigmoid', units = 1)(model)
# Compile the Neural network
model_fin = Model(inputs = model_in, outputs = model)
model_fin.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])

num_of_test_samples = 600
batch_size = 32
# Fitting the CNN to the images
# The function ImageDataGenerator augments your image by iterating through image as your CNN is getting ready to process that image
train_datagen = ImageDataGenerator(rescale = 1./255,
shear_range = 0.2,
zoom_range = 0.2,
horizontal_flip = True)
test_datagen = ImageDataGenerator(rescale = 1./255)
#Image normalization.
training_set = train_datagen.flow_from_directory('./archive/chest_xray/train',
target_size = (64, 64),
batch_size = 32,
class_mode = 'binary')

validation_generator = test_datagen.flow_from_directory('./archive/chest_xray/val/',
target_size=(64, 64),
batch_size=32,
class_mode='binary')
test_set = test_datagen.flow_from_directory('./archive/chest_xray/test',
target_size = (64, 64),
batch_size = 32,
class_mode = 'binary')
model_fin.summary()

cnn_model = model_fin.fit(training_set,
steps_per_epoch = 163,
epochs = 10,
validation_data = validation_generator,
validation_steps = 624)
test_accu = model_fin.evaluate(test_set,steps=624)

with open('history_Medical_CNN', 'wb') as file_pi:
    pickle.dump(cnn_model.history, file_pi)

model_fin.save('medical_ann.h5')
print('The testing accuracy is :',test_accu[1]*100, '%')
Y_pred = model_fin.predict(test_set, 100)
y_pred = np.argmax(Y_pred, axis=1)
max(y_pred)