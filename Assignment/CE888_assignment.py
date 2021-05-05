#!/usr/bin/env python
# coding: utf-8

# In[5]:


import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import datasets, layers, models
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Activation, Dense, Flatten, BatchNormalization, Conv2D, MaxPool2D
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.metrics import categorical_crossentropy
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report
from shutil import copy
import itertools
import os
import shutil
import random
import glob
import matplotlib.pyplot as plt
import warnings
import pickle
warnings.simplefilter(action='ignore', category=FutureWarning)
get_ipython().run_line_magic('matplotlib', 'inline')

from numpy.random import seed
seed(1)


# In[6]:


#edit home_path variable to the location of Fire-vs-NoFire dataset
home_path = "/home/austin/Downloads/Fire-vs-NoFire/"
os.chdir(home_path)


#checking if sample training and test datsets exist
if os.path.isdir('Training_sample/Fire') is False:
    os.makedirs('Training_sample/Fire')
    os.makedirs('Training_sample/No_Fire')
    os.makedirs('Test_sample/Fire')
    os.makedirs('Test_sample/No_Fire')

    #if traing and test sample dataset do not exist they are created
    for each in ['Fire', 'No_Fire']:    
        for i in random.sample(glob.glob('Training/'+each+'/*'), 1000):
            copy(i, 'Training_sample/'+each+'/')
        for i in random.sample(glob.glob('Test/'+each+'/*'), 500):
            copy(i, 'Test_sample/'+each+'/')        


# In[7]:


#setting paths to test and train sample datasets
train_path = home_path +'Training_sample/'
test_path = home_path +'Test_sample/'


# In[8]:


#initialising ImageDataGenerator with vgg16 preprocessing
image_generator = ImageDataGenerator(
    preprocessing_function=tf.keras.applications.vgg16.preprocess_input, validation_split=0.1) 

#creating training and validation batches
train_batches = image_generator.flow_from_directory(directory=train_path, target_size=(224,224), classes=['No_Fire', 'Fire'], batch_size=10, subset="training", class_mode='categorical')
valid_batches = image_generator.flow_from_directory(directory=train_path, target_size=(224,224), classes=['No_Fire', 'Fire'], batch_size=10, subset="validation", class_mode='categorical')

#creating test batches
test_batches = ImageDataGenerator(preprocessing_function= tf.keras.applications.vgg16.preprocess_input)     .flow_from_directory(directory=test_path, target_size=(224,224), classes=['No_Fire', 'Fire'], batch_size=10, shuffle=False)


# In[9]:


#reading a single batch of data
imgs, labels = next(train_batches)


# In[10]:


#plotting images
def plotImages(images_arr):
    fig, axes = plt.subplots(1, 10, figsize=(20,20))
    axes = axes.flatten()
    for img, ax in zip( images_arr, axes):
        ax.imshow(img)
        ax.axis('off')
    plt.tight_layout()
    plt.show()
plotImages(imgs)
print(labels)


# In[15]:


#defininf layer of CNN 
model = Sequential([
    Conv2D(filters=32, kernel_size=(3, 3), activation='relu', padding = 'same', input_shape=(224,224,3)),
    MaxPool2D(pool_size=(2, 2), strides=1),
    Conv2D(filters=64, kernel_size=(4, 4), activation='relu', padding = 'same'),
    MaxPool2D(pool_size=(2, 2), strides=1),
    Conv2D(filters=128, kernel_size=(3, 3), activation='relu', padding = 'same'),
    MaxPool2D(pool_size=(2, 2), strides=1),
    Flatten(),
    Dense(units=2, activation='sigmoid')
])


# In[16]:


#model summary
model.summary()


# In[17]:


#compiling model with learning rate and loss function
model.compile(optimizer=Adam(learning_rate=0.0001), loss='binary_crossentropy', metrics=['accuracy'])


# In[18]:


#fitting model to training data
model.fit(x=train_batches,
    steps_per_epoch=len(train_batches),
    validation_data=valid_batches,
    validation_steps=len(valid_batches),
    epochs=5,
#     verbose=2
)


# In[ ]:


#predicting test data
predictions = model.predict(x=test_batches, steps=len(test_batches))
cm = confusion_matrix(y_true=test_batches.classes, y_pred=np.argmax(predictions, axis=-1))


# In[ ]:


def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i, j],
            horizontalalignment="center",
            color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')


# In[ ]:


#classes of test data
test_batches.class_indices


# In[ ]:


cm_plot_labels = ['No_Fire','Fire']
#plotting confusion matrix
a = plot_confusion_matrix(cm=cm, classes=cm_plot_labels, title='Confusion Matrix')


# In[ ]:





# In[ ]:


#printing accuracy
print(classification_report(test_batches.classes, np.argmax(predictions, axis=-1), target_names=['No_Fire', 'Fire']))


# In[ ]:





# In[ ]:





# In[21]:


# vgg16 model for transfer learning
transfer_model = tf.keras.applications.ResNet50V2()


# In[23]:



#vgg16 model for transfer learning
transfer_model = tf.keras.applications.vgg16.VGG16()

#creating new sequential model
model_transfer = Sequential()

#adding layers from vgg16 to new model
for layer in transfer_model.layers[:-1]:
    model_transfer.add(layer)

#setting layers to non-trainable    
for layer in model_transfer.layers:
    layer.trainable = False  
    


# In[24]:


model_transfer.add(Dense(units=2, activation='softmax'))


# In[25]:


model_transfer.summary()


# In[ ]:


#compiling transfer learnig model
model_transfer.compile(optimizer=Adam(learning_rate=0.0001), loss='categorical_crossentropy', metrics=['accuracy'])


# In[ ]:


#training transfer learning mdoel
model_transfer.fit(x=train_batches,
    steps_per_epoch=len(train_batches),
    validation_data=valid_batches,
    validation_steps=len(valid_batches),
    epochs=5,
#     verbose=2
)


# In[ ]:


#prediction on test data
predictions_transfer = model_transfer.predict(x=test_batches, steps=len(test_batches), verbose=0)

#printing classifiction report
print(classification_report(test_batches.classes, np.argmax(predictions_transfer, axis=-1)))

