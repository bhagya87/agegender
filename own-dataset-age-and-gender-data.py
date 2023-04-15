#!/usr/bin/env python
# coding: utf-8

# In[ ]:


#Importing Modules
import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
from tqdm.notebook import tqdm
warnings.filterwarnings('ignore')
get_ipython().run_line_magic('matplotlib', 'inline')

import tensorflow as tf
#from keras.preprocessing.image import load_img
from keras.models import Sequential, Model
from keras.layers import Dense, Conv2D, Dropout, Flatten, MaxPooling2D, Input


# In[ ]:


#load the dataset
BASE_DIR = '../input/genderagedata/agegenderdata/'


# In[ ]:


#Labels - age,gender,ethnicity
image_paths = []
age_labels = []
gender_labels = []

for filename in tqdm(os.listdir(BASE_DIR)):
    image_path = os.path.join(BASE_DIR,filename)
    temp = filename.split('_')
    age = int(temp[0])
    gender = int(temp[1])
    image_paths.append(image_path)
    age_labels.append(age)
    gender_labels.append(gender)


# In[ ]:


df = pd.DataFrame()
df['image'], df['age'], df['gender'] = image_paths, age_labels, gender_labels
df.head()


# In[ ]:


#mapping of labels for gender
gender_dict = {0:'Male', 1:'Female'}


# In[ ]:


#Exploratory Data Analysis
from PIL import Image
img = Image.open(df['image'][0])
plt.imshow(img);


# In[ ]:


sns.distplot(df['age'])


# In[ ]:


import seaborn as sns
import matplotlib.pyplot as plt

sns.set(style='darkgrid')
plt.figure(figsize=(10, 5))
ax = sns.countplot(x='gender', data=df)
ax.set_xlabel('gender', fontsize=12)
plt.xticks(rotation=45)
plt.show()


# In[ ]:


#To display grid of images
plt.figure(figsize=(25, 25))
files = df.iloc[0:25]

for index, file, age, gender in files.itertuples():
    plt.subplot(5,5,index+1)
    #img = load_img(file)
    img = plt.imread(file)
    img = np.array(img)
    plt.imshow(img)
    plt.title(f"Age:{age} Gender:{gender_dict[gender]}")
    plt.axis('off')


# In[ ]:


import cv2
#Feature Extraction
def extract_features(images):
    features = []
    for image in tqdm(images):
       # img = plt.imread(image, grayscale=True)
        img = cv2.imread(image, cv2.IMREAD_GRAYSCALE)
        img = img.resize((128,128), Image.ANTIALIAS)
        img = np.array(img)
        features.append(img)
        
    features = np.array(features)
    features = features.reshape(len(features), 128, 128, 1)
    return features


# In[ ]:


import cv2

def extract_features(images):
    features = []
    for image in tqdm(images):
        img = cv2.imread(image, cv2.IMREAD_GRAYSCALE)
        img = cv2.resize(img, (128, 128), interpolation=cv2.INTER_AREA)
        img = np.array(img)
        features.append(img)
    features = np.array(features)
    features = features.reshape(len(features), 128, 128, 1)
    return features


# In[ ]:


X = extract_features(df['image'])


# In[ ]:


X.shape


# In[ ]:


X = X/255.0


# In[ ]:


y_gender = np.array(df['gender'])
y_age = np.array(df['age'])


# In[ ]:


input_shape = (128,128,1)


# In[ ]:


inputs = Input((input_shape))
# convolutional layers
#conv_0 = Conv2D(16, kernel_size=(3, 3), activation='relu') (inputs)
#maxp_0 = MaxPooling2D(pool_size=(2, 2)) (conv_0)

conv_1 = Conv2D(32, kernel_size=(3, 3), activation='relu') (inputs)
maxp_1 = MaxPooling2D(pool_size=(2, 2)) (conv_1)

conv_2 = Conv2D(64, kernel_size=(3, 3), activation='relu') (maxp_1)
maxp_2 = MaxPooling2D(pool_size=(2, 2)) (conv_2)

conv_3 = Conv2D(128, kernel_size=(3, 3), activation='relu') (maxp_2)
maxp_3 = MaxPooling2D(pool_size=(2, 2)) (conv_3)

conv_4 = Conv2D(256, kernel_size=(3, 3), activation='relu') (maxp_3)
maxp_4 = MaxPooling2D(pool_size=(2, 2)) (conv_4)

flatten = Flatten() (maxp_4)

# fully connected layers
dense_1 = Dense(256, activation='relu') (flatten)
dense_2 = Dense(256, activation='relu') (flatten)

dropout_1 = Dropout(0.3) (dense_1)
dropout_2 = Dropout(0.3) (dense_2)

output_1 = Dense(1, activation='sigmoid', name='gender_out') (dropout_1)
output_2 = Dense(1, activation='relu', name='age_out') (dropout_2)

model = Model(inputs=[inputs], outputs=[output_1, output_2])

model.compile(loss=['binary_crossentropy', 'mae'], optimizer='adam', metrics=['accuracy'])


# In[ ]:


model.summary()


# In[ ]:


#Plotting model
from tensorflow.keras.utils import plot_model
plot_model(model)


# In[ ]:


# train model
history = model.fit(x=X, y=[y_gender, y_age], batch_size=32, epochs=42, validation_split=0.2)


# In[ ]:


# plot results for gender
acc = history.history['gender_out_accuracy']
val_acc = history.history['val_gender_out_accuracy']
epochs = range(len(acc))

plt.plot(epochs, acc, 'b', label='Training Accuracy')
plt.plot(epochs, val_acc, 'r', label='Validation Accuracy')
plt.title('Accuracy Graph')
plt.legend()
plt.figure()

loss = history.history['gender_out_loss']
val_loss = history.history['val_gender_out_loss']

plt.plot(epochs, loss, 'b', label='Training Loss')
plt.plot(epochs, val_loss, 'r', label='Validation Loss')
plt.title('Loss Graph')
plt.legend()
plt.show()


# In[ ]:


# plot results for age
loss = history.history['age_out_loss']
val_loss = history.history['val_age_out_loss']
epochs = range(len(loss))

plt.plot(epochs, loss, 'b', label='Training Loss')
plt.plot(epochs, val_loss, 'r', label='Validation Loss')
plt.title('Loss Graph')
plt.legend()
plt.show()


# In[ ]:


#prediction with test data
image_index = 7
print("Original Gender:", gender_dict[y_gender[image_index]], "Original Age:", y_age[image_index])
# predict from model
pred = model.predict(X[image_index].reshape(1, 128, 128, 1))
pred_gender = gender_dict[round(pred[0][0][0])]
pred_age = round(pred[1][0][0])
print("Predicted Gender:", pred_gender, "Predicted Age:", pred_age)
plt.axis('off')
plt.imshow(X[image_index].reshape(128, 128), cmap='gray');

