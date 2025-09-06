#!/usr/bin/env python
# coding: utf-8

# # Importing the Dependencies

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sklearn.datasets
from sklearn.model_selection import train_test_split


# #Data Collection & Processing
# 

# In[2]:


#loading the data from sklearn
breast_cancer_dataset = sklearn.datasets.load_breast_cancer()


# In[3]:


print(breast_cancer_dataset)


# In[4]:


#loading the data to a data frame
data_frame = pd.DataFrame(breast_cancer_dataset.data , columns = breast_cancer_dataset.feature_names)


# In[5]:


#print the first 5 rows of the dataframe
data_frame.head()


# In[6]:


#adding the 'target' column to the data frame
data_frame['label'] = breast_cancer_dataset.target


# In[7]:


#print last 5 rows of the dataframe
data_frame.tail()


# In[8]:


# number of rows and columns in the dataset
data_frame.shape


# In[9]:


#getting some information about the data
data_frame.info()


# In[10]:


#checking for missing values
data_frame.isnull().sum()


# In[11]:


#statistical measure about the data
data_frame.describe()


# In[12]:


#checking the distribution of Target Varibale
data_frame['label'].value_counts()


# # 1 --> Benign
# # 0 --> Malignant

# In[13]:


data_frame.groupby('label').mean()


# #Separating the features and target
# 

# In[15]:


X = data_frame.drop(columns = 'label' , axis=1)
Y = data_frame['label']


# In[16]:


print(X)


# In[17]:


print(Y)


# #Splitting the data into training data & Testing data

# In[18]:


X_train, X_test, Y_train , Y_test = train_test_split(X,Y,test_size=0.2,random_state=2)


# In[20]:


print(X.shape,X_train.shape,X_test.shape)


# #Standardization the data

# In[32]:


from sklearn.preprocessing import StandardScaler



# In[35]:


scaler = StandardScaler()


# In[50]:


X_train_std = scaler.fit_transform(X_train)
X_test_std = scaler.transform(X_test)


# In[51]:


print(X_train_std)


# # Buiding the Neural Networks

# In[52]:


# importing tensorflow and keras
import tensorflow as tf
tf.random.set_seed(3)
from tensorflow import keras


# In[55]:


#Setting up the layers of Neural Network

model = keras.Sequential([
    keras.layers.Flatten(input_shape=(30,)),
    keras.layers.Dense(20, activation = 'relu'),
    keras.layers.Dense(2 , activation = 'sigmoid')
])
    


# In[56]:


# compiling the Neural Networks

model.compile(optimizer = 'adam',
              loss = 'sparse_categorical_crossentropy',
              metrics = ['accuracy']
             )


# In[57]:


#training the Neural Network

history = model.fit(X_train_std,Y_train,validation_split=0.1,epochs =10)


# #Visualizing accuracy and loss

# In[58]:
# Medical Data Visualization Enhancements

#Line chart
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])

plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['training data','validation data'],loc = 'lower right')


# In[59]:


plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])

plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['training data','validation data'],loc = 'upper right')


# #Accuracy of the model on test data

# In[60]:


loss , accuracy = model.evaluate(X_test_std, Y_test)
print(accuracy)


# In[62]:


print(X_test_std.shape)
print(X_test_std[0])


# In[64]:


Y_pred = model.predict(X_test_std)
print(Y_pred[0])


# In[65]:


print(X_test_std)


# In[66]:


print(Y_pred)


# #model.predict() gives me prediction probability of each class for the data point

# In[71]:


# argmax function

my_list = [0.25,0.56]

index_of_max_value = np.argmax(my_list)
print(my_list)
print(index_of_max_value)


# In[68]:


# converting the prediction probability to calss labels

Y_pred_labels = [np.argmax(i) for i in Y_pred]
print(Y_pred_labels)


# #Building the predictive System

# In[76]:


input_data = (20.57	,17.77	,132.9,	1326	,0.08474	,0.07864	,0.0869	,0.07017	,0.1812	,0.05667	,0.5435	,0.7339	,3.398	,74.08	,0.005225	,0.01308	,0.0186	,0.0134	,0.01389	,0.003532	,24.99	,23.41	,158.8	,1956	,0.1238	,0.1866	,0.2416	,0.186	,0.275	,0.08902)

#change the input_data to a numpy aray
input_data_as_numpy_array = np.asarray(input_data)

#reshape the numpy array as we are predicting for one data point
input_data_reshaped = input_data_as_numpy_array.reshape(1,-1)

#standardizing the input data
input_data_std =scaler.transform(input_data_reshaped)

prediction = model.predict(input_data_std)
print(prediction)

prediction_label = [np.argmax(prediction)]
print(prediction_label)

if (prediction_label[0] ==0):
    print('The tumor is Malignant')
else:
    print('The tumor is Benign')


# In[ ]:




