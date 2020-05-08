#!/usr/bin/env python
# coding: utf-8

# In[3]:


### https://www.tensorflow.org/tutorials/load_data/pandas_dataframe

import numpy as np
import tensorflow as tf
import pandas as pd


# In[4]:



X = pd.read_csv("triData.csv")
X_t = pd.read_csv("triDataBig.csv")


# In[5]:


y = X.pop('tri')
y_t = X_t.pop('tri')


# In[6]:


dataset = tf.data.Dataset.from_tensor_slices((X.values, y.values))




for feat, targ in dataset.take(5):
  print ('Features: {}, Target: {}'.format(feat, targ))


# In[7]:


train_dataset = dataset.shuffle(len(X)).batch(1)


# In[10]:


####  http://faroit.com/keras-docs/1.2.2/optimizers/#sgd
####  http://faroit.com/keras-docs/1.2.2/activations/#available-activations


def get_compiled_model():
  model = tf.keras.Sequential([
    tf.keras.layers.Dense(3, activation='relu'),  ### this is custom input layer
      
    tf.keras.layers.Dense(3, activation='tanh'),  ### this is custom
      
    ##tf.keras.layers.Dense(2, activation='hard_sigmoid'),  ### this is custom
    tf.keras.layers.Dense(1,activation='hard_sigmoid')
  ])

  sgd = tf.keras.optimizers.SGD(lr=0.1, momentum=0.0, decay=0.0025, nesterov=False)
  model.compile(optimizer=sgd,
                loss='mean_squared_error',
                metrics=['accuracy'])
  return model


# In[16]:




model = get_compiled_model()

model.load_weights('tri_side_w', by_name=False)

model.fit(train_dataset, epochs=3)

model.save_weights('tri_side_w')


# In[17]:


val_loss, val_acc = model.evaluate(X_t, y_t)
print(val_loss)
print(val_acc)


# In[18]:


X['tri'] = y
X.head()


# In[19]:


pred = model.predict_classes(X_t)

k=0
for i in range(0,len(y_t),1):
    if(pred[i] ==y_t[i]):
       ## print(X_t.loc[i])
       ## print(y_t[i])
        k+=1

print(k)        
        


# In[ ]:




