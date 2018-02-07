# -*- coding: utf-8 -*-
"""
Created on Tue Jan 16 16:46:18 2018

@author: shrey
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from math import sqrt
from sklearn.metrics import mean_squared_error


data=pd.read_excel('/home/shrey/Desktop/project/project_data/data_english/total_diary_4579.ko.en.xlsx',sep=',',header=2)

mig_data=data[['Become sensitive to light','Light / Sound Sensitive','Migraine presence','stress','Sleeping too much','Lack of sleep','Exercise','Not to Exercise',
'Physical fatigue','Weather / temperature changes','Excessive sunlight','noise','Excessive drinking','Irregular eating (fasting, etc.)','Excessive caffeine',
'Excessive smoking','Chocolate Cheesecake','Travel']]

mig_data=mig_data.reset_index(drop=True)

mig_data=mig_data[:4579]

mig_data.dropna(inplace=True)

mig_data['Migraine_forecast']=mig_data['Migraine presence'].shift(-1)

train_data=mig_data[:1000]
test_data=mig_data[1000:1050]

train_x_df=train_data[['Become sensitive to light','Light / Sound Sensitive','Migraine presence','stress','Sleeping too much','Lack of sleep','Exercise',
'Physical fatigue','Weather / temperature changes','Excessive sunlight','noise','Excessive drinking','Irregular eating (fasting, etc.)','Excessive caffeine',
'Excessive smoking','Chocolate Cheesecake','Travel']]

train_y_df=train_data['Migraine_forecast']

test_x_df=test_data[['Become sensitive to light','Light / Sound Sensitive','Migraine presence','stress','Sleeping too much','Lack of sleep','Exercise',
'Physical fatigue','Weather / temperature changes','Excessive sunlight','noise','Excessive drinking','Irregular eating (fasting, etc.)','Excessive caffeine',
'Excessive smoking','Chocolate Cheesecake','Travel']]

test_y_df=test_data['Migraine_forecast']

train_x=np.array(train_x_df)
train_y=np.array(train_y_df)

test_x=np.array(test_x_df)
test_y=np.array(test_y_df)

train_X=train_x.reshape((train_x.shape[0],1,train_x.shape[1]))
test_X = test_x.reshape((test_x.shape[0], 1, test_x.shape[1]))

model = Sequential()
model.add(LSTM(50, input_shape=(train_X.shape[1], train_X.shape[2])))
model.add(Dense(1))
model.compile(loss='mae', optimizer='adam')

history = model.fit(train_X, train_y, epochs=100, batch_size=50, validation_data=(test_X, test_y), verbose=2, shuffle=False)

#plt.plot(history.history['loss'], label='train')
#plt.plot(history.history['val_loss'], label='test')

yhat = model.predict(test_X)

rmse = sqrt(mean_squared_error(test_y, yhat))

fig = plt.figure(figsize=(5, 4))
ax = fig.add_subplot(1, 1, 1)
ax.set_xlabel(r'iteration')
ax.set_ylabel(r'error')  
ax.plot(history.history['loss'])     

fig1 = plt.figure(figsize=(5, 4))
ax1 = fig1.add_subplot(1, 1, 1)
ax1.set_xlabel(r'iteration')
ax1.set_ylabel(r'error')  
ax1.plot(history.history['val_loss'])        








