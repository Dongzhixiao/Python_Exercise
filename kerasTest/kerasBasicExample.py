# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

#A Basic Example                   一个基本的例子
'''
Keras is a powerful and easy-to-use deep learning library for Theano and TensorFlow 
that provides a high-level neural networks API to develop and evaluate deep learning models.
'''
import numpy as np
from keras.models import Sequential
from keras.layers import Dense
data = np.random.random((1000,100))
labels = np.random.randint(2,size=(1000,1))
model = Sequential()
model.add(Dense(32, activation='relu', input_dim=100))
model.add(Dense(1, activation='sigmoid'))

model.summary()

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

model.fit(data,labels,validation_split=0.1)

