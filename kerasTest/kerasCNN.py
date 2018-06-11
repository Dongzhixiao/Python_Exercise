# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import numpy as np
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout

#Data                   1. 数据
'''
Your data needs to be stored as NumPy arrays or as a list of NumPy arrays. 
Ideally, you split the data in training and test sets, for which you can also 
resort to the train_test_split module of sklearn.cross_validation.
'''
##Keras Data Sets      1.1 Keras数据集合
from keras.datasets import boston_housing, mnist, cifar10, imdb
(x_train,y_train),(x_test,y_test) = mnist.load_data()
(x_train2,y_train2),(x_test2,y_test2) = boston_housing.load_data()
(x_train3,y_train3),(x_test3,y_test3) = cifar10.load_data()
(x_train4,y_train4),(x_test4,y_test4) = imdb.load_data(num_words=20000)
num_classes = 10

#Preprocessing            2. 预处理
##Sequence Padding        2.1 序列填充
from keras.preprocessing import sequence
x_train4 = sequence.pad_sequences(x_train4,maxlen=80)
x_test4 = sequence.pad_sequences(x_test4,maxlen=80)
##One-Hot Encoding        2.2 独热编码
from keras.utils import to_categorical
Y_train = to_categorical(y_train, num_classes)
Y_test = to_categorical(y_test, num_classes)
Y_train3 = to_categorical(y_train3, num_classes)
Y_test3 = to_categorical(y_test3, num_classes)

#Standardization/Normalization        3. 标准化
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler().fit(x_train2)
standardized_X = scaler.transform(x_train2)
standardized_X_test = scaler.transform(x_test2)

#Model Architecture                   4. 模型结构
##Sequential Model                    4.1 序列模型
model2 = Sequential()
##Convolutional Neural Network (CNN)  4.2 卷积神经网络
from keras.layers import Activation, Conv2D, MaxPooling2D, Flatten
model2.add(Conv2D(32, (3,3), padding='same', input_shape=(28,28,1)))
model2.add(Activation('relu'))
model2.add(Conv2D(32, (3,3)))
model2.add(Activation('relu'))
model2.add(MaxPooling2D(pool_size=(2,2)))
model2.add(Dropout(0.25))
model2.add(Conv2D(64, (3,3), padding='same'))
model2.add(Activation('relu'))
model2.add(Conv2D(64, (3, 3)))
model2.add(Activation('relu'))
model2.add(MaxPooling2D(pool_size=(2,2)))
model2.add(Dropout(0.25))
model2.add(Flatten())
model2.add(Dense(512))
model2.add(Activation('relu'))
model2.add(Dropout(0.5))
model2.add(Dense(num_classes))
model2.add(Activation('softmax'))

#Compile Model and Model Fine-Tuning    5. 编译模型
##Optimization Parameters               5.1 优化参数
from keras.optimizers import RMSprop
opt = RMSprop(lr=0.0001, decay=1e-6)
model2.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy'])

##Early Stopping                       5.2 早停
from keras.callbacks import EarlyStopping
early_stopping_monitor = EarlyStopping(patience=2)
x_train = x_train.reshape(x_train.shape[0], 28, 28, 1)   #转换成图像的四维
x_test = x_test.reshape(x_test.shape[0], 28, 28, 1)
model2.fit(x_train, y_train, batch_size=32, epochs=15, validation_data=(x_test, y_test), callbacks=[early_stopping_monitor])
