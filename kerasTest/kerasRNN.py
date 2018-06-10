# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

#A Basic Example
'''
Keras is a powerful and easy-to-use deep learning library for Theano and TensorFlow 
that provides a high-level neural networks API to develop and evaluate deep learning models.
'''
import numpy as np
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout

#Data
'''
Your data needs to be stored as NumPy arrays or as a list of NumPy arrays. 
Ideally, you split the data in training and test sets, for which you can also 
resort to the train_test_split module of sklearn.cross_validation.
'''
##Keras Data Sets
from keras.datasets import boston_housing, mnist, cifar10, imdb
(x_train,y_train),(x_test,y_test) = mnist.load_data()
(x_train2,y_train2),(x_test2,y_test2) = boston_housing.load_data()
(x_train3,y_train3),(x_test3,y_test3) = cifar10.load_data()
(x_train4,y_train4),(x_test4,y_test4) = imdb.load_data(num_words=20000)
num_classes = 10

#Preprocessing
##Sequence Padding
from keras.preprocessing import sequence
x_train4 = sequence.pad_sequences(x_train4,maxlen=80)
x_test4 = sequence.pad_sequences(x_test4,maxlen=80)
##One-Hot Encoding
from keras.utils import to_categorical
Y_train = to_categorical(y_train, num_classes)
Y_test = to_categorical(y_test, num_classes)
Y_train3 = to_categorical(y_train3, num_classes)
Y_test3 = to_categorical(y_test3, num_classes)

#Standardization/Normalization
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler().fit(x_train2)
standardized_X = scaler.transform(x_train2)
standardized_X_test = scaler.transform(x_test2)

#Model Architecture
##Sequential Model
model3 = Sequential()
##Recurrent Neural Network (RNN)
from keras.layers import Embedding,LSTM
model3.add(Embedding(20000,128))
model3.add(LSTM(128,dropout=0.2,recurrent_dropout=0.2))
model3.add(Dense(1,activation='sigmoid'))

#Compile Model
##Recurrent Neural Network (RNN)
model3.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

#Model Training
model3.fit(x_train4, y_train4, batch_size=32, epochs=15, verbose=1, validation_data=(x_test4, y_test4))

#Evaluate Your Model's Performance
score = model3.evaluate(x_test, y_test, batch_size=32)

#Prediction
model3.predict(x_test4, batch_size=32)
model3.predict_classes(x_test4,batch_size=32)

#Save/Reload Models
from keras.models import load_model
model3.save('model_file.h5')
my_model = load_model('my_model.h5')

#Model Fine-Tuning
##Early Stopping
from keras.callbacks import EarlyStopping
early_stopping_monitor = EarlyStopping(patience=2)
model3.fit(x_train4, y_train4, batch_size=32, epochs=15, validation_data=(x_test4, y_test4), callbacks=[early_stopping_monitor])
