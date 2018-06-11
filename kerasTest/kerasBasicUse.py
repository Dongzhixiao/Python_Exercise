# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

#Data                                                                   1. 数据
'''
Your data needs to be stored as NumPy arrays or as a list of NumPy arrays. 
Ideally, you split the data in training and test sets, for which you can also 
resort to the train_test_split module of sklearn.cross_validation.
'''
##Keras Data Sets                                                       1.1 Keras数据集
from keras.datasets import boston_housing, mnist, cifar10, imdb
(x_train,y_train),(x_test,y_test) = mnist.load_data()
(x_train2,y_train2),(x_test2,y_test2) = boston_housing.load_data()
(x_train3,y_train3),(x_test3,y_test3) = cifar10.load_data()
(x_train4,y_train4),(x_test4,y_test4) = imdb.load_data(num_words=20000)
num_classes = 10
##Other   the URL below can not found, so comment the code temporary   1.2 其他数据集
#from urllib.request import urlopen
#data = np.loadtxt(urlopen("http://archive.ics.uci.edu/ml/machine-learning-databases/pima-indians-diabetes/pima-indians-diabetes.data"),delimiter=",")
#X = data[:,0:8]
#y = data [:,8]

#Preprocessing          2. 预处理
##Sequence Padding      2.1 序列填充
from keras.preprocessing import sequence
x_train4 = sequence.pad_sequences(x_train4,maxlen=80)
x_test4 = sequence.pad_sequences(x_test4,maxlen=80)
##One-Hot Encoding      2.2 独热编码
from keras.utils import to_categorical
Y_train = to_categorical(y_train, num_classes)
Y_test = to_categorical(y_test, num_classes)
Y_train3 = to_categorical(y_train3, num_classes)
Y_test3 = to_categorical(y_test3, num_classes)
##Train And Test Sets   2.3 训练和测试集合
#from sklearn.model_selection import train_test_split
#X_train5, X_test5, y_train5, y_test5 = train_test_split(X, y, test_size=0.33, random_state=42)

#Standardization/Normalization    3. 标准化
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler().fit(x_train2)
standardized_X = scaler.transform(x_train2)
standardized_X_test = scaler.transform(x_test2)

#Model Architecture               4. 模型结构
##Sequential Model                4.1 序列模型
from keras.models import Sequential
model = Sequential()
##Multi-Layer Perceptron (MLP)    4.2 多层感知机
###Binary Classification          4.2.1 二分类
from keras.layers import Dense
#model.add(Dense(12, input_dim=8, kernel_initializer='uniform', activation='relu'))
#model.add(Dense(8, kernel_initializer='uniform', activation='relu'))
#model.add(Dense(1, kernel_initializer='uniform', activation='sigmoid'))
###Multi-Class Classification     4.2.2 多分类
from keras.layers import Dropout
#model.add(Dense(512,activation='relu',input_shape=(784,)))
#model.add(Dropout(0.2))
#model.add(Dense(512,activation='relu'))
#model.add(Dropout(0.2))
#model.add(Dense(10,activation='softmax'))
###Regression                    4.2.3 回归
model.add(Dense(64, activation='relu', input_dim=train_data.shape[1]))
model.add(Dense(1))

#Inspect Model                           5. 模型检查
##Model output shape                     5.1 模型输出形状 
model.output_shape
##Model summary representation           5.2 模型总结表示
model.summary()
##Model configuratio                     5.3 模型配置
model.get_config()
##List all weight tensors in the model   5.4 列出模型所有的权重张量
model.get_weights()

#Compile Model                        6. 编译模型
##Multi-Layer Perceptron (MLP)        6.1 多层感知机
###MLP: Binary Classification         6.1.1 二分类
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
###MLP: Multi-Class Classification    6.1.2 多分类
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
###MLP: Regression                    6.1.3 回归 
model.compile(optimizer='rmsprop', loss='mse', metrics=['mae'])
