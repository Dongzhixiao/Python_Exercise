# coding: utf-8
from __future__ import print_function
from keras.models import Sequential
from keras.layers import Dense, Activation
from keras.layers import Dropout
from keras.optimizers import RMSprop
from keras.utils import to_categorical
from keras.layers import LSTM,GRU,Bidirectional
from keras.layers.convolutional import Conv1D,Conv2D
from keras.layers.pooling import MaxPooling1D,AveragePooling1D,GlobalMaxPooling1D
from keras.callbacks import EarlyStopping
from keras.utils import plot_model
from keras.models import Model
from keras.layers import Input, Dense, Permute, Reshape, Lambda, RepeatVector,merge,Embedding
import keras.backend as K
import keras
import read_utils
import os
import pickle
import numpy as np
import sys
import json

import dense_new

def readJsonPoetry(fn,pl):
    '''
    fn代表读取json文件的路径，pl代表读取的诗词的长度
    '''
    with open(fn,encoding='UTF-8') as f:
        content = f.read()
        JsonResult = json.loads(content)
        allX = []
        allY = []
        allData = []
        for dic in JsonResult:
            singleP = dic['paragraphs']
            for line in singleP:
                if len(line) == pl*2+2 and line[pl]=='，' and line[pl*2+1]=='。':
                    allX.append(line)
                    temline = line[1:] + 'e'
                    allY.append(temline)
                    for word in line:
                        allData.append(word)
                    allData.append('e')
        return allX,allY,allData

def getNumberData(x,y,converter):
    inp = np.zeros((len(x),poetry_line_length))
    outp = np.zeros((len(y),poetry_line_length,converter.vocab_size))
    for i,c in enumerate(x):
        inp[i,:] = converter.text_to_arr(c)
    for i,c in enumerate(y):
        outp[i,:,:] = to_categorical(converter.text_to_arr(c),
                                     converter.vocab_size) 
    return inp,outp  
  
def attention_3d_block(inputs,time_steps,SINGLE_ATTENTION_VECTOR):
    # inputs.shape = (batch_size, time_steps, input_dim)
    input_dim = int(inputs.shape[2])
#    time_steps = TIME_STEPS
    a = Permute((2, 1),name='Permute')(inputs)
#    a = Reshape((input_dim, time_steps))(a) # this line is not useful. It's just to know which dimension is what.
    a = dense_new.Dense_New(time_steps,use_bias=False,name='My_Layer')(a)    #测试使用自定义层
    if SINGLE_ATTENTION_VECTOR:
        a = Lambda(lambda x: K.mean(x, axis=1), name='Dim_Reduction')(a)
        a = RepeatVector(input_dim)(a)
    a_probs = Permute((2, 1), name='Attention_Vec')(a)
    output_attention_mul = merge([inputs, a_probs], name='Attention_Mul', mode='mul')
    return output_attention_mul        

class UserRNNReader():
    def __init__(self,modelFile,converter,start,end,steps):
       
        self.model = keras.models.load_model(modelFile)
        self.converter = converter
        self.startToken = start
        self.endToken = end
        self.steps = steps
        
    def prediction_next_n(self, prime, vocab_size, next_n=3,**k):  #prime：开始的几个基本单元；vocab_size：一共有多少个类型的基本单元+1(未知数据编码)
        index = len(prime)-1  #检索的位置
        prime = [prime]
        x = keras.preprocessing.sequence.pad_sequences(prime, maxlen=self.steps, dtype='int32',padding='post', value=self.converter.word_to_int_table[self.endToken])
#        x = to_categorical(x, self.converter.vocab_size)
#        x = x.reshape(-1,x.shape[0],x.shape[1])
        y = self.model.predict(x)
        
        preds = y[0,index,:]
        
        p = np.squeeze(preds)    #squeeze函数从数组的形状中删除单维度条目，即把shape中为1的维度去掉
        # 将next_n个最大的概率的位置得到
        next_n_num = np.argsort(p,kind = 'mergesort')[-next_n:]  #argsort函数可以按照给定方法排序
        #返回的应该是标号和对应的概率值
        s_p_d = []
        for i in next_n_num:
            s_p_d.append((i,p[i]))
        return s_p_d
    
if __name__ == '__main__':
    batch_size = 100  #每批训练的大小
    poetry_line_length = 12   #输入是固定的序列长，包括一个逗号和一个句号
    
    #数据读取
    JsonFileName = 'p.json'
    allX,allY,allData = readJsonPoetry(JsonFileName,int(poetry_line_length/2-1))
    converter = read_utils.TextConverter(allData)
    x_train,y_train = getNumberData(allX[:1000],allY[:1000],converter)
    x_test,y_test = getNumberData(allX[1001:],allY[1001:],converter)
    
    inp = Input(shape=(poetry_line_length,),name='Input')
    emb = Embedding(input_dim=converter.vocab_size, 
                    output_dim=64, 
                    input_length=poetry_line_length)(inp)
    # 下面是普通RNN网络
    rnn1 = GRU(128, input_shape=(poetry_line_length, converter.vocab_size),return_sequences=True,dropout=0.5,name='RNN')(emb)
    rnn2 = GRU(128,return_sequences=True,dropout=0.5,name='2RNN')(rnn1)
    #下面是LSTM网络    
    lstm1 = LSTM(128, input_shape=(poetry_line_length, converter.vocab_size),return_sequences=True,dropout=0.5,name = 'LSTM')(emb)
    lstm2 = LSTM(128,return_sequences=True,dropout=0.5,name = '2LSTM')(lstm1)
    #下面attention
#    attention_mul = attention_3d_block(lstm1,n_steps,True)
    #下面是最后全连接输出
#    dense1 = Dense(128,name='Dense_hind')(emb)
    op_dense = Dense(converter.vocab_size,name='Dense')(rnn2)   #这里修改输出的从哪里开始
    #attention应该在这里！
    attention_mul = attention_3d_block(op_dense,poetry_line_length,True)
    op = Activation('softmax',name='Softmax_Activate')(attention_mul)
    model = Model(inputs=inp,outputs=op)
    #模型的所有参数和层进行总结
    model.summary()   
#    read_activations.get_activations(model,)
    #简单绘制一下网络图的结构
    plot_model(model, to_file='model.png')
#    sys.exit()   #这里测试，先退出
    
    #设置优化器和损失函数
    optimizer = RMSprop(lr=0.01) #该优化器通常是面对递归神经网络时的一个良好选择
#    optimizer = keras.optimizers.Adam()
    model.compile(loss='categorical_crossentropy', optimizer=optimizer,metrics=['accuracy'])
    #设置回调函数相关的操作
#    early_stopping_monitor = EarlyStopping(patience=5)  #早停
    tensor_board = keras.callbacks.TensorBoard()  #数据输出到TensorBoard
    #下面设置模型保存点,一步检测一次，只保存最好的模型
    check_point = keras.callbacks.ModelCheckpoint(
            filepath='logs\weights-{epoch:03d}-{val_acc:.4f}.hdf5',
            monitor='val_acc',
            period=1,
            save_best_only=True)  
    # histories = Histories()   #记录一些数据
    
    hsty = model.fit(x_train, y_train, batch_size=batch_size, epochs=100,validation_data=(x_test,y_test),
              callbacks=[tensor_board,check_point])  # 训练
#                  callbacks=[early_stopping_monitor,tensor_board,histories])  # 训练
    
    model.save('my_model.h5')    
        
    
    