# -*- coding: utf-8 -*-
"""
Created on Tue Jul 31 16:19:59 2018

@author: dell
"""

import os
#import model
import  read_utils
import pickle
import tensorflow as tf
from treelib import Tree
from graphviz import Digraph
import numpy as np

from dense_new import Dense_New
from dense_attention import Dense_Attention

import keras
from keras.models import Model

import matplotlib.pyplot as plt

from keras.utils import plot_model

import model_keras


def decodeToContent(Co,conver,endT,isContent):
    content = Co[0,:,:]
    conLis = []
    for i in range(content.shape[0]):
        conLis.append(np.where(content[i,:]==1)[0][0])
    inputLength = conLis.index(converter.word_to_int_table[endToken])+1
    a = conLis[0:inputLength]
    b = conLis[1:1+inputLength]
    if isContent:
        a = [conver.int_to_word_table[i] for i in a]
        b = [conver.int_to_word_table[i] for i in b]
    return a,b

if __name__ == '__main__':
    startToken = 'start'
    endToken = 'end'
    n_seqs = 10
    n_steps = 100
    model_path = os.path.join('model', 'behavior')
    checkpoint_path = model_path
    with open(os.path.join(model_path,'allData.pkl'),'rb') as f:   #读取所有数据
        allData = pickle.load(f) 
    with open(os.path.join(model_path,'testData.pkl'),'rb') as f:  #读取测试数据
        testData = pickle.load(f)
    with open(os.path.join(model_path,'trainData.pkl'),'rb') as f:  #读取测试数据
        trainData = pickle.load(f)
    converter = read_utils.TextConverter(allData)  #要用所有数据编码
    
    if os.path.isdir(checkpoint_path):
        checkpoint_path =\
            tf.train.latest_checkpoint(checkpoint_path)
    x,y = model_keras.readAndOneHot(allData,converter,startToken,endToken)
#    model = model.CharRNN(converter.vocab_size,sampling=True,
#                    num_seqs=n_seqs,
#                    num_steps=n_steps,
#                    # lstm_size=FLAGS.lstm_size,
#                    # num_layers=FLAGS.num_layers,
#                    # learning_rate=FLAGS.learning_rate,
#                    # train_keep_prob=FLAGS.train_keep_prob,
#                    use_embedding=True,
#                    # embedding_size=FLAGS.embedding_size
#                    )
#    model.load(checkpoint_path)   #读取权重
    
     
    with keras.utils.CustomObjectScope({'Dense_Attention': Dense_Attention}):
        model  = model_keras.UserRNNReader('my_model.h5',converter,startToken,endToken,n_steps)
    
    model = model.model
    
    #建立绘图的模型
    theInput = model.input
    theOutput = model.get_layer('My_Layer').output  #注意修改！！！！！！！！！！！
    draw_model = Model(inputs=theInput,outputs=theOutput)
    # draw_model.predict()

    plot_model(model, to_file='model.png')
    
    
#    activation_map = model.get_layer('My_Layer').get_weights()[0]
    Num = 6   #设置输入的行为模式的序号
    inputCode = x[Num:Num+1,:,:]     #得到输入编码内容
    outputCode = y[Num:Num+1,:,:]    #得到输出编码内容
    activation_map = draw_model.predict(inputCode)[0,:,:]
    inputContent,outputContent = decodeToContent(inputCode,converter,endToken,True)
    input_length = len(inputContent)
    output_length = len(outputContent)
    activation_map = activation_map[:input_length,:output_length]   #这里的20是测试，代表输入是20个单位长度的序列
    
 # import seaborn as sns
    plt.clf()
    f = plt.figure(figsize=(8, 8.5))
    ax = f.add_subplot(1, 1, 1)
    
    # add image
    i = ax.imshow(activation_map, interpolation='nearest', cmap='gray')
    
    # add colorbar
    from mpl_toolkits.axes_grid1 import make_axes_locatable
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.25)
    cbar = f.colorbar(i, cax=cax, orientation='vertical')
    cbar.ax.set_xlabel('Probability', labelpad=2)

    # add labels
    ax.set_yticks(range(output_length))
    ax.set_yticklabels(outputContent)
    
    ax.set_xticks(range(input_length))
    ax.set_xticklabels(inputContent, rotation=90)
    
    ax.set_xlabel('Input Sequence')
    ax.set_ylabel('Output Sequence')

    # add grid and legend
    ax.grid()
    ax.legend(loc='best')

#    f.savefig(os.path.join(HERE, 'attention_maps', text.replace('/', '')+'.pdf'), bbox_inches='tight')
#    f.show()


    plt.show()
    
    
