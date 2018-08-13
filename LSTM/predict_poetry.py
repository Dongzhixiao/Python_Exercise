import os
#import model
import  read_utils
import pickle
import tensorflow as tf
#from treelib import Tree
from graphviz import Digraph

from dense_new import Dense_New
import gen_poetry

import keras
import random

if __name__ == '__main__':
    startToken = 'start'
    endToken = 'e'
    poetry_line_length = 12
    n_steps = 100
    
    #数据读取
    JsonFileName = 'p.json'
    allX,allY,allData = gen_poetry.readJsonPoetry(JsonFileName,int(poetry_line_length/2-1))
    converter = read_utils.TextConverter(allData)
    
    from gen_poetry import UserRNNReader         
    with keras.utils.CustomObjectScope({'Dense_New': Dense_New}):
        model  = UserRNNReader('my_model.h5',converter,startToken,endToken,poetry_line_length)
        
    next_n = 3   #设置预测步数    
    before_list = [converter.word_to_int_table['爱']]
    pred = model.prediction_next_n(before_list,converter.vocab_size,next_n)
    for i in range(4):
        num = pred[random.randint(0,2)][0]
        before_list.append(num)
        pred = model.prediction_next_n(before_list,converter.vocab_size,next_n)
    before_list.append(converter.word_to_int_table['，'])
    for i in range(5):
        num = pred[random.randint(0,2)][0]
        before_list.append(num)
        pred = model.prediction_next_n(before_list,converter.vocab_size,next_n)
    before_list.append(converter.word_to_int_table['。'])
    
    w = converter.arr_to_text(before_list)
    print(w) 
    