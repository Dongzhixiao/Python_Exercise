import numpy as np
import copy
import time
import tensorflow as tf
import pickle

'''
注意：后面使用“数据单元”代表数据的一个最小单元，比如训练英文数据就可以代表一个字符——'a'，训练中文数据就可以代表一个汉字——'王'etc
代码介绍：
该文件是一个工具类，用于把一个输入文件根据编码输出对应的一批一批的数据用于RNN/LSTM之类的文本处理神经网络训练，
用法是先使用TextConverter类编码所有的内容为数据单元对应数字，然后使用batch_generator函数将编码好的数字分批返回
比如：
text = f.read()  #f是open后得到的文件指针
converter = TextConverter(text)
arr = converter.text_to_arr(text)
g = batch_generator(arr,num_seqs,num_steps)   #如果输入本来就是编码好的数据，则直接使用这个函数即可
'''

def batch_generator(arr, n_seqs, n_steps):   #根据输入的arr(这个输入一般就是全部样本组成的文本，并且已经根据所有数据单元编码成为了数字列表)，返回对应的生成器，满足输入的序列个数和序列长度
    arr = copy.copy(arr)
    batch_size = n_seqs * n_steps         #计算没次输入需要使用的数据单元
    n_batches = int(len(arr) / batch_size)   #一共可以得到多少组输入数据
    arr = arr[:batch_size * n_batches]     #直接忽略了后面不能构成一组输入的数据！
    arr = arr.reshape((n_seqs, -1))
    while True:
        np.random.shuffle(arr)     #将所有行打乱顺序
        for n in range(0, arr.shape[1], n_steps):
            x = arr[:, n:n + n_steps]          #每次选择对应n_seqs行，n_steps列的数据
            y = np.zeros_like(x)    #返回跟x同形状的n维数组，数据全部都是0
            y[:, :-1], y[:, -1] = x[:, 1:], x[:, 0]
            yield x, y
#如果x是[[48 49 50]
#       [ 0  1  2]]
# 则y是[[49 50 48]
#       [ 1  2  0]]


class TextConverter(object):
    def __init__(self, text=None, max_vocab=5000, filename=None):
        if filename is not None:
            with open(filename, 'rb') as f:
                self.vocab = pickle.load(f)
        else:
            vocab = set(text)    #存储读取文件中的数据单元所有类型的集合，比如英文文件会是：{'\n','A','b',...,'\r'}
            print(len(vocab))    #打印数据单元的种类的数目
            # max_vocab_process
            vocab_count = {}     #存储每一个数据单元在整个读入的文本中出现的次数的字典
            for word in vocab:
                vocab_count[word] = 0
            for word in text:
                vocab_count[word] += 1
            vocab_count_list = []    #存储元组(数据单元，对应数量)组成的列表，然后按照数量的大小排序，比如[('a',100),('d',20),...,('x',3)]
            for word in vocab_count:
                vocab_count_list.append((word, vocab_count[word]))
            vocab_count_list.sort(key=lambda x: x[1], reverse=True)
            if len(vocab_count_list) > max_vocab:      #根据传入的最大数量的数据单元数截断前max_vocab大的数据单元，基本上不可能，除非遇到汉字之类的文本
                vocab_count_list = vocab_count_list[:max_vocab]
            vocab = [x[0] for x in vocab_count_list]
            self.vocab = vocab     #vocab仅仅存储数据单元按照出现数量从大到小的列表，例如：['a','d',...,'x']

        self.word_to_int_table = {c: i for i, c in enumerate(self.vocab)}    #  数据单元到数字字典{' ':0,'e':1,...,'c':20,...}
        self.int_to_word_table = dict(enumerate(self.vocab))         #  数字到数据单元字典{0:‘ ’，1:'e',...,20:'c',..}

    @property
    def vocab_size(self):            #这里增加一个1，为了兼容没有出现的词对应的序号编码
        return len(self.vocab) + 1

    def word_to_int(self, word):    #返回数据单元对应的整数
        if word in self.word_to_int_table:
            return self.word_to_int_table[word]
        else:
            return len(self.vocab)    #如果出现了没有出现的词，则变为<unk>对应的标记

    def int_to_word(self, index):    #返回整数对应的数据单元
        if index == len(self.vocab):
            return '<unk>'          #没有出现的词被标记为unknown的缩写
        elif index < len(self.vocab):
            return self.int_to_word_table[index]
        else:
            raise Exception('Unknown index!')

    def text_to_arr(self, text):     #将输入的text根据word_to_int返回得到对应的编码数，并构成np.ndarray并返回，例如：输入'  a\n'，则返回类似array([ 0,  0,  4, 10])
        arr = []
        for word in text:
            arr.append(self.word_to_int(word))
        return np.array(arr)

    def arr_to_text(self, arr):    #输入列表类型的数据，返回对应的数据单元的组合
        words = []
        for index in arr:
            words.append(self.int_to_word(index))
        return "".join(words)

    def save_to_file(self, filename):      #仅仅存储数据单元按照出现数量从大到小的列表到指定文件filename处，例如：['a','d',...,'x']
        with open(filename, 'wb') as f:
            pickle.dump(self.vocab, f)
