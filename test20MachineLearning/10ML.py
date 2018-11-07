# -*- coding: utf-8 -*-
"""
Created on 2018-10-15

@author: XiaoDong Wang
"""
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

def printTip(s):
    print(len(s)*2*'*'+'\n'+s)

if __name__ == '__main__':
    dataset = pd.read_csv('Iris.csv')   #读取数据
    printTip('数据类型是：')
    print(type(dataset))    #打印数据类型
    printTip('数据形状是：')
    print(dataset.shape)   #打印数据形状
    printTip('数据大小是：')
    print(dataset.size)    #打印数据大小
    printTip('数据信息是：')
    print(dataset.info())
    printTip('数据种类是:')
    print(dataset['Species'].unique())
    printTip('数据种类对应统计数目是：')
    print(dataset["Species"].value_counts())
    printTip('开头五行数据是：')
    print(dataset.head(5))
    printTip('结尾五行数据是：')
    print(dataset.tail(5))
    printTip('随机五行数据是：')
    print(dataset.sample(5))
    
    sns.FacetGrid(dataset, hue="Species", size=5).map(plt.scatter, "SepalLengthCm", "SepalWidthCm") \
    .add_legend()
#    plt.figure()
    dataset.plot(kind='box', subplots=True, layout=(2,3), sharex=False, sharey=False)
#    plt.figure()
    sns.boxplot(x="Species", y="PetalLengthCm", data=dataset )
#    plt.figure()
    ax= sns.boxplot(x="Species", y="PetalLengthCm", data=dataset)
    ax= sns.stripplot(x="Species", y="PetalLengthCm", data=dataset, jitter=True, edgecolor="gray")
#    plt.show()
#    plt.figure()
    dataset.hist(figsize=(15,20))
#    plt.figure()
    pd.plotting.scatter_matrix(dataset,figsize=(10,10))
#    plt.figure()
    plt.show()