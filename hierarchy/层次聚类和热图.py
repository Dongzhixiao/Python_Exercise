# -*- coding: utf-8 -*-
"""
Created on Sun Jan 14 18:32:33 2018

@author: dell
"""

import pandas as pd
import seaborn as sns  #用于话热图的工具包
from scipy.cluster import hierarchy  #用于进行层次聚类，话层次聚类图的工具包
from scipy import cluster   
import matplotlib.pyplot as plt
from sklearn import decomposition as skldec #用于主成分分析降维的包

df = pd.read_excel("test.xlsx")
#df = df.T    #python默认每行是一个样本，如果数据每列是一个样本的话，转置一下即可

#开始画层次聚类树状图    
Z = hierarchy.linkage(df, method ='ward',metric='euclidean')
plt.figure()
dn = hierarchy.dendrogram(Z,labels = df.index)

#在某个高度进行剪切
label = cluster.hierarchy.cut_tree(Z,height=0.8)

pca = skldec.PCA(n_components = 0.90)    #这就保证两个数据了
pca.fit(df)   #主城分析时每一行是一个输入数据
result = pca.transform(df)  #计算结果
plt.figure()
plt.scatter(result[:, 0], result[:, 1],
           c=label, edgecolor='k')
for i in range(result[:,0].size):
    print(result[i,0],result[i,1])
    plt.text(result[i,0],result[i,1],df.index[i])
x_label = 'PC1(%s%%)' % round((pca.explained_variance_ratio_[0]*100.0),2)
y_label = 'PC1(%s%%)' % round((pca.explained_variance_ratio_[1]*100.0),2)
plt.xlabel(x_label)
plt.ylabel(y_label)
plt.show()

#层次聚类的热图和聚类图
sns.clustermap(df,method ='ward',metric='euclidean')

