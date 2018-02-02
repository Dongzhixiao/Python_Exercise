# -*- coding: utf-8 -*-
"""
Created on Thu Feb  1 23:37:16 2018

@author: Dongzhixiao
"""

import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt

def generateDrawDF(df):    #得到可以进行绘画的数据
    data = [];
    for j in range(df.iloc[:,0].size):
        LineSeries = df.iloc[j]  #得到行的序列
        for k in range(LineSeries.size):
            data.append((LineSeries[k],LineSeries.index[k]))
            #下面一行绝对不要用，DataFrame绝对不要动态加行列！！！！
#            Drawdf.loc[Drawdf.iloc[:,0].size] = [LineSeries[k],LineSeries.index[k],strList[i]]
    DataToDraw = pd.DataFrame(columns=('value','property'),data =data)
    return DataToDraw

if __name__ == '__main__':
    tips = sns.load_dataset("tips")   #读取seaborn自带的一个测试数据
 #   tips.to_excel('tips.xlsx')   #保存成excel文件，进行观察
    sns.stripplot(x="sex", y="total_bill", hue="day",data=tips,jitter=True)
    
    df = pd.read_excel("test.xlsx")
    dfToDraw = generateDrawDF(df)
    plt.figure()
    sns.boxplot(x='property',y='value',data = dfToDraw) #绘制不同属性分布的箱图
    plt.figure()
    sns.violinplot(x='property',y='value',data = dfToDraw) #绘制不同属性小提琴图