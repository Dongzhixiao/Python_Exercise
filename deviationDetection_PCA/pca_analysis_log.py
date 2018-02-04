# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import matplotlib.pyplot as plt  #用于类似matlab绘图的包
plt.rcParams['font.sans-serif']=['SimHei'] #用来正常显示中文标签
plt.rcParams['axes.unicode_minus']=False #用来正常显示负号

from sklearn import decomposition as skldec #用于主成分分析降维的包
from mpl_toolkits.mplot3d import Axes3D #用于3D绘图的包
import seaborn as sns   #用于高阶画图的包
import pandas as pd  
import numpy as np
from sklearn import preprocessing #用于预处理的包
import math   #用于计算开方
from scipy.stats import norm  #用于计算标准正态分布的分位数

def testPCA():
    a = np.array([1,2,3,4,5])
    
def drawTypeCorrlation(fileName):
    df = pd.read_excel(fileName)
    #sns.linearmodels.corrplot(df)  #画相关性图
    
def caculateLogVector(fileName):
     # 读取数据
    df = pd.read_excel(fileName)
    pca = skldec.PCA(n_components = 0.90)
    pca.fit(df)   #主城分析时每一行是一个输入数据
    result = pca.transform(df)
    df_origin = pca.inverse_transform(result)
    
    ndarr = np.array(df)  #输入数据，这里每一行是一个输入数据
    ndarr = preprocessing.scale(ndarr,with_std=False) #中心化数据
    Y = ndarr.T #使得每一个列是一个输入数据
    #result2 = pca.components_.dot(Y).T   #该结果等于使用pca的transform的结果
    P = pca.components_.T   #原来公式里的P是主成分列向量组成的
    Y_a = Y - P.dot(P.T).dot(Y)
    Y_a_2norm_sqr = np.zeros(Y_a.shape[1]).reshape(1,Y_a.shape[1])
    for i in range(Y_a.shape[1]):
        Y_a_2norm_sqr[0,i] = np.linalg.norm(Y_a[:,i])**2
        
    #sns.distplot(Y_a_2norm_sqr);
    #sns.tsplot(data=Y_a_2norm_sqr[0,:10000],err_style = "ci_bars",interpolate = False)
    
    x = np.linspace(1, Y_a.shape[1] , Y_a.shape[1])
    plt.scatter(x,Y_a_2norm_sqr, s=2)
    plt.xlabel(u'Sample Datas',{'color':'r'})
    plt.ylabel(u'Q Values',{'color':'r'})
    #plt.hold
    lineToDraw = caculateQ_Alpha(df,pca,0.05)
    plt.plot(x,[lineToDraw for i in range(Y_a.shape[1])],'-r',label = 'normal',linewidth=1)
    plt.plot(x,[2*lineToDraw for i in range(Y_a.shape[1])],'-g',label = 'slight',linewidth=1)
    plt.plot(x,[4*lineToDraw for i in range(Y_a.shape[1])],'-b',label = 'warning',linewidth=1)
    plt.plot(x,[8*lineToDraw for i in range(Y_a.shape[1])],'-k',label = 'critical',linewidth=1)
    plt.legend()
    # return 异常对应的序号
    
#==============================================================================
#     numListToReturn = {'Critical':[],
#                        'Error':[],
#                        'Warning':[],
#                        'Slight':[],
#                        'Normal':[]};
#     for i in range(Y_a_2norm_sqr.shape[1]):
#         if Y_a_2norm_sqr[0,i] > 8*lineToDraw:
#             numListToReturn['Critical'].append((i,Y_a_2norm_sqr[0,i]))
#         elif Y_a_2norm_sqr[0,i] <= 8*lineToDraw and Y_a_2norm_sqr[0,i] > 4*lineToDraw:
#             numListToReturn['Error'].append((i,Y_a_2norm_sqr[0,i]))
#         elif Y_a_2norm_sqr[0,i] <= 4*lineToDraw and Y_a_2norm_sqr[0,i] > 2*lineToDraw:
#             numListToReturn['Warning'].append((i,Y_a_2norm_sqr[0,i]))
#         elif Y_a_2norm_sqr[0,i] <= 2*lineToDraw and Y_a_2norm_sqr[0,i] > 1*lineToDraw:
#             numListToReturn['Slight'].append((i,Y_a_2norm_sqr[0,i]))
#         else:
#             numListToReturn['Normal'].append((i,Y_a_2norm_sqr[0,i]))
#==============================================================================
    
    numListToReturn = {'Abnormal':[],
                       'Normal':[]};
    for i in range(Y_a_2norm_sqr.shape[1]):
        if Y_a_2norm_sqr[0,i] > lineToDraw:
            numListToReturn['Abnormal'].append((i,Y_a_2norm_sqr[0,i]))
        else:
            numListToReturn['Normal'].append((i,Y_a_2norm_sqr[0,i]))
        
    return numListToReturn,df
    
def caculateQ_Alpha(df,pca,alpha):
    #pca.fit(df)
    r = pca.explained_variance_.size
    pca = skldec.PCA()
    pca.fit(df)
    Phi_1 = 0
    Phi_2 = 0
    Phi_3 = 0
    h_0 = 0
    if r == pca.explained_variance_.size:  #如果所有数据投影到原空间，则Q值都为零，该方法失效
        return 0
    for i in range(r,pca.explained_variance_.size):
        Phi_1 = Phi_1 + pca.explained_variance_[i]**1
        Phi_2 = Phi_2 + pca.explained_variance_[i]**2
        Phi_3 = Phi_3 + pca.explained_variance_[i]**3
#    print(str(Phi_1)+'\t'+str(Phi_2)+'\t'+str(Phi_3))        
    h_0 = 1 - 2*Phi_1*Phi_3/(3*Phi_2**2)
    C_alpha = norm.cdf(alpha)   #得到正态分布alpha的百分位数
    delta2_Alpha = Phi_1*(C_alpha*math.sqrt(2*Phi_2*h_0**2)/Phi_1 + 1 +
                          Phi_2*h_0*(h_0-1)/Phi_1**2)**(1/h_0) 
    return delta2_Alpha

def draw_histogram(ls):
    pass

def generateDrawDF(strList,df,AnomalousLine):    #得到可以进行绘画的数据
    data = [];
    for i in range(len(strList)):
        levelList = AnomalousLine[strList[i]]  #得到等级对应的列表
        for j in range(len(levelList)):
            lineNum = levelList[j][0]  #得到行号
            LineSeries = df.iloc[lineNum]  #得到行的序列
            for k in range(int(LineSeries.size)):    
                #print(LineSeries[k],LineSeries.index[k],strList[i])
                data.append((LineSeries[k],LineSeries.index[k],strList[i]))
                #下面一行绝对不要用，DataFrame绝对不要动态加行列！！！！
#                Drawdf.loc[Drawdf.iloc[:,0].size] = [LineSeries[k],LineSeries.index[k],strList[i]]     
#        print('计算了一次')
#    print(data)
    Drawdf = pd.DataFrame(columns=('times','type','level'),data = data)
    return Drawdf

def getSum(arrLike):
    labelList = [i for i in range(46)]
    sumToReturn = 0
    for i in labelList:
        sumToReturn = sumToReturn + arrLike[i]
    return sumToReturn

if __name__ == '__main__':
    FileName = "test.xlsx"
#    drawTypeCorrlation(FileName)
    [AnomalousLine,df] = caculateLogVector(FileName)

#####################################################################下面是找到Q值最大的几行和正常的Q值平均进行比较
#    listLevel = ['null' for i in range(df.shape[0])] ;  listQ_value = [i for i in range(df.shape[0])];
#    listSecNum = [i for i in range(df.shape[0])]
#    for key,value in AnomalousLine.items():
#        for tup in value:
#            lineNum = tup[0]
#            Q_value = tup[1]
#            listLevel[lineNum] = key 
#            listQ_value[lineNum] = Q_value            
#    df['Level'] = listLevel; df['Q_value'] = listQ_value; #df['SecNum'] = listSecNum    #增加三列——等级/Q值/秒数    
#    df['typeSum'] = df.apply(getSum , axis = 1)
#    normalMean = df[df['Level'].isin(['Normal'])].mean()    #选取Normal的行求平均值组成Series    
#    largeNQ = df.nlargest(50,'Q_value').drop('Level',axis = 1)  #选取Q值最大的10个行组成DataFrame    
#    result = largeNQ.sub(normalMean)
#    result = result.T
#    result.plot(kind = 'bar')
#    sns.pairplot(df, vars=["type9","type10"], hue ="Level", diag_kind="kde" #kind='reg'
#                 )
#    
#    sns.pairplot(df, vars=["type1","type7","type11"], hue ="Level", diag_kind="kde" #kind='reg'
#                 )

#    sns.pairplot(df, vars=["type4","type5","type6","type8","type10","type11","type12","type15",
#                           "type16","type19"], hue ="Level", diag_kind="kde" #kind='reg'
#                 )
    
####################################################################下面是使用循环将多个类型每GroupNum为一组绘制出来
#    GroupNum = 3   
#    TypeList = ['type'+str(i) for i in range(46) ]
#    ListToDraw = []
#    for i in range(len(TypeList)):
#        ListToDraw.append(TypeList[i])
#        if (i+1)%GroupNum == 0:
#            #plt.figure()
#            
#            sns.pairplot(df, vars=ListToDraw, diag_kind="kde", hue = "Level" #kind='reg'
#                 )
#            ListToDraw.clear()

            
    
    
#####################################################################下面是Seaborn画图代码    
    strList = ['Normal','Abnormal']
#    strList = ['Normal','Critical']
#    strList = ['Normal']
    Drawdf = generateDrawDF(strList,df,AnomalousLine)
    
    plt.figure()
    sns.boxplot(x="type", y="times", hue="level",
                     data=Drawdf, palette="Set2")
    plt.figure()
    result = sns.barplot(x="type", y="times", hue="level", ci = None , capsize=.1, #errwidth = 0,
                     data=Drawdf, palette="Set2")

    plt.figure()
    sns.violinplot(x="type", y="times", hue="level",
                     data=Drawdf, palette="Set2")

    