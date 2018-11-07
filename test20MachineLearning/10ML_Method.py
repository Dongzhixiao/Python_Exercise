#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Created on 2018-10-15

@author: XiaoDong Wang
"""
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

#进行评价需要导入的包
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score

def printTip(s):  
'''
根据传入字符串，打印对应长度的“*”的函数
Args:
    s 传入字符串
'''
    print(len(s)*2*'*'+'\n'+s)

MethodName = {}   #用于保存需要调用的函数方法以及对应参数的字典
# 1. 最近邻算法  (http://scikit-learn.org/stable/modules/neighbors.html)
## 1.1 K近邻算法  (http://scikit-learn.org/stable/modules/neighbors.html#classification)
from sklearn.neighbors import KNeighborsClassifier
MethodName.update({'KNeighborsClassifier':{'n_neighbors':8}})
## 1.2 固定半径法  (方法说明网址同上)
from sklearn.neighbors import  RadiusNeighborsClassifier
MethodName.update({'RadiusNeighborsClassifier':{'radius':8.0}})
# 2. 线性模型  (http://scikit-learn.org/stable/modules/linear_model.html
## 2.1 逻辑回归  (http://scikit-learn.org/stable/modules/linear_model.html#logistic-regression)
from sklearn.linear_model import LogisticRegression
MethodName.update({'LogisticRegression':{}})
## 2.2 被动攻击分类  (http://scikit-learn.org/0.19/modules/linear_model.html#passive-aggressive)
from sklearn.linear_model import PassiveAggressiveClassifier   #这个方法参数可调,每次结果不固定，可能出现预测某一行是0的情况
MethodName.update({'PassiveAggressiveClassifier':{'max_iter':100}})

# 3 朴素贝叶斯  (http://scikit-learn.org/stable/modules/naive_bayes.html)
## 3.1 高斯朴素贝叶斯  (http://scikit-learn.org/stable/modules/naive_bayes.html#gaussian-naive-bayes)
from sklearn.naive_bayes import GaussianNB
MethodName.update({'GaussianNB':{}})
## 3.2 MultinomialNB  多项式朴素贝叶斯  (http://scikit-learn.org/stable/modules/naive_bayes.html#multinomial-naive-bayes)
from sklearn.naive_bayes import MultinomialNB
MethodName.update({'MultinomialNB':{}})
## 3.3 BernoulliNB   伯努利朴素贝叶斯  (http://scikit-learn.org/stable/modules/naive_bayes.html#bernoulli-naive-bayes)
from sklearn.naive_bayes import BernoulliNB     #每次结果不固定，可能出现预测某一行是0的情况
MethodName.update({'BernoulliNB':{}})

# 4. Support Vector Machine  (http://scikit-learn.org/stable/modules/svm.html#svm-classification)
## 4.1 (http://scikit-learn.org/stable/modules/svm.html#svm-classification)
from sklearn.svm import SVC
MethodName.update({'SVC':{}})
## 4.2 Support Vector Machine's (方法说明网址同上)
from sklearn.svm import NuSVC
MethodName.update({'NuSVC':{}})
## 4.3 Linear Support Vector Classification (方法说明网址同上)
from sklearn.svm import LinearSVC
MethodName.update({'LinearSVC':{}})
# 5. Decision Tree's  (http://scikit-learn.org/stable/modules/tree.html#tree)
## 5.1 (http://scikit-learn.org/stable/modules/tree.html#classification)
from sklearn.tree import DecisionTreeClassifier
MethodName.update({'DecisionTreeClassifier':{}})
# ExtraTreeClassifier
#from sklearn.tree import ExtraTreeClassifier     #这个方法仅用于集成方法中，这里不能单独使用
#MethodName.update({'ExtraTreeClassifier':{}})
# 6. 神经网络方法 
#from sklearn.neural_network import MLPClassifier     #这个方法有点问题，待测试
#MethodName.update({'MLPClassifier':{}})
# 7. 集成学习方法  (http://scikit-learn.org/stable/modules/ensemble.html)
## 7.1 随机森林  (http://scikit-learn.org/stable/modules/ensemble.html#random-forests)
from sklearn.ensemble import RandomForestClassifier   #会引入Numpy removed的提示
MethodName.update({'RandomForestClassifier':{'max_depth':2}})
## 7.2 bagging方法  (http://scikit-learn.org/stable/modules/ensemble.html#bagging-meta-estimator)
from sklearn.ensemble import BaggingClassifier
MethodName.update({'BaggingClassifier':{}})
## 7.3 AdaBoost方法  (http://scikit-learn.org/stable/modules/ensemble.html#adaboost)
from sklearn.ensemble import AdaBoostClassifier
MethodName.update({'AdaBoostClassifier':{}})
## 7.4 梯度提升分类器  (http://scikit-learn.org/stable/modules/ensemble.html#gradient-tree-boosting)
from sklearn.ensemble import GradientBoostingClassifier
MethodName.update({'GradientBoostingClassifier':{}})
# 8. 线性与二次判别分析  (http://scikit-learn.org/stable/modules/lda_qda.html)
## 8.1 LDA方法  (http://scikit-learn.org/stable/modules/lda_qda.html#dimensionality-reduction-using-linear-discriminant-analysis)
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
MethodName.update({'LinearDiscriminantAnalysis':{}})
## 8.2 QDA方法  (方法说明网址同上)
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
MethodName.update({'QuadraticDiscriminantAnalysis':{}})

def testAllMethod():
'''
进行多种机器学习方法测试的函数
'''
    for s,v in MethodName.items():
        printTip('使用'+s+'方法的结果如下：')
        Model = eval(s)(**v)
        Model.fit(X_train, y_train)
        y_pred = Model.predict(X_test)
        # Summary of the predictions made by the classifier
        print(classification_report(y_test, y_pred))
        print(confusion_matrix(y_test, y_pred))
        # Accuracy score
        print('accuracy is',accuracy_score(y_test,y_pred))

if __name__ == '__main__':
    dataset = pd.read_csv('Iris.csv')   #读取数据
    
    X = dataset.iloc[:, :-1].values
    y = dataset.iloc[:, -1].values
    
    # Splitting the dataset into the Training set and Test set
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)
    
    testAllMethod()

#下面是绘图测试   
#    sns.FacetGrid(dataset, hue="Species", size=5) \
#    .map(plt.scatter, "SepalLengthCm", "SepalWidthCm") \
#    .add_legend()
#    plt.show()
        

    








