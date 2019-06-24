# -*- coding: utf-8 -*-
"""
Created on Wed Jan 10 22:01:57 2018

@author: Dongzhixiao
"""

import pandas as pd
import datetime   #用来计算日期差的包
import orangecontrib.associate.fpgrowth as oaf  #进行关联规则分析的包
import functools

def dataInterval(data1,data2):
    d1 = datetime.datetime.strptime(data1, '%Y-%m-%d')
    d2 = datetime.datetime.strptime(data2, '%Y-%m-%d')
    delta = d1 - d2
    return delta.days

def getInterval(arrLike):  #用来计算日期间隔天数的调用的函数
    PublishedTime = arrLike['PublishedTime']
    ReceivedTime = arrLike['ReceivedTime']
#    print(PublishedTime.strip(),ReceivedTime.strip())
    days = dataInterval(PublishedTime.strip(),ReceivedTime.strip())  #注意去掉两端空白
    return days

def dealRules(rules,strDecode):
    returnRules = []
    for i in rules:
        temStr = '';
        for j in i[0]:   #处理第一个frozenset
            temStr = temStr+strDecode[j]+'&'
        temStr = temStr[:-1]
        temStr = temStr + ' ==> '
        for j in i[1]:
            temStr = temStr+strDecode[j]+'&'
        temStr = temStr[:-1]
        temStr = temStr + ';' +'\t'+str(i[2])+ ';' +'\t'+str(i[3])
#        print(temStr)
        returnRules.append(temStr)
    return returnRules

def dealResult(rules,strDecode):
    returnRules = []
    for i in rules:
        temStr = '';
        for j in i[0]:   #处理第一个frozenset
            temStr = temStr+strDecode[j]+'&'
        temStr = temStr[:-1]
        temStr = temStr + ' ==> '
        for j in i[1]:
            temStr = temStr+strDecode[j]+'&'
        temStr = temStr[:-1]
        temStr = temStr + ';' +'\t'+str(i[2])+ ';' +'\t'+str(i[3])+ ';' +'\t'+str(i[4])+ ';' +'\t'+str(i[5])+ ';' +'\t'+str(i[6])+ ';' +'\t'+str(i[7])
#        print(temStr)
        returnRules.append(temStr)
    return returnRules

def ResultDFToSave(rules,strDecode):   #根据Qrange3关联分析生成的规则得到并返回对于的DataFrame数据结构的函数
    returnRules = []
    for i in rules:
        temList = []
        temStr = '';
        for j in i[0]:   #处理第一个frozenset
            temStr = temStr + strDecode[j] + '&'
        temStr = temStr[:-1]
        temStr = temStr + ' ==> '
        for j in i[1]:
            temStr = temStr + strDecode[j] + '&'
        temStr = temStr[:-1]
        temList.append(temStr); temList.append(i[2]); temList.append(i[3]); temList.append(i[4])
        temList.append(i[5]); temList.append(i[6]); temList.append(i[7])
        returnRules.append(temList)
    return pd.DataFrame(returnRules,columns=('规则','项集出现数目','置信度','覆盖度','力度','提升度','利用度'))
    

if __name__ == '__main__':    
    fileName = "NS_new.xls";
    df = pd.read_excel(fileName) 
    df['TimeInterval'] = df.apply(getInterval , axis = 1)
    listToAnalysis = []
    listToStore = []
    for i in range(df.iloc[:,0].size):             #df.iloc[:,0].size
        #处理ArticleTag段位
        s = df.iloc[i]['ArticleTag']
        #s = re.sub('\s','_',s.split('/')[1].strip())
        s = s.strip()
        s = 'ArticleTag_'+s
        listToStore.append(s)
        #处理TimeInterval段位
        s = df.iloc[i]['TimeInterval']
        if s > 300:
            s = 'TimeInterval_'+'300'
        elif s > 200 and s <= 300:
            s = 'TimeInterval_'+'200_300'
        elif s > 100 and s <= 200:
            s = 'TimeInterval_'+'100_200'
        elif s <= 100:
            s = 'TimeInterval_'+'0_100'
        listToStore.append(s)
        #处理ReferencesNumber段位
        s = df.iloc[i]['ReferencesNumber']
        if s > 80:
            s = 'ReferencesNumber'+'80'
        elif s > 60 and s <= 80:
            s = 'ReferencesNumber'+'60_80'
        elif s > 40 and s <= 60:
            s = 'ReferencesNumber'+'40_60'
        elif s > 20 and s <= 40:
            s = 'ReferencesNumber'+'20_40'
        elif s > 0 and s <= 20:
            s = 'ReferencesNumber'+'0_20'
        listToStore.append(s)
        #处理Country段位
        s = df.iloc[i]['Country']
        s = 'Country_'+s.strip()
        listToStore.append(s)
        #print(listToStore)
        listToAnalysis.append(listToStore.copy())
        listToStore.clear()
    #进行编码，将listToAnalysis里面的字符串转换成整数
    strSet = set(functools.reduce(lambda a,b:a+b, listToAnalysis))
    strEncode = dict(zip(strSet,range(len(strSet)))) #编码字典，即:{'ArticleTag_BS': 6,'Country_Argentina': 53,etc...}
    strDecode = dict(zip(strEncode.values(), strEncode.keys()))  #解码字典，即:{6:'ArticleTag_BS',53:'Country_Argentina',etc...}
    listToAnalysis_int = [list(map(lambda item:strEncode[item],row)) for row in listToAnalysis]
    #开始进行关联分析     
    supportRate = 0.02
    confidenceRate = 0.5     
    itemsets = dict(oaf.frequent_itemsets(listToAnalysis_int, supportRate))        
    rules = oaf.association_rules(itemsets, confidenceRate)
    rules = list(rules)
    regularNum = len(rules)
    printRules = dealRules(rules,strDecode)  #该变量可以打印查看生成的规则
    result = list(oaf.rules_stats(rules, itemsets, len(listToAnalysis_int)))   #下面这个函数改变了rules，把rules用完了！
    printResult = dealResult(result,strDecode)  #该变量可以打印查看结果
    
#################################################下面将结果保存成excel格式的文件    
    dfToSave = ResultDFToSave(result,strDecode)
    saveRegularName = str(supportRate)+'支持度_'+str(confidenceRate)+'置信度_产生了'+str(regularNum)+'条规则'+'.xlsx'
    dfToSave.to_excel(saveRegularName)

#######################################################下面是根据不同置信度和关联度得到关联规则数目
    listTable = []
    supportRate = 0.01
    confidenceRate = 0.1
    for i in range(9):
        support = supportRate*(i+1)
        listS = []
        for j in range(9):
            confidence = confidenceRate*(j+1)
            itemsets = dict(oaf.frequent_itemsets(listToAnalysis_int, support))
            rules = list(oaf.association_rules(itemsets, confidence))
            listS.append(len(rules))
        listTable.append(listS)    
    dfList = pd.DataFrame(listTable,index = [supportRate*(i+1) for i in range(9)],columns=[confidenceRate*(i+1) for i in range(9)])
    dfList.to_excel('regularNum.xlsx')

####################################################################下面是测试pymining工具包的代码
#from pymining import itemmining, assocrules, perftesting
#transactions = (('a', 'b', 'c'), ('b'), ('a'), ('a', 'c', 'd'), ('b', 'c'), ('b', 'c'))
#relim_input = itemmining.get_relim_input(transactions)
#
#
#item_sets = itemmining.relim(relim_input, min_support=2)
#
#rules = assocrules.mine_assoc_rules(item_sets, min_support=2, min_confidence=0.5)

