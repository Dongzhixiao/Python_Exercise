#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# def fact(n):
    # if n==1:
        # return 1
    # return n * fact(n - 1)

import requests
import itchat
from itchat.content import *

import json

def writeConfig(OPEN, I):
    new_dict = {'OPEN':OPEN,'I':I}
    with open('config.txt',"w") as f:
        json.dump(new_dict,f)
def readConfig():
    with open('config.txt',"r") as f:
        load = json.load(f)
        return load

KEY = ['a',
       'b',
       'c',
       'd',
       'e']   #这里输入你自己注册的KEY，详见http://www.tuling123.com/

# I = 0

# OPEN = False

def get_response(msg,i):
    # 这里实现与图灵机器人的交互
    # 构造了要发送给服务器的数据
    apiUrl = 'http://www.tuling123.com/openapi/api'
    data = {
        'key' : KEY[i],
      'info' : msg,
      'userid' : 'wechat-robot',
    }
    try:
        r = requests.post(apiUrl, data=data).json()
        # 字典的get方法在字典没有'text'值的时候会返回None而不会抛出异常
        if 'url'in r:                                         
            return r.get('text')+'\n网址是：' + r.get('url')      
        return r.get('text')
    # 为了防止服务器没有正常响应导致程序异常退出，这里用try-except捕获了异常
    # 如果服务器没能正常交互（返回非json或无法连接），那么就会进入下面的return
    except:
        # 将会返回一个None
        return None

# 这里实现微信消息的获取
# @itchat.msg_register(itchat.content.TEXT)
# def tuling_reply(msg):
#     if OPEN:
#         # 为了保证在图灵Key出现问题的时候仍旧可以回复，这里设置一个默认回复
#         defaultReply = 'I received: ' + msg['Text']
#         # 如果图灵Key出现问题，那么reply将会是None
#         reply = get_response(msg['Text'])
#         # a or b的意思是，如果a有内容，那么返回a，否则返回b
#         # 有内容一般就是指非空或者非None，你可以用`if a: print('True')`来测试
#         print(reply)
#         return reply or defaultReply
@itchat.msg_register([TEXT])
def fileHelper(msg):
    config = readConfig()
    # print(config)
    OPEN = config['OPEN']
    I = config['I']
    try:
        if msg['FromUserName'] == '@1dde4063ba4deff1ce21537aafcea605b339dcf888bbcdf9379fe4d7a5fa5ae5':   #这个是我的FromUserName，唯一值
            theReturn = ''
            if msg['Text'] in ['启动机器人','启动']:
                writeConfig(True,I)
                theReturn = '机器人启动成功'        
            elif msg['Text'] in ['关闭机器人','关闭']:
                writeConfig(False,I)
                theReturn= '机器人关闭成功'
            elif msg['Text'] in ['机器人1号','机器人一号','1号','一号']:
                writeConfig(OPEN,0)
                theReturn='机器人1号启动成功'
            elif msg['Text'] in ['机器人2号','机器人二号','2号','二号']:
                writeConfig(OPEN,1)
                theReturn = '机器人2号启动成功'
            elif msg['Text']  in ['机器人3号','机器人三号','3号','三号']:
                writeConfig(OPEN,2)
                theReturn='机器人3号启动成功'
            elif msg['Text']  in ['机器人4号','机器人四号','4号','四号']:
                writeConfig(OPEN,3)
                theReturn='机器人4号启动成功'
            elif msg['Text']  in ['机器人5号','机器人五号','5号','五号']:
                writeConfig(OPEN,4)
                theReturn='机器人5号启动成功'
            elif msg['Text'] in ['信息']:
                if OPEN:
                    theReturn='机器人开启状态'
                else:
                    theReturn='机器人开启状态'
                theReturn = theReturn + '\n机器人%s号正在服务' % I
            elif msg['ToUserName'] == 'filehelper' and OPEN:
                reply = get_response(msg['Text'], I)
                itchat.send(reply, 'filehelper')
            if theReturn != '':
                print(theReturn)
                itchat.send(theReturn, 'filehelper')
        else:
            if OPEN:
                # 为了保证在图灵Key出现问题的时候仍旧可以回复，这里设置一个默认回复
                defaultReply = 'I received: ' + msg['Text']
                # 如果图灵Key出现问题，那么reply将会是None
                reply = get_response(msg['Text'],I)
                # a or b的意思是，如果a有内容，那么返回a，否则返回b
                # 有内容一般就是指非空或者非None，你可以用`if a: print('True')`来测试
                print(reply)
                return reply or defaultReply
    except  Exception as e:
        print(e)
# 为了让实验过程更加方便（修改程序不用多次扫码），我们使用热启动
itchat.auto_login(hotReload=True)
itchat.run()

# writeConfig(True,1)
# readConfig()
