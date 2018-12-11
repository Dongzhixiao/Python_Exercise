#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jun 23 13:57:58 2018

@author: dell
"""

#import win32con, win32api
import os
import logging

#Configure the log output location information
LOG_FORMAT = '%(asctime)s : %(levelname)s : %(message)s'
if not os.path.exists('tem'):
    os.mkdir('tem')
#win32api.SetFileAttributes('tem', win32con.FILE_ATTRIBUTE_HIDDEN)    #Set the tem folder to a hidden file
logPath = os.path.join('tem','TheLog.log')   
logging.basicConfig(format=LOG_FORMAT, level=logging.INFO, filename=logPath,filemode='a')

