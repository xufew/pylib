# =======================
# -*- coding: utf-8 -*-
# author: LONGFEI XU
# Try your best
# ============


def one_hot(processDic, inputString):
    '''
    根据传入的编码顺序，和需要编码的字符，进行one_hot转换
    processDic = {'xxx':0, 'xxxx':1}
    '''
    totalLen = len(processDic)
    initList = [0]*totalLen
    if inputString in processDic:
        initList[processDic[inputString]] = 1
    return initList


def get_one_hot_name(processDic, addName=''):
    '''
    根据顺序获取one_hot的命名
    '''
    totalLen = len(processDic)
    initList = ['']*totalLen
    for name in processDic:
        initList[processDic[name]] = '{}{}'.format(addName, name)
    return initList
