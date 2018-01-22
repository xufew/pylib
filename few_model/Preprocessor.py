# =======================
# -*- coding: utf-8 -*-
# author: LONGFEI XU
# Try your best
# ============
import numpy as np


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


class NaProducer:
    def __init__(self):
        self.naList = []
        self.useList = []

    def produce(self, inputSeries, sameThres=0.7, diffThres=0.8):
        '''
        根据输入list来判断用哪个值来填充空缺值
        '''
        inputSeries = inputSeries.dropna()
        totalValue = len(inputSeries)
        count = np.unique(inputSeries.dropna(), return_counts=True)
        countDic = dict(zip(count[1], count[0]))
        # 计算是否可以用众数
        sameValue = max(countDic)
        samePer = sameValue/float(totalValue)
        con1 = samePer > sameThres
        # 计算是否可以用均值
        diffValue = len(countDic)
        diffPer = diffValue/float(totalValue)
        con2 = diffPer > diffThres
        if con1:
            outValue = countDic[sameValue]
        elif con2:
            outValue = np.mean(inputSeries)
        else:
            outValue = np.median(inputSeries)
        self.naList.append(outValue)
        self.useList.append(outValue)
        return outValue

    def get(self):
        outValue = self.naList[0]
        self.naList.pop(0)
        if len(self.naList) == 0:
            self.naList = self.useList.copy()
            print('finish')
        return outValue
