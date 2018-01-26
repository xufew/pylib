# =======================
# -*- coding: utf-8 -*-
# author: LONGFEI XU
# Try your best
# ============
import os
import subprocess

import numpy as np
from sklearn import preprocessing


def one_hot(processDic, inputString, sep=','):
    '''
    根据传入的编码顺序，和需要编码的字符，进行one_hot转换
    processDic = {'xxx':0, 'xxxx':1}
    '''
    totalLen = len(processDic)
    initList = ['0']*totalLen
    if inputString in processDic:
        initList[processDic[inputString]] = '1'
    return sep.join(initList)


def get_one_hot_name(processDic, addName='', sep=','):
    '''
    根据顺序获取one_hot的命名
    '''
    totalLen = len(processDic)
    initList = ['']*totalLen
    for name in processDic:
        initList[processDic[name]] = '{}{}'.format(addName, name)
    return sep.join(initList)


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


class Normalizer():
    '''
    将数据进行处理的方法，主要包括
    标准化，区间缩放，正则化
    '''
    def __init__(self):
        self.model = {}

    def min_max_first(self, inputData, useRange=[0, 1]):
        '''
        进行缩放，缩放到useRange区间
        useRange例如(0, 1) 缩放到0到1的区间
        '''
        min_max_scaler = preprocessing.MinMaxScaler(useRange)
        inputDataMinMax = min_max_scaler.fit_transform(inputData)
        self.model['min_max'] = min_max_scaler
        return inputDataMinMax

    def min_max_second(self, inputData, model=None):
        '''
        之前对已有数据进行过缩放，第二次对另一数据进行相同缩放
        '''
        if model is not None:
            inputDataMinMax = model.transform(inputData)
        else:
            inputDataMinMax = self.model['min_max'].transform(inputData)
        return inputDataMinMax

    def z_score_first(self, inputData):
        '''
        进行z_score的标准化处理，即正态分布
        对于每个属性/每列来说所有数据都聚集在0附近，方差为1
        '''
        scalar = preprocessing.StandardScaler()
        scalar.fit(inputData)
        inputDataTrans = scalar.transform(inputData)
        self.model['z_score'] = scalar
        return inputDataTrans

    def z_score_second(self, inputData, model=None):
        '''
        用之前记录的z-score进行转换
        '''
        if model is not None:
            inputDataTrans = model.transform(inputData)
        else:
            inputDataTrans = self.model['z_score'].transform(inputData)
        return inputDataTrans

    def normalize_first(self, inputData):
        '''
        正则化，如果后面要使用如二次型（点积）或者其它核方法
        计算两个样本之间的相似性这个方法会很有用
        '''
        normalizer = preprocessing.Normalizer()
        normalizer.fit(inputData)
        inputDataTrans = normalizer.transform(inputData)
        self.model['normalize'] = normalizer
        return inputDataTrans

    def normalize_second(self, inputData, model=None):
        '''
        '''
        if model is not None:
            inputDataTrans = model.transform(inputData)
        else:
            inputDataTrans = self.model['normalize'].transform(inputData)
        return inputDataTrans


class Spliter():
    '''
    将原始数据进行分割
    '''
    def __init__(self):
        pass

    def split_train_test(
            self,
            dataPath,
            trainPath,
            testPath,
            feature={}
            ):
        '''
        根据输入的文件地址，将其按比例分为两份，
        一份作为训练集，一份作为测试集，并进行保存
        '''
        featureDic = {
                'splitPer': 0.7,
                }
        featureDic.update(feature)
        # 获取行数
        runSh = 'wc -l {}'.format(dataPath)
        value = subprocess.getstatusoutput(runSh)
        lineNum = int(value[1].split(' ')[0])
        trainNum = int(lineNum*featureDic['splitPer'])
        #
        runSh = 'shuf -n{} {}>{}'.format(
                trainNum,
                dataPath,
                trainPath
                )
        os.system(runSh)
        #
        runSh = '''awk '{if(NR==FNR){a[$0]=1}else{if(!a[$0]){print $0}}}' %s %s>%s'''%(
                trainPath,
                dataPath,
                testPath
                )
        os.system(runSh)
