# =======================
# -*- coding: utf-8 -*-
# author: LONGFEI XU
# Try your best
# ============
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold


class Stacking():
    '''
    进行10折交叉验证，每次交叉验证将剩余的那层的结果进行输出
    最终生成和训练集等同的预测结果，测试集进行平均，用平均输出作为最终输出
    '''
    def __init__(self, trainX, trainY, testX, param={}):
        self.trainX = trainX
        self.trainY = trainY
        self.testX = testX
        self.param = self.set_param(param)

    def set_param(self, inputDic):
        '''
        涉及的变量,传入修改字典进行更改
        '''
        param = {
                'k_fold': 5,
                'label_name': 'label'
                }
        for paramName in inputDic:
            if paramName not in param:
                raise Exception('有不存在的变量')
        param.update(inputDic)
        return param

    def cross_stacking(self, modelList):
        '''
        丢入需要进行stacking的模型初始化方法，模型需要封装为train和predict
        '''
        kf = KFold(n_splits=self.param['k_fold'], shuffle=True)
        evalPredict = []
        testPredict = []
        for train_index, test_index in kf.split(self.trainX):
            trainX = self.trainX.iloc[train_index, :].copy()
            trainY = self.trainY.iloc[train_index].copy()
            evalX = self.trainX.iloc[test_index, :].copy()
            # 初始化模型
            initList = []
            for modelInit in modelList:
                initList.append(modelInit())
            # 模型训练
            trainList = []
            for model in initList:
                model.train(trainX.copy(), trainY.copy())
                trainList.append(model)
            # 结果预测
            evalList = []
            for model in trainList:
                evalList.append(model.predict(evalX.copy()))
            thisEval = pd.DataFrame(
                    np.array(evalList).T,
                    index=list(evalX.index)
                    )
            if len(evalPredict) == 0:
                evalPredict = thisEval
            else:
                evalPredict = pd.concat([evalPredict, thisEval])
            # 对测试集进行预测
            testList = []
            for model in trainList:
                testList.append(model.predict(self.testX.copy()))
            thisTest = pd.DataFrame(
                    np.array(testList).T,
                    index=list(self.testX.index)
                    )
            testPredict.append(thisTest)
        evalPredict = evalPredict.loc[self.trainX.index, :]
        evalPredict[self.param['label_name']] = self.trainY
        # 对测试集的结果取平均值
        testOutFrame = []
        count = 0
        for oneFrame in testPredict:
            count += 1
            if len(testOutFrame) == 0:
                testOutFrame = oneFrame
            else:
                testOutFrame += oneFrame
        testOutFrame = testOutFrame/float(count)
        return evalPredict, testOutFrame
