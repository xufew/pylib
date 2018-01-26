# =======================
# -*- coding: utf-8 -*-
# author: LONGFEI XU
# Try your best
# ============
import pickle
import itertools
import json
import datetime

import lightgbm as lgb
import numpy as np
import pandas as pd


class Lightgbm():
    def __init__(self, inputDic):
        self.param = self.set_param(inputDic)

    def set_param(self, inputDic):
        '''
        涉及的变量,传入修改字典进行更改
        '''
        param = {
                'learning_rate': 0.1,                   # 学习速率
                'num_leaves': 31,                       # 一棵树最多的叶子节点
                'num_trees': 200,                       # 多少颗树
                'min_sum_hessian_in_leaf': 0.001,       # minimal sum hessian in one leaf
                'min_data_in_leaf': 20,                 # minimal number of data in one leaf
                'feature_fraction': 1,                  # 按百分比选feature
                'bagging_fraction': 1,                  # 按百分比选数据
                'bagging_freq': 0,                      # 多少次迭代进行一次bagging
                'lambda_l1': 0,                         # l1 regularization
                'lambda_l2': 0,                         # l2 regularization
                'scale_pos_weight': 1,                  # 1所占的权重,2分类问题中
                'zero_as_missing': False,
                'boosting': 'gbdt',
                'application': 'binary',                # loss种类
                'metric': ['auc', 'binary_logloss'],
                'verbosity': 0,
                'num_threads': 4,
                # 'feval': None,                          # 定义评估函数
                # 'fobj': None,                           # 定义目标函数
                'device': 'cpu',
                }
        for paramName in inputDic:
            if paramName not in param:
                raise Exception('有不存在的变量')
        param.update(inputDic)
        return param

    def cv(self, trainX, trainY, verbose_eval=True, nfold=5):
        '''
        进行交叉验证
        '''
        trainData = lgb.Dataset(
                trainX,
                label=trainY,
                free_raw_data=False
                )
        evalDic = lgb.cv(
                self.param,
                trainData,
                verbose_eval=verbose_eval,
                nfold=nfold
                )
        return evalDic

    def line_search(
            self, trainX, trainY, searchDic={}, outPath='./tmp_search'
            ):
        '''
        进行最佳参数的搜索
        '''
        if len(searchDic) == 0:
            searchDic = {
                    'learning_rate': np.linspace(0.001, 0.2, 20),
                    'num_leaves': list(range(30, 150, 10)),
                    'bagging_fraction': np.linspace(0.1, 1, 10),
                    'feature_fraction': np.linspace(0.1, 1, 10),
                    'lambda_l2': np.linspace(1, 50, 10),
                    'min_data_in_leaf': list(range(50, 200, 20)),
                    'min_sum_hessian_in_leaf': np.linspace(0.001, 0.2, 20),
                    'num_trees': [2000],
                    'application': ['binary', 'regression'],
                    }
        searchNameList = list(searchDic.keys())
        searchValueList = []
        for featureName in searchNameList:
            searchValueList.append(searchDic[featureName])
        searchList = list(itertools.product(*tuple(searchValueList)))
        print(len(searchList))
        #
        for valueSet in searchList:
            useDic = {}
            for i, value in enumerate(valueSet):
                useDic[searchNameList[i]] = value
            self.param = self.set_param(useDic)
            result = self.cv(trainX, trainY)
            maxValue = max(result['auc-mean'])
            with open(outPath, 'a') as fileWriter:
                fileWriter.write(
                        '{}\t{}\t{}\n'.format(
                            datetime.datetime.now().strftime(
                                '%Y.%m.%d-%H:%M:%S'
                                ),
                            json.dumps(useDic),
                            maxValue
                            )
                        )

    def train(
            self,
            trainX,
            trainY,
            modelSavePath='./tmp_model.pkl',
            verbose_eval=True
            ):
        '''
        训练
        '''
        featureName = list(trainX.columns)
        trainX.columns = [str(i) for i in range(len(featureName))]
        trainData = lgb.Dataset(
                trainX,
                label=trainY,
                free_raw_data=False
                )
        lgbModel = lgb.train(
                self.param,
                trainData,
                valid_sets=trainData,
                verbose_eval=verbose_eval,
                keep_training_booster=False
                )
        importance = lgbModel.feature_importance()
        lgbModel.featureName = featureName
        featureIm = pd.Series(importance, lgbModel.featureName).sort_values(
                ascending=False
                )
        lgbModel.featureIm = featureIm
        print(featureIm)
        with open(modelSavePath, 'wb') as fileWriter:
            pickle.dump(lgbModel, fileWriter)
        return lgbModel

    def predict(self, testX, gbmModel):
        '''
        预测
        '''
        featureName = gbmModel.featureName
        testX = testX.loc[:, featureName]
        if featureName != list(testX.columns):
            raise Exception('wrong feature')
        testX.columns = [str(i) for i in range(len(featureName))]
        predictValue = gbmModel.predict(testX)
        return predictValue
