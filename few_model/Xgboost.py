# =======================
# -*- coding: utf-8 -*-
# author: LONGFEI XU
# Try your best
# ============
import json
import pickle
import itertools
import datetime

import xgboost as xgb
import numpy as np
import pandas as pd


class Xgboost():
    def __init__(self, inputDic):
        self.param = self.set_param(inputDic)

    def set_param(self, inputDic):
        '''
        涉及的变量,传入修改字典进行更改
        '''
        param = {
                'max_depth': 6,
                'min_child_weight': 1,
                'col_sample_bytree': 1,
                'subsample': 1,
                'lambda': 1,
                'eta': 0.3,
                'scale_pos_weight': 1,
                'base_score': 0.5,
                'objective': 'binary:logistic',
                'eval_metric': 'auc',
                'reg_lambda': 1,
                'nthread': 4,
                'silent': 1,
                'num_roud': 500,
                'nfold': 10,
                'feval': None,
                'naData': -99999,
                }
        for paramName in inputDic:
            if paramName not in param:
                raise Exception('有不存在的变量')
        param.update(inputDic)
        return param

    def cv(self, trainData, trainLabel):
        '''
        交叉验证
        '''
        featureName = list(trainData.columns)
        trainData.columns = [str(i) for i in range(len(featureName))]
        trainData = trainData.fillna(
                self.param['naData']
                )
        # 转换为xgb形式
        dtrain = xgb.DMatrix(
                trainData,
                label=trainLabel,
                missing=self.param['naData'],
                )
        # 设置变量
        num_roud = self.param['num_roud']
        nfold = self.param['nfold']
        feval = self.param['feval']
        value = xgb.cv(
                self.param,
                dtrain,
                num_roud,
                nfold=nfold,
                verbose_eval=True,
                feval=feval
                )
        return value

    def line_search(
            self, trainX, trainY, searchDic={}, outPath='./tmp_search'
            ):
        '''
        进行最佳参数的搜索
        '''
        if len(searchDic) == 0:
            searchDic = {
                    'eta': np.linspace(0.001, 0.2, 20),
                    'max_depth': list(range(3, 10)),
                    'subsample': np.linspace(0.1, 1, 10),
                    'col_sample_bytree': np.linspace(0.1, 1, 10),
                    'min_child_weight': list(range(1, 50, 5)),
                    'reg_lambda': np.linspace(1, 50, 10),
                    'num_roud': [2000],
                    'objective': ['reg:linear', 'binary:logistic'],
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
            maxValue = result['test-auc-mean'].max()
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

    def train(self, trainX, trainY, modelSavePath='./tmp_model.pkl'):
        '''
        对模型进行训练
        '''
        featureName = list(trainX.columns)
        trainX.columns = [str(i) for i in range(len(featureName))]
        trainX = trainX.fillna(
                self.param['naData']
                )
        # 转换为xgb形式
        dtrain = xgb.DMatrix(
                trainX,
                label=trainY,
                missing=self.param['naData']
                )
        # 训练样本
        evallist = [(dtrain, 'train')]
        num_roud = self.param['num_roud']
        feval = self.param['feval']
        self.model = xgb.train(
                self.param,
                dtrain,
                num_roud,
                evallist,
                feval=feval
                )
        self.model.featureName = featureName
        featureImportance = self._feature_importance(self.model)
        print(featureImportance)
        self.model.featureIm = featureImportance
        # # 模型储存
        with open(modelSavePath, 'wb') as f:
            pickle.dump(self.model, f)
        return self.model

    def _feature_importance(self, gbdtModel):
        '''
        获取gbdt训练的重要性
        '''
        featureDic = gbdtModel.get_fscore()
        featureSeries = pd.Series(featureDic, self.model.featureName)
        featureSeries.sort_values(ascending=False, inplace=True)
        return featureSeries

    def predict(self, testX, gbdtModel=''):
        '''
        预测
        '''
        if gbdtModel == '':
            gbdtModel = self.model
        naData = self.param['naData']
        testX = testX.loc[:, gbdtModel.featureName]
        testX.columns = [str(i) for i in range(len(gbdtModel.featureName))]
        testX = testX.fillna(naData)
        testDMATRIX = xgb.DMatrix(testX, missing=naData)
        predictValue = gbdtModel.predict(testDMATRIX)
        return predictValue
