# =======================
# -*- coding: utf-8 -*-
# author: LONGFEI XU
# Try your best
# ============
import pickle

import xgboost as xgb
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
                'objective': 'binary:logistic',
                'eval_metric': 'auc',
                'nthread': 4,
                'silent': 1,
                'num_roud': 500,
                'nfold': 10,
                'feval': None,
                'naData': -999,
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
        xgb.cv(
                self.param,
                dtrain,
                num_roud,
                nfold=nfold,
                verbose_eval=True,
                feval=feval
                )

    def train(self, trainX, trainY, modelSavePath='./tmp_model.pkl'):
        '''
        对模型进行训练
        '''
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
        featureImportance = self._feature_importance(self.model)
        print(featureImportance)
        # # 模型储存
        with open(modelSavePath, 'wb') as f:
            pickle.dump(self.model, f)
        return self.model

    def _feature_importance(self, gbdtModel):
        '''
        获取gbdt训练的重要性
        '''
        featureDic = gbdtModel.get_fscore()
        featureSeries = pd.Series(featureDic)
        featureSeries.sort_values(ascending=False, inplace=True)
        return featureSeries

    def predict(self, testX, gbdtModel):
        '''
        预测
        '''
        naData = self.param['naData']
        testX = testX.fillna(naData)
        testDMATRIX = xgb.DMatrix(testX, missing=naData)
        predictValue = gbdtModel.predict(testDMATRIX)
        return predictValue
