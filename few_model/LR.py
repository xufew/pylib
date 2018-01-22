# =======================
# -*- coding: utf-8 -*-
# author: LONGFEI XU
# Try your best
# ============
import pickle

import pandas as pd

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import KFold, cross_val_score


class LR():
    def __init__(self, inputDic):
        self.param = self.set_param(inputDic)
        self.model = LogisticRegression(
                penalty=self.param['penalty'],
                tol=self.param['tol'],
                C=self.param['C'],
                class_weight=self.param['class_weight'],
                solver=self.param['solver'],
                max_iter=self.param['max_iter'],
                multi_class=self.param['multi_class'],
                n_jobs=self.param['n_jobs'],
                )

    def set_param(self, inputDic):
        '''
        涉及的变量
        '''
        params = {
                'penalty': 'l2',
                'tol': 0.0001,
                'C': 1,
                'class_weight': 'balanced',
                'solver': 'liblinear',
                'max_iter': 100,
                'multi_class': 'ovr',
                'n_jobs': 1,
                'scoring': 'roc_auc',
                'verbose': 0,
                'nfold': 5,
                }
        for paramName in inputDic:
            if paramName not in params:
                raise Exception('有不存在的变量')
        params.update(inputDic)
        return params

    def cv(self, trainX, trainY):
        '''
        交叉验证
        '''
        k_fold = KFold(n_splits=self.param['nfold'])
        value = cross_val_score(
                self.model,
                trainX,
                trainY,
                cv=k_fold,
                n_jobs=self.param['n_jobs'],
                scoring=self.param['scoring']
                )
        print(value)
        print(sum(value)/len(value))

    def train(self, trainX, trainY, modelSavePath='./tmp_model.pkl'):
        '''
        训练
        '''
        featureName = list(trainX.columns)
        trainX.columns = [str(i) for i in range(len(featureName))]
        self.model.fit(trainX, trainY)
        importance = self.model.coef_[0]
        self.model.featureName = featureName
        featureIm = pd.Series(importance, self.model.featureName).sort_values(
                ascending=False
                )
        self.model.featureIm = featureIm
        print(featureIm)
        with open(modelSavePath, 'wb') as fileWriter:
            pickle.dump(self.model, fileWriter)
        return self.model

    def predict(self, model, testX):
        '''
        预测
        '''
        predictValue = model.predict_proba(testX)[:, 1]
        return predictValue
