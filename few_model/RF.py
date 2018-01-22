# =======================
# -*- coding: utf-8 -*-
# author: LONGFEI XU
# Try your best
# ============
import pickle

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import KFold, cross_val_score


class RF():
    def __init__(self, inputDic):
        self.param = self.set_param(inputDic)
        print(self.param)
        self.model = RandomForestClassifier(
                n_estimators=self.param['n_estimators'],
                criterion=self.param['criterion'],
                max_features=self.param['max_features'],
                max_depth=self.param['max_depth'],
                min_samples_split=self.param['min_samples_split'],
                min_samples_leaf=self.param['min_samples_leaf'],
                min_weight_fraction_leaf=self.param[
                    'min_weight_fraction_leaf'
                    ],
                n_jobs=self.param['n_jobs'],
                verbose=self.param['verbose'],
                class_weight=self.param['class_weight'],
                )

    def set_param(self, inputDic):
        '''
        涉及的变量
        '''
        params = {
                'n_estimators': 10,
                'criterion': 'gini',
                'max_features': 'auto',
                'max_depth': None,
                'min_samples_split': 2,
                'min_samples_leaf': 1,
                'min_weight_fraction_leaf': 0,
                'n_jobs': 1,
                'n_jobs_cv': 1,
                'verbose': 1,
                'class_weight': 'balanced',
                'nfold': 5,
                'scoring': 'roc_auc',
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
                n_jobs=self.param['n_jobs_cv'],
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
        importance = self.model.feature_importances_[0]
        self.model.featureName = featureName
        featureIm = pd.Series(importance, self.model.featureName).sort_values(
                ascending=False
                )
        self.model.featureIm = featureIm
        with open(modelSavePath, 'wb') as fileWriter:
            pickle.dump(self.model, fileWriter)
        return self.model

    def predict(self, model, testX):
        '''
        预测
        '''
        predictValue = model.predict_proba(testX)[:, 1]
        return predictValue
