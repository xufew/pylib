# =======================
# -*- coding: utf-8 -*-
# author: LONGFEI XU
# Try your best
# ============
import pickle
import itertools
import json
import datetime

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import KFold, cross_val_score


class RF():
    def __init__(self, inputDic):
        self.param = self.set_param(inputDic)

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
        self.param = params
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
        return sum(value)/len(value)

    def line_search(
            self, trainX, trainY, searchDic={}, outPath='./tmp_search'
            ):
        '''
        进行最佳参数的搜索
        '''
        if len(searchDic) == 0:
            searchDic = {
                    'min_samples_split': list(range(2, 20, 1)),
                    'min_samples_leaf': list(range(2, 20, 1)),
                    'n_estimators': [200],
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
            print(useDic)
            self.param = self.set_param(useDic)
            result = self.cv(trainX, trainY)
            maxValue = result
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
