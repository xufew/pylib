# =======================
# -*- coding: utf-8 -*-
# author: LONGFEI XU
# Try your best
# ============
import pickle

import lightgbm as lgb
import pandas as pd


def set_param(inputDic):
    '''
    涉及的变量,传入修改字典进行更改
    '''
    params = {
            'learning_rate': 0.1,                   # 学习速率
            'num_leaves': 31,                       # 一棵树最多的叶子节点
            'num_trees': 200,                       # 多少颗树
            'min_sum_hessian_in_leaf': 0.001,       # minimal sum hessian in one leaf
            'min_data_in_leaf': 20,                 # minimal number of data in one leaf
            'feature_fraction': 1,                  # 按百分比选feature
            'bagging_fraction': 1,                  # 按百分比选数据
            'lambda_l1': 0,                         # l1 regularization
            'lambda_l2': 0,                         # l2 regularization
            'zero_as_missing': False,
            'boosting': 'gbdt',
            'application': 'binary',                # loss种类
            'metric': ['auc', 'binary_logloss'],
            'verbosity': 0,
            'num_threads': 4,
            'feval': None,                          # 定义评估函数
            'fobj': None,                           # 定义目标函数
            'device': 'cpu',
            }
    for paramName in inputDic:
        if paramName not in params:
            raise Exception('有不存在的变量')
    params.update(inputDic)
    print(params)
    return params


def cv(trainX, trainY, params, verbose_eval=True, nfold=5):
    '''
    进行交叉验证
    '''
    trainData = lgb.Dataset(
            trainX,
            label=trainY,
            free_raw_data=False
            )
    lgb.cv(
            params,
            trainData,
            verbose_eval=verbose_eval,
            nfold=nfold
            )


def train(trainX, trainY, params, modelSavePath, verbose_eval=True):
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
            params,
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


def predict(testX, gbmModel):
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