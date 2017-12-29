# =======================
# -*- coding: utf-8 -*-
# author: LONGFEI XU
# Try your best
# ============
import pickle

from sklearn import metrics
import xgboost as xgb
import pandas as pd


def cross_validation(trainData, trainLabel, param):
    '''
    交叉验证
    '''
    # 转换为xgb形式
    dtrain = xgb.DMatrix(
            trainData,
            label=trainLabel,
            missing=param['naData'],
            )
    # 设置变量
    num_roud = param['num_roud']
    nfold = param['nfold']
    feval = param['feval']
    xgb.cv(
            param,
            dtrain,
            num_roud,
            nfold=nfold,
            verbose_eval=True,
            feval=feval
            )


def set_param(inputDic):
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
    print(param)
    return param


def ks_value(preds, dtrain):
    '''
    ks的评估方式
    '''
    train = dtrain.get_label()
    fpr, tpr, thresholds = metrics.roc_curve(train, preds)
    ks = max(tpr-fpr)
    return 'ksvalue', ks


def train(trainX, trainY, param, modelSavePath):
    '''
    对模型进行训练
    '''
    naData = param['naData']
    trainX = trainX.fillna(naData)
    # 转换为xgb形式
    dtrain = xgb.DMatrix(trainX, label=trainY, missing=naData)
    # 训练样本
    evallist = [(dtrain, 'train')]
    num_roud = param['num_roud']
    feval = param['feval']
    gbdtModel = xgb.train(
            param,
            dtrain,
            num_roud,
            evallist,
            feval=feval
            )
    featureImportance = _feature_importance(gbdtModel)
    print(featureImportance)
    # # 模型储存
    with open(modelSavePath, 'wb') as f:
        pickle.dump(gbdtModel, f)
    return gbdtModel


def _feature_importance(gbdtModel):
    '''
    获取gbdt训练的重要性
    '''
    featureDic = gbdtModel.get_fscore()
    featureSeries = pd.Series(featureDic)
    featureSeries.sort_values(ascending=False, inplace=True)
    return featureSeries


def predict(testX, gbdtModel, param):
    '''
    预测
    '''
    naData = param['naData']
    testX = testX.fillna(naData)
    testDMATRIX = xgb.DMatrix(testX, missing=naData)
    predictValue = gbdtModel.predict(testDMATRIX)
    return predictValue
