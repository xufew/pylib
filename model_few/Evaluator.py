# =======================
# -*- coding: utf-8 -*-
# author: LONGFEI XU
# Try your best
# ============
import numpy as np
from sklearn.metrics import roc_auc_score
from sklearn import metrics


class Zero_One():
    '''
    0,1二分类问题时，使用的评估方法
    '''
    def __init__(self, trueY=[], predictY=[]):
        self.trueY = np.array(trueY)
        self.predictY = np.array(predictY)

    def __set_value(self, trueY, predictY):
        con1 = len(trueY)==0
        con2 = len(predictY)==0
        if not (con1 | con2):
            self.trueY = np.array(trueY)
            self.predictY = np.array(predictY)

    def cal_confusion_matrix(self, thres=0.5, trueY=[], predictY=[]):
        '''
        计算二类问题的混淆矩阵
        '''
        self.__set_value(trueY, predictY)
        self.predictY[self.predictY>=thres] = 1
        self.predictY[self.predictY<thres] = 0
        # 初始化混淆矩阵
        matrixDic = {
                'TP': 0,
                'FN': 0,
                'FP': 0,
                'TN': 0
                }
        # 计算混淆矩阵
        con1 = self.trueY==1
        con2 = self.trueY==0
        con3 = self.predictY==1
        con4 = self.predictY==0
        matrixDic['TP'] = (con1&con3).sum()
        matrixDic['FP'] = (con2&con3).sum()
        matrixDic['FN'] = (con1&con4).sum()
        matrixDic['TN'] = (con2&con4).sum()
        return matrixDic

    def roc_curve(self, trueY=[], predictY=[]):
        '''
        画roc曲线的时候需要的
        tpr为accuracy，准确率
        fpr为error，误分率
        '''
        self.__set_value(trueY, predictY)
        fpr, tpr, thresholds = metrics.roc_curve(
                self.trueY,
                self.predictY
                )
        return (fpr, tpr, thresholds)

    def ks_value(self, trueY=[], predictY=[]):
        '''
        金融里面ks值得计算
        '''
        self.__set_value(trueY, predictY)
        fpr, tpr, thresholds = metrics.roc_curve(
                self.trueY,
                self.predictY
                )
        ks = max(abs(fpr-tpr))
        return ks

    def roc_auc(self, trueY=[], predictY=[]):
        '''
        计算auc值
        '''
        self.__set_value(trueY, predictY)
        aucScore = roc_auc_score(
                self.trueY,
                self.predictY
                )
        return aucScore

    def four_eval(self, trueY=[], predictY=[], thres=0.5):
        '''
        计算四个值：准确率，误分率，召回率，精确度
        '''
        self.__set_value(trueY, predictY)
        confusionMatrix = self.cal_confusion_matrix(thres)
        # 准确率
        accuracy = (
                confusionMatrix['TP']+confusionMatrix['TN']
                )/float(
                        confusionMatrix['TP']+
                        confusionMatrix['FP']+
                        confusionMatrix['FN']+
                        confusionMatrix['TN']
                        )
        # 误分率
        error = (
                confusionMatrix['FP']+confusionMatrix['FN']
                )/float(
                        confusionMatrix['TP']+
                        confusionMatrix['FP']+
                        confusionMatrix['FN']+
                        confusionMatrix['TN']
                        )
        # 召回率
        con1 = confusionMatrix['TP']+confusionMatrix['FN']==0
        if not con1:
            recall = confusionMatrix['TP']/float(
                    confusionMatrix['TP']+confusionMatrix['FN']
                    )
        else:
            recall = -1
        # 精确度
        con1 = (confusionMatrix['TP']+confusionMatrix['FP'])==0
        if not con1:
            precision = confusionMatrix['TP']/float(
                    confusionMatrix['TP']+confusionMatrix['FP']
                    )
        else:
            precision = -1 # 输出
        outDic = {
                'accuracy': accuracy,
                'error': error,
                'recall': recall,
                'precision': precision
                }
        return outDic

    def precision_recall_list(self, trueY=[], predictY=[]):
        '''
        精确度,召回率,根据阈值的变化曲线
        '''
        self.__set_value(trueY, predictY)
        precisionList,recallList,thresList = metrics.precision_recall_curve(
                self.trueY,
                self.predictY
                )
        return (precisionList, recallList, thresList)

    def pr_auc(self, trueY=[], predictY=[]):
        '''
        精确和召回的线下面积
        纵坐标是精确度，横坐标是召回率
        '''
        self.__set_value(trueY, predictY)
        precisionList,recallList,_ = metrics.precision_recall_curve(
                self.trueY,
                self.predictY
                )
        auc = 0
        leftRange= 0
        total = len(precisionList)
        for i in range(total):
            auc += (recallList[total-i-1]-leftRange)*precisionList[total-i-1]
            leftRange = recallList[total-i-1]
        return auc
