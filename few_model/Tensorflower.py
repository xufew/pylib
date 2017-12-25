# =======================
# -*- coding: utf-8 -*-
# author: LONGFEI XU
# Try your best
# ============
import tensorflow as tf


class ModelReader():
    '''
    进行建立网络之后，所存储的网络的读取
    '''
    def __init__(self, metaPath, modelPath):
        self.graph=tf.Graph()
        with self.graph.as_default():
            saver = tf.train.import_meta_graph(metaPath)
        self.sess = tf.Session(graph=self.graph)
        with self.sess.as_default():
            with self.graph.as_default():
                saver.restore(self.sess, modelPath)

    def return_tensor(self, tensorName):
        '''
        返回run里面所调用的tensor
        '''
        return self.graph.get_tensor_by_name(
                '{}:0'.format(tensorName)
                )

    def predict(self, tensorName, feedDic):
        '''
        根据tensor和feed进行结果的获取
        '''
        feedTensor = {}
        for thisTensorName in feedDic:
            feedTensor[self.return_tensor(thisTensorName)] = feedDic[thisTensorName]
        result = self.sess.run(
                self.return_tensor(tensorName),
                feed_dict=feedTensor
                )
        return result
