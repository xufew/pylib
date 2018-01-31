# =======================
# -*- coding: utf-8 -*-
# author: LONGFEI XU
# Try your best
# ============


class PathAnalysis:
    def __init__(self):
        pass

    def count_directed_frequence(
            self, inputArray, smallestNum=3, biggestNum=6
            ):
        outDic = {}
        count = 0
        for oneArray in inputArray:
            count += 1
            if count == 1000:
                break
            arrayLen = len(oneArray)
            if arrayLen < smallestNum:
                # 跳过不需要统计的最小范畴
                continue
            for i in range(arrayLen):
                if arrayLen-i < smallestNum:
                    break
                # 获取最小path
                smallPath = str(oneArray[i])
                for j in range(i+1, i+smallestNum):
                    smallPath += '->{}'.format(oneArray[j])
                if smallPath not in outDic:
                    outDic[smallPath] = 0
                outDic[smallPath] += 1
                # 慢慢增长到最大num
                if i+biggestNum > arrayLen:
                    toNum = arrayLen
                else:
                    toNum = i+biggestNum
                for j in range(i+smallestNum, toNum):
                    smallPath += '->{}'.format(oneArray[j])
                    if smallPath not in outDic:
                        outDic[smallPath] = 0
                    outDic[smallPath] += 1
        return outDic
