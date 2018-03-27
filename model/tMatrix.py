import copy
import numpy as np
import pandas as pd
import math
from tqdm import trange, tqdm
tqdm.pandas(desc="Progress apply")
from collections import defaultdict

import sys


class tMatrix():

    def __init__(self):
        self.matrix = None
        self.numUsers = None

    def estimateVolumeBatch(self, data, aMatrix, ccMAtrix, volume):
        # TODO Implement
        data.addContagionID()
        data.constructEventLogGrouped()
        indicators = []
        I = np.full((data.numUsers, data.numContagions), False, dtype=bool)
        eventID = 0
        while eventID < data.eventLog['eventID'].max():
            for index, row in data.eventLog[(data.eventLog['eventID'] > eventID) & (data.eventLog['eventID'] <= eventID + volume)].iterrows():
                I[row['user']][row['contagionID']] = True
            indicators.append(I)
            I = copy.deepcopy(I)
            eventID += volume
        Y = np.sum(indicators[0], axis=1)
        df_thresholds = []
        aMatrix.transpose()
        # print('aMatrix.matrixTransposed.shape', aMatrix.matrixTransposed.shape)
        # print('indicators[0].shape', indicators[0].shape)
        max_neg = defaultdict(lambda : -2)
        min_pos = defaultdict(lambda : 2)
        for l in range(len(indicators) - 1):
            U = aMatrix.matrixTransposed.dot(indicators[l])
            F = U.dot(ccMAtrix.matrix) / data.numContagions
            temp = np.logical_xor(indicators[l], indicators[l + 1]) #aktywowane z l na l+1
            temp1 = np.logical_or(temp, indicators[l]) #nieaktywowane z l na l+1 z wylaczeniem wczesniej aktywnych (po nalozeniu nagacji)
            activated = set()
            for i in range(data.numUsers):
                for j in range(data.numContagions):
                    if temp[i][j]:
                        if F[i][j] > 0:
                            min_pos[i] = min(min_pos[i], 1 - math.pow(1 - F[i][j], 1 / float(Y[i] + 1)))
                        else:
                            min_pos[i] = min(min_pos[i], 0) #czy chcemy wyeliminować aktywacje, przy ujemnym wpływie?
                        activated.add(i)
                    if not temp1[i][j]:
                        max_neg[i] = max(max_neg[i], 1 - math.pow(1 - F[i][j], 1 / float(Y[i] + 1)))
            for i in activated:
                Y[i] += 1
        results = []
        for user in range(data.numUsers):
            if min_pos[user] > 1:
                results.append(max(max_neg[user],0))
            else:
                results.append(max((max_neg[user] + min_pos[user]) / 2, 0))
        # print(results)
        # print([(i,e) for i, e in enumerate(results) if e != 0])
        self.matrix = np.repeat(np.asarray(results)[np.newaxis].T, data.numContagions, axis=1)
        # review

    def estimateTimeBatch(self, data):
        # TODO Implement
        pass

    def estimateHybrideBatch(self, data):
        # TODO Implement
        pass

    # def estimateVector(self,data):
    #     #TODO Implement
    #     indykatory_est = []
    #     I = np.full((data.numUsers, data.numContagions), False, dtype=bool)
    #     for i in range(history):
    #         for index, row in event_log[event_log['ts'] == i].iterrows():
    #             I[row['userNEW'], row['tagID']] = True
    #         indykatory_est.append(I)
    #         I = copy.deepcopy(I)
    #
    # def estimate(self,data):
    #     #TODO Implement
    #     # Construct matrix from vector
