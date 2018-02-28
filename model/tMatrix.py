import copy
import numpy as np
import pandas as pd
import math

class tMatrix():

    def __init__(self):
        self.matrix=None
        self.numUsers=None

    def estimateVectorVolumeBatch(self,data,aMatrix,ccMAtrix,volume):
        #TODO Implement
        data.addContagionID()
        data.constructEventLogGrouped()
        indicators=[]
        I=np.full((data.numUsers,data.numContagions),False,dtype=bool)
        eventID=0
        while eventID<data.eventLog['eventID'].max():
            for index, row in data.eventLog[(data.eventLog['eventID'] >eventID) & (data.eventLog['eventID'] <= eventID+volume)].iterrows():
                I[row['user']][row['contagionID']] = True
            indicators.append(I)
            I = copy.deepcopy(I)
            eventID += volume
        Y = np.sum(indicators[0], axis=1)
        df_thresholds = []
        for l in range(len(indicators) - 1):
            U = aMatrix.matrix.transpose().dot(indicators[l])
            F = U.dot(ccMAtrix.matrix) / data.numContagions
            temp = np.logical_xor(indicators[l], indicators[l + 1])
            temp1 = np.logical_or(temp, indicators[l])
            activated = set()
            for i in range(data.numUsers):
                for j in range(data.numContagions):
                    if (temp[i][j] == True):
                        if (F[i][j] > 0):
                            df_thresholds.append([i, F[i][j], 1, Y[i]])
                        else:
                            df_thresholds.append([i, 0, 1, Y[i]])
                        activated.add(i)
                    if (temp1[i][j] == False):
                        df_thresholds.append([i, F[i][j], 0, Y[i]])
            for i in activated:
                Y[i] += 1
        df_thresholds = pd.DataFrame(df_thresholds)
        df_thresholds.columns = ['user', 'x', 'y', 'w']
        results=[]
        for user in np.arange(data.numUsers):
            res_max_neg = []
            res_min_pos = []
            X = df_thresholds[df_thresholds['user']==user]
            X = X[['x', 'y', 'w']]
            for i in X['w'].unique():
                if (i != X['w'].max()):
                    max_neg = X[(X['w'] == i) & (X['y'] == 0)]['x'].max()
                    min_pos = X[(X['w'] == i) & (X['y'] == 1)]['x'].min()
                    res_max_neg.append(1 - math.pow(1 - max_neg, 1 / float(i + 1)))
                    res_min_pos.append(1 - math.pow(1 - min_pos, 1 / float(i + 1)))
                else:
                    max_neg = X[(X['w'] == i) & (X['y'] == 0)]['x'].max()
                    res_max_neg.append(1 - math.pow(1 - max_neg, 1 / float(i + 1)))
            if (len(res_min_pos) != 0):
                result = [(max(res_max_neg) + min(res_min_pos)) / 2, max(res_max_neg), min(res_min_pos)]
            else:
                res_max_neg.append(0)
                result = [max(res_max_neg), max(res_max_neg), 1]
            results.append(result)
        self.matrix=np.repeat(np.asarray(results)[:, 0][np.newaxis].T, data.numContagions, axis=1)
        # review

    def estimateVectorTimeBatch(self,data):
        #TODO Implement
        pass

    def estimateVectorHybrideBatch(self,data):
        #TODO Implement
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
