import copy
import numpy as np
import pandas as pd
import math
from tqdm import trange, tqdm
tqdm.pandas(desc="Progress apply")
from collections import defaultdict
import copy
from data.data import Data

import sys


class Threshold():

    def __init__(self):
        self.matrix = None
        self.initial_matrix = None
        self.num_users = None

    def estimate_volume_batch(self, data, a_matrix, cc_matrix, volume):
        data.add_contagion_id()
        data.construct_event_log_grouped()
        indicators = []
        I = np.full((data.num_users, data.num_contagions), False, dtype=bool)
        event_id = 0
        while event_id < data.event_log[Data.event_id].max():
            for index, row in data.event_log[(data.event_log[Data.event_id] > event_id) & (data.event_log[Data.event_id] <= event_id + volume)].iterrows():
                I[row[Data.user]][row[Data.contagion_id]] = True
            indicators.append(I)
            I = copy.deepcopy(I)
            event_id += volume
        Y = np.sum(indicators[0], axis=1)
        self._estimate(Y, a_matrix, cc_matrix, data, indicators)

    def _estimate(self, Y, a_matrix, cc_matrix, data, indicators):
        a_matrix.transpose()
        # print('Adjacency.matrix_transposed_.shape', Adjacency.matrix_transposed_.shape)
        # print('indicators[0].shape', indicators[0].shape)
        max_neg = defaultdict(lambda : -2)
        min_pos = defaultdict(lambda : 2)
        for l in range(len(indicators) - 1):
            U = a_matrix.matrix_transposed_.dot(indicators[l])
            F = U.dot(cc_matrix.matrix) / data.num_contagions
            temp = np.logical_xor(indicators[l], indicators[l + 1])  # aktywowane z l na l+1
            temp1 = np.logical_or(temp, indicators[l])  # nieaktywowane z l na l+1 z wylaczeniem wczesniej aktywnych (po nalozeniu nagacji)
            activated = set()
            for i in range(data.num_users):
                for j in range(data.num_contagions):
                    if temp[i][j]:
                        if F[i][j] > 0:
                            min_pos[i] = min(min_pos[i], 1 - math.pow(1 - F[i][j], 1 / float(Y[i] + 1)))
                        else:
                            min_pos[i] = min(min_pos[i], 0)  # czy chcemy wyeliminować aktywacje, przy ujemnym wpływie?
                        activated.add(i)
                    if not temp1[i][j]:
                        max_neg[i] = max(max_neg[i], 1 - math.pow(1 - F[i][j], 1 / float(Y[i] + 1)))
            for i in activated:
                Y[i] += 1
        results = []
        for user in range(data.num_users):
            if min_pos[user] > 1:
                results.append(max(max_neg[user], 0))
            else:
                results.append(max((max_neg[user] + min_pos[user]) / 2, 0))
        # print(Results)
        # print([(i,e) for i, e in enumerate(Results) if e != 0])
        self.matrix = np.repeat(np.asarray(results)[np.newaxis].T, data.num_contagions, axis=1)
        self.initial_matrix = copy.copy(self.matrix)
        # review

    def estimate_time_batch(self, data, a_matrix, cc_matrix, volume):
        data.add_contagion_id()
        data.construct_event_log_grouped()
        indicators = []
        I = np.full((data.num_users, data.num_contagions), False, dtype=bool)
        ts = 0
        while ts < data.event_log[Data.time_stamp].max():
            for index, row in data.event_log[(data.event_log[Data.time_stamp] > ts) & (data.event_log[Data.time_stamp] <= ts + volume)].iterrows():
                I[row[Data.user]][row[Data.contagion_id]] = True
            indicators.append(I)
            I = copy.deepcopy(I)
            ts += volume
        Y = np.sum(indicators[0], axis=1)
        self._estimate(Y, a_matrix, cc_matrix, data, indicators)

    def estimate_hybride_batch(self, data):
        # TODO Implement
        pass

    # def estimateVector(self,Data):
    #     #TODO Implement
    #     indykatory_est = []
    #     I = np.full((Data.num_users_, Data.num_contagions), False, dtype=bool)
    #     for i in range(history):
    #         for index, row in event_log[event_log['ts'] == i].iterrows():
    #             I[row['userNEW'], row['tagID']] = True
    #         indykatory_est.append(I)
    #         I = copy.deepcopy(I)
    #
    # def _estimate(self,Data):
    #     #TODO Implement
    #     # Construct matrix from vector
