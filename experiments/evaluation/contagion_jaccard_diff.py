import pandas as pd
import os
import sys
sys.path.append('/home/maciek/pyCharmProjects/mc-doi')
from matplotlib import pyplot as plt
import csv
import numpy as np
import pickle
from data.data import Data
from collections import defaultdict
import functools
import itertools
import copy
from tqdm import tqdm

sets_to_evaluate_file = list(sys.argv)[1]
with open(sets_to_evaluate_file, 'r', encoding='utf-8') as sets_to_evaluate:
    sets_to_evaluate = sets_to_evaluate.readlines()
sets_to_evaluate = [x.strip() for x in sets_to_evaluate]

start_time = 1332565200
end_time = 1335416399
duration_24h_in_sec = 60*60*24
time_grid = np.arange(start_time+duration_24h_in_sec,end_time+duration_24h_in_sec,duration_24h_in_sec)

directory = '/nfs/maciej/mcdoi/louvain/'

evaluated = set()
for batch_size in [3600, 43200, 86400, 604800]:
    with open(directory + 'frequencies/contagion_jaccard_diff_'+str(batch_size), 'r', encoding='utf-8') as file:
        e = file.readlines()
    evaluated.update([x.strip() for x in e])

def diff(first, second):
    second = set(second)
    return [item for item in first if item not in second]


def evaluate(path, iter_length):
    history = int(path.split('/')[6].split('_')[1])
    batch_size = int(path.split('/')[8].split('_')[1])
    with open(os.path.dirname(os.path.dirname(os.path.dirname(path)))+'/edges', 'r', encoding='utf-8') as f:
        edges = pd.read_csv(f, header=None, names=[Data.user_1, Data.user_2])

    user_dict = defaultdict(functools.partial(next, itertools.count()))
    edges[Data.user_1] = edges[Data.user_1].map(user_dict)
    edges[Data.user_2] = edges[Data.user_2].map(user_dict)

    with open(os.path.dirname(os.path.dirname(os.path.dirname(path)))+'/event_log', 'r', encoding='utf-8') as f:
        whole_event_log = pd.read_csv(f, header=None, names=[Data.time_stamp, Data.user, Data.contagion])
    whole_event_log.user = whole_event_log.user.map(user_dict)

    with open(os.path.dirname(os.path.dirname(path))+'/data_obj.pickle', 'rb') as f:
        d=pickle.load(f)

    with open(os.path.dirname(os.path.dirname(path))+'/contagion_dict.pickle', 'rb') as f:
        contagion_dict=pickle.load(f)

    max_contagion_id = max(contagion_dict.values())

    rev_contagion_dict = {v:k for k,v in contagion_dict.items()}

    whole_event_log[Data.contagion_id] = whole_event_log[Data.contagion].apply(lambda x: contagion_dict[x])
    whole_event_log=whole_event_log[whole_event_log[Data.contagion_id]<=max_contagion_id]

    I_beginning = np.full((d.num_users, d.num_contagions), False, dtype=bool)
    for index, row in whole_event_log[whole_event_log[Data.time_stamp]<=time_grid[history-1]].iterrows():
        I_beginning[row[Data.user]][row[Data.contagion_id]] = True

    indicators = []
    I = np.full((d.num_users, d.num_contagions), False, dtype=bool)
    for i in range(1,min(7,33-history)+1):
        for index, row in whole_event_log[whole_event_log[Data.time_stamp]<=time_grid[history-1] + i * iter_length].iterrows():
            I[row[Data.user]][row[Data.contagion_id]] = True
        indicators.append(I)
        I = copy.deepcopy(I)

    results = []
    for i in range(0, 7):
        with open(path + '/result_' + str(i) + '.pickle', 'rb') as result:
            results.append(pickle.load(result))

    for i in range(1,min(7,33-history)+1):
        open(path + '/contagion_jaccard_diff_' + str(i - 1), 'w', encoding='utf-8').close()
        result_diff = np.logical_xor(results[i - 1], I_beginning)
        real_diff = np.logical_xor(indicators[i - 1], I_beginning)
        for contagion_id in range(d.num_contagions):
            set_from_prediction = np.where(result_diff[:,contagion_id])
            set_real = np.where(real_diff[:,contagion_id])
            intersection = np.intersect1d(set_from_prediction,set_real)
            union = np.union1d(set_from_prediction,set_real)
            if union.size==0:
                with open(path + '/contagion_jaccard_diff_' + str(i - 1), 'a', encoding='utf-8') as file:
                    file.write(rev_contagion_dict[contagion_id] + ',' + str(1) + '\n')
            else:
                with open(path + '/contagion_jaccard_diff_' + str(i - 1), 'a', encoding='utf-8') as file:
                    file.write(rev_contagion_dict[contagion_id] + ',' + str(intersection.size/union.size) + '\n')
        with open(directory + 'frequencies/contagion_jaccard_diff_' + str(batch_size), 'a', encoding='utf-8') as file:
            file.write(path + '/contagion_jaccard_diff_' + str(i - 1) + '\n')

if __name__ == '__main__':
    paths = diff(sets_to_evaluate,evaluated)
    for path in tqdm(paths):
        evaluate(path,86400)