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

path = '/nfs/maciej/mcdoi/louvain/louvain_55_123/history_20/time/size_604800'
history=20

start_time = 1332565200
end_time = 1335416399
duration_24h_in_sec = 60*60*24
time_grid = np.arange(start_time+duration_24h_in_sec,end_time+duration_24h_in_sec,duration_24h_in_sec)
iter_length = 86400

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

# print(d.event_log.head())

results = []
for i in range(0, 7):
    with open(path + '/result_' + str(i) + '.pickle', 'rb') as result:
        results.append(pickle.load(result))

for i in range(1,min(7,33-history)+1):
    open(path + '/contagion_fractions_' + str(i - 1), 'w', encoding='utf-8').close()
    e = whole_event_log[whole_event_log['ts'] <= time_grid[history - 1] + i * iter_length].drop_duplicates(subset=['contagion', 'user'], keep='first')
    e = e.groupby(by=['user']).count()['ts']
    for user in range(d.num_users):
        with open(path + '/contagion_fractions_' + str(i - 1), 'a', encoding='utf-8') as file:
            file.write(str(user) + ',' + str(e.get(user,0) / d.num_contagions) + ',' + str(np.sum(results[i - 1], axis=1)[user] / d.num_contagions) + '\n')


