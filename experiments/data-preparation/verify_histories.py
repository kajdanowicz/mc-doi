import sys
import os
sys.path.append('/home/maciek/pyCharmProjects/mc-doi')
from datetime import datetime
import logging
from model.multi_contagion_models import MultiContagionDynamicThresholdModel as MCDOI
from data.data import Data
from model.results import Results
import pickle
from tqdm import tqdm
import numpy as np
import pandas as pd
from data.data import Data

directory = '/datasets/mcdoi/louvain/'

start_time = 1332565200
end_time = 1335416399
duration_24h_in_sec = 60*60*24
time_grid = np.arange(start_time+duration_24h_in_sec,end_time+duration_24h_in_sec,duration_24h_in_sec)

for dataset in tqdm(next(os.walk(directory))[1]):
    secik = directory + dataset
    edges = pd.read_csv(secik + '/edges')
    edges.columns = ['user1','user2']
    for history in range(1,31):
        file_name = secik + '/history_' + str(history)
        if sum(1 for line in open(file_name + '/event_log', 'r', encoding='utf-8')) > 0:
            event_log = pd.read_csv(file_name + '/event_log')
            event_log.columns = ['ts', 'user', 'contagion']
            if not set(event_log['user']).issubset(edges['user1'].append(edges['user2'])):
                print(file_name)
