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

directory = '/nfs/maciej/mcdoi/louvain/'

start_time = 1332565200
end_time = 1335416399
duration_24h_in_sec = 60*60*24
time_grid = np.arange(start_time+duration_24h_in_sec,end_time+duration_24h_in_sec,duration_24h_in_sec)

for dataset in tqdm(next(os.walk(directory))[1]):
    set = directory + dataset
    event_log = pd.read_csv(set + '/event_log', header = None)
    event_log.columns = ['ts', 'user', 'contagion']
    for counter, time_limit in tqdm(enumerate(time_grid, 1)):
        file_name = set + '/history_' + str(counter) + '/event_log'
        os.makedirs(os.path.dirname(file_name), exist_ok=True)
        with open(file_name, 'w', encoding='utf-8') as handle:
            event_log[event_log.ts <= time_limit].to_csv(handle, header=False, index=False)