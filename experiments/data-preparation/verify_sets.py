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

for dataset in tqdm(next(os.walk(directory))[1]):
    secik = directory + dataset
    edges = pd.read_csv(secik + '/edges')
    edges.columns = ['user1','user2']
    if sum(1 for line in open(secik + '/event_log', 'r', encoding='utf-8')) > 0:
        event_log = pd.read_csv(secik + '/event_log')
        event_log.columns = ['ts', 'user', 'contagion']
        if not set(event_log['user']).issubset(edges['user1'].append(edges['user2'])):
            print(dataset)
    else:
        print('event_log empty in '+dataset)

