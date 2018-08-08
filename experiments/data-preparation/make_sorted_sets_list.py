import sys
import os
sys.path.append('/home/maciek/pyCharmProjects/mc-doi')
from datetime import datetime
import logging
from model.multi_contagion_models import MultiContagionLinearThresholdModel as MCDOI
from data.data import Data
from model.results import Results
import pickle
from tqdm import tqdm
import numpy as np
import pandas as pd
from data.data import Data
import operator

directory = '/nfs/maciej/mcdoi/louvain/'

d = dict()

for dataset in next(os.walk(directory))[1]:
    secik = directory + dataset
    edges = pd.read_csv(secik + '/edges', header=None)
    edges.columns = ['user1','user2']
    if sum(1 for line in open(secik + '/event_log', 'r', encoding='utf-8')) > 0:
        event_log = pd.read_csv(secik + '/event_log', header=None)
        event_log.columns = ['ts', 'user', 'contagion']
        set_id = secik.split('_')[-2] + '_' + secik.split('_')[-1]
        d[set_id] = len(event_log['contagion'].unique())
sorted_tuples = sorted(d.items(), key=operator.itemgetter(1))
for t in sorted_tuples:
    with open(directory+'sorted_sets_list', 'a+', encoding='utf-8') as handle:
        handle.write(directory +'louvain_' + t[0] + '\n')



