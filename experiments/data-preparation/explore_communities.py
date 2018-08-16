import sys
import os
sys.path.append('/home/maciek/pyCharmProjects/mc-doi')
from datetime import datetime
import logging
from model.multi_contagion_models import MultiContagionDynamicLinearThresholdModel as MCDOI
from data.data import Data
from model.results import Results
import pickle
from tqdm import tqdm
import numpy as np
import pandas as pd
from data.data import Data
from model.parameters import ContagionCorrelation, Adjacency
from copy import copy

directory = '/nfs/maciej/mcdoi/louvain/'

with open(directory+'louvain_communities.pickle', 'rb') as handle:
    communities = pickle.load(handle)

user = 38346

for id, community in enumerate(communities):
    if user in set(community):
        print(id)
