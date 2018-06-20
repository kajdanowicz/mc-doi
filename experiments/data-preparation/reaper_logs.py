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
from model.parameters import ContagionCorrelation, Adjacency
from copy import copy

file_to_repare = list(sys.argv)[1]

with open(file_to_repare, 'r', encoding='utf-8') as file:
    records = file.readlines()
records = [x.strip() for x in records]

open(file_to_repare, 'w').close()
for record in records:
    with open(file_to_repare+'_test', 'a', encoding='utf-8') as file:
        file.write(record.split('//')[0]+'/'+record.split('//')[1] + '\n')
