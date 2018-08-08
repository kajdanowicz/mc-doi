import pickle
import os
import sys
sys.path.append('/home/maciek/pyCharmProjects/mc-doi')
from model.results import Results
import pandas as pd
from data.data import Data
from model.parameters import ContagionCorrelation, Adjacency
from model.multi_contagion_models import MultiContagionLinearThresholdModel as MCDOI
import numpy as np

sets_to_compare_file = list(sys.argv)[1]
with open(sets_to_compare_file, 'r', encoding='utf-8') as sets_to_compare:
    sets_to_compare = sets_to_compare.readlines()
sets_to_compare = [x.strip() for x in sets_to_compare]

directory = '/nfs/maciej/mcdoi/louvain/'

def compare(path):
    pass