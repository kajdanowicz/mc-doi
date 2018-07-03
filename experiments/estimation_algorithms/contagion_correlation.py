import pickle
import os
import sys
sys.path.append('/home/maciek/pyCharmProjects/mc-doi')
from model.results import Results
import pandas as pd
from data.data import Data
from model.parameters import ContagionCorrelation, Adjacency
from model.multi_contagion_models import MultiContagionDynamicThresholdModel as MCDOI
import numpy as np

sets_to_compare_file = list(sys.argv)[1]
with open(sets_to_compare_file, 'r', encoding='utf-8') as sets_to_compare:
    sets_to_compare = sets_to_compare.readlines()
sets_to_compare = [x.strip() for x in sets_to_compare]

directory = '/nfs/maciej/mcdoi/louvain/'


def compare(path):
    with open(path + '/data_obj.pickle', 'rb') as f:
        d = pickle.load(f)
    cc = ContagionCorrelation()
    cc.estimate(d)
    with open(path + '/contagion.pickle', 'rb') as f:
        cc_present = pickle.load(f)
    if not np.all(cc.matrix==cc_present):
        print(path)
        print(cc.matrix[cc.matrix!=cc_present])
        print(cc_present[cc.matrix!=cc_present])

if __name__ == '__main__':
    for path in sets_to_compare:
        compare(path)