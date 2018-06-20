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

sets_to_evaluate_file = list(sys.argv)[1]
with open(sets_to_evaluate_file, 'r', encoding='utf-8') as sets_to_evaluate:
    sets_to_evaluate = sets_to_evaluate.readlines()
sets_to_evaluate = [x.strip() for x in sets_to_evaluate]

directory = '/datasets/mcdoi/louvain/'

with open(directory + 'frequencies_evaluated', 'r', encoding='utf-8') as evaluated:
    evaluated = evaluated.readlines()
evaluated = set([x.strip() for x in evaluated])

def diff(first, second):
    second = set(second)
    return [item for item in first if item not in second]

from joblib import Parallel, delayed
import time
def text_progessbar(seq, total=None):
    step = 1
    tick = time.time()
    while True:
        time_diff = time.time()-tick
        avg_speed = time_diff/step
        total_str = 'of %n' % total if total else ''
        print('step', step, '%.2f' % time_diff, 'avg: %.2f iter/sec' % avg_speed, total_str)
        step += 1
        yield next(seq)
all_bar_funcs = {
    'txt': lambda args: lambda x: text_progessbar(x, **args),
    'None': lambda args: iter,
}
def ParallelExecutor(use_bar='tqdm', **joblib_args):
    def aprun(bar=use_bar, **tq_args):
        def tmp(op_iter):
            if str(bar) in all_bar_funcs.keys():
                bar_func = all_bar_funcs[str(bar)](tq_args)
            else:
                raise ValueError("Value %s not supported as bar type"%bar)
            return Parallel(**joblib_args)(bar_func(op_iter))
        return tmp
    return aprun

aprun = ParallelExecutor(n_jobs=1)

def directories_to_evaluate(directory):
    paths = []
    for dat in next(os.walk(directory))[1]:
        for history_length in np.arange(1, 31, 1):
            if 'time' in next(os.walk(directory+dat+'/history_'+str(history_length)))[1]:
                for batch_size in next(os.walk(directory+dat+'/history_'+str(history_length)+'/time'))[1]:
                    paths.append(directory+dat+'/history_'+str(history_length)+'/time/'+batch_size)
    return paths

def evaluate(path):
    print(path)
    batch_size = int(path.split('/')[7].split('_')[1])
    history = int(path.split('/')[5].split('_')[1])
    edges = pd.read_csv(os.path.dirname(os.path.dirname(os.path.dirname(path))) + '/edges',header=None)
    event_log = pd.read_csv(os.path.dirname(os.path.dirname(path)) + '/event_log',header=None)
    d = Data()
    d.load_data_data_frame(event_log, edges)
    results = []
    for i in range(0,3):
        with open(path+'/result_'+str(i)+'.pickle', 'rb') as result:
            results.append(pickle.load(result))
    print(d.contagion_id_dict)


if __name__ == '__main__':
    paths = diff(sets_to_evaluate,evaluated)
    for path in paths[:1]:
        evaluate(path)