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

evaluated = set()
for batch_size in [3600, 43200, 86400, 604800]:
    with open(directory + 'frequencies/frequencies_'+str(batch_size), 'r', encoding='utf-8') as file:
        e = file.readlines()
    evaluated.update([x.strip() for x in e])

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

# def directories_to_evaluate(directory):
#     paths = []
#     for dat in next(os.walk(directory))[1]:
#         for history_length in np.arange(1, 31, 1):
#             if 'time' in next(os.walk(directory+dat+'/history_'+str(history_length)))[1]:
#                 for batch_size in next(os.walk(directory+dat+'/history_'+str(history_length)+'/time'))[1]:
#                     paths.append(directory+dat+'/history_'+str(history_length)+'/time/'+batch_size)
#     return paths

start_time = 1332565200
end_time = 1335416399
duration_24h_in_sec = 60*60*24
time_grid = np.arange(start_time+duration_24h_in_sec,end_time+duration_24h_in_sec,duration_24h_in_sec)

def evaluate(path):
    batch_size = int(path.split('/')[7].split('_')[1])
    history = int(path.split('/')[5].split('_')[1])
    edges = pd.read_csv(os.path.dirname(os.path.dirname(os.path.dirname(path))) + '/edges',header=None)
    event_log = pd.read_csv(os.path.dirname(os.path.dirname(path)) + '/event_log',header=None)
    event_log.columns = ['ts', 'user', 'contagion']
    with open(os.path.dirname(os.path.dirname(path)) + '/contagion_dict' + '.pickle', 'rb') as file:
        dict = pickle.load(file)
    # inv_dict = {v: k for k, v in dict.items()}
    results = []
    for i in range(0,3):
        with open(path+'/result_'+str(i)+'.pickle', 'rb') as result:
            results.append(pickle.load(result))
    if batch_size==3600:
        if history <= 32:
            for i in range(1,4):
                open(path+'/frequencies_'+str(i-1), 'w', encoding='utf-8').close()
                e = event_log[event_log['ts'] <= time_grid[history-1]+i*batch_size].drop_duplicates(subset=['contagion', 'user'], keep='first')
                e = e.groupby(by=['contagion']).count()['ts']
                for key, value in dict.items():
                    with open(path+'/frequencies_'+str(i-1), 'a', encoding='utf-8') as file:
                        file.write(key + ',' + str(e.loc[key]/results[0].shape[0]) + ',' + str(np.sum(results[i-1], axis=0)[value]/results[0].shape[0]) + '\n')
            with open(directory+ 'frequencies/frequencies_'+str(batch_size), 'a', encoding='utf-8') as file:
                file.write(path + '\n')
    elif batch_size==43200:
        if history <= 31:
            for i in range(1,4):
                open(path+'/frequencies_'+str(i-1), 'w', encoding='utf-8').close()
                e = event_log[event_log['ts'] <= time_grid[history-1]+i*batch_size].drop_duplicates(subset=['contagion', 'user'], keep='first')
                e = e.groupby(by=['contagion']).count()['ts']
                for key, value in dict.items():
                    with open(path+'/frequencies_'+str(i-1), 'a', encoding='utf-8') as file:
                        file.write(key + ',' + str(e.loc[key]/results[0].shape[0]) + ',' + str(np.sum(results[i-1], axis=0)[value]/results[0].shape[0]) + '\n')
            with open(directory + 'frequencies/frequencies_' + str(batch_size), 'a', encoding='utf-8') as file:
                file.write(path + '\n')
    elif batch_size==86400:
        if history <= 30:
            for i in range(1,4):
                open(path+'/frequencies_'+str(i-1), 'w', encoding='utf-8').close()
                e = event_log[event_log['ts'] <= time_grid[history-1]+i*batch_size].drop_duplicates(subset=['contagion', 'user'], keep='first')
                e = e.groupby(by=['contagion']).count()['ts']
                for key, value in dict.items():
                    with open(path+'/frequencies_'+str(i-1), 'a', encoding='utf-8') as file:
                        file.write(key + ',' + str(e.loc[key]/results[0].shape[0]) + ',' + str(np.sum(results[i-1], axis=0)[value]/results[0].shape[0]) + '\n')
            with open(directory+ 'frequencies/frequencies_'+str(batch_size), 'a', encoding='utf-8') as file:
                file.write(path + '\n')
    elif batch_size==604800:
        if history <= 12:
            for i in range(1,4):
                open(path+'/frequencies_'+str(i-1), 'w', encoding='utf-8').close()
                e = event_log[event_log['ts'] <= time_grid[history-1]+i*batch_size].drop_duplicates(subset=['contagion', 'user'], keep='first')
                e = e.groupby(by=['contagion']).count()['ts']
                for key, value in dict.items():
                    with open(path+'/frequencies_'+str(i-1), 'a', encoding='utf-8') as file:
                        file.write(key + ',' + str(e.loc[key]/results[0].shape[0]) + ',' + str(np.sum(results[i-1], axis=0)[value]/results[0].shape[0]) + '\n')
            with open(directory+ 'frequencies/frequencies_'+str(batch_size), 'a', encoding='utf-8') as file:
                file.write(path + '\n')
        elif 12 < history <= 19:
            for i in range(1, 3):
                open(path + '/frequencies_' + str(i - 1), 'w', encoding='utf-8').close()
                e = event_log[event_log['ts'] <= time_grid[history - 1] + i * batch_size].drop_duplicates(
                    subset=['contagion', 'user'], keep='first')
                e = e.groupby(by=['contagion']).count()['ts']
                for key, value in dict.items():
                    with open(path + '/frequencies_' + str(i - 1), 'a', encoding='utf-8') as file:
                        file.write(key + ',' + str(e.loc[key] / results[0].shape[0]) + ',' + str(
                            np.sum(results[i - 1], axis=0)[value] / results[0].shape[0]) + '\n')
            with open(directory+ 'frequencies/frequencies_'+str(batch_size), 'a', encoding='utf-8') as file:
                file.write(path + '\n')
        elif 19 < history <= 26:
            for i in range(1, 2):
                open(path + '/frequencies_' + str(i - 1), 'w', encoding='utf-8').close()
                e = event_log[event_log['ts'] <= time_grid[history - 1] + i * batch_size].drop_duplicates(
                    subset=['contagion', 'user'], keep='first')
                e = e.groupby(by=['contagion']).count()['ts']
                for key, value in dict.items():
                    with open(path + '/frequencies_' + str(i - 1), 'a', encoding='utf-8') as file:
                        file.write(key + ',' + str(e.loc[key] / results[0].shape[0]) + ',' + str(
                            np.sum(results[i - 1], axis=0)[value] / results[0].shape[0]) + '\n')
            with open(directory+ 'frequencies/frequencies_'+str(batch_size), 'a', encoding='utf-8') as file:
                file.write(path + '\n')
    else:
        pass


if __name__ == '__main__':
    paths = diff(sets_to_evaluate,evaluated)
    for path in tqdm(paths):
        evaluate(path)