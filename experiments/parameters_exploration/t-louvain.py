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

aprun = ParallelExecutor(n_jobs=20)


directory = '/datasets/mcdoi/louvain/'

batch_sizes = [3600, 43200, 86400, 604800]
batch_sizes.reverse()

with open(directory + 'sets_to_omit', 'r', encoding='utf-8') as sets_to_omit:
    sets_to_omit = sets_to_omit.readlines()
sets_to_omit = set([x.strip() for x in sets_to_omit])

with open(directory + 'not_estimated', 'r', encoding='utf-8') as not_estimated:
    not_estimated = not_estimated.readlines()
not_estimated = set([x.strip() for x in not_estimated])


def save_results(result: Results, dir, num_predictions):
    for iter in range(num_predictions):
        matrix = result.get_result(iter).matrix
        file_name = dir + '/result_' + str(iter) + '.pickle'
        os.makedirs(os.path.dirname(file_name), exist_ok=True)
        with open(file_name, 'wb') as file:
            pickle.dump(matrix, file)


def estimate_t_and_predict(path_dataset_history, sets_to_omit, not_estimated, batch_type, batch_sizes, num_predictions):
    if path_dataset_history.split('/')[4] not in sets_to_omit:
        if path_dataset_history not in not_estimated:
            edges = pd.read_csv(os.path.dirname(path_dataset_history) + '/edges')
            event_log = pd.read_csv(path_dataset_history + '/event_log', header=None)
            with open(path_dataset_history + '/contagion.pickle', 'rb') as file:
                cc = pickle.load(file)
            with open(path_dataset_history + '/adjacency.pickle', 'rb') as file:
                a = pickle.load(file)
            for batch_size in batch_sizes:
                d = Data()
                d.load_data_data_frame(event_log, edges)
                m = MCDOI()
                m.assign_contagions_correlation_matrix(cc)
                m.assign_adjacency_matrix(a)
                m.fit_only_thresholds_states(d, batch_type = batch_type, batch_size = batch_size)
                result = m.predict(num_predictions)
                save_results(result, path_dataset_history + '/' + batch_type + '/size_' + str(batch_size), num_predictions)
                print(path_dataset_history + '/' + batch_type + '/size_' + str(batch_size))


def make_dataset_history_paths():
    paths = []
    for dat in next(os.walk(directory))[1]:
        for history_length in np.arange(1, 31, 1):
            paths.append(directory+dat+'/history_'+str(history_length))
    return paths

if __name__ == '__main__':
    aprun(bar='None')(delayed(estimate_t_and_predict)(dat, sets_to_omit, not_estimated, 'time', batch_sizes, 3) for dat in make_dataset_history_paths())








