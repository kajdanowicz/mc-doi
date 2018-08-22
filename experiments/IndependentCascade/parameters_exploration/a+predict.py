import pickle
import os
import sys
sys.path.append('/home/maciek/pyCharmProjects/mc-doi')
from model.results import Results
import pandas as pd
from data.data import Data
from model.parameters import ContagionCorrelation, Adjacency
from model.single_contagion_models import SingleContagionIndependentCascade as IndependentCascade
import numpy as np

sets_to_estimate_file = list(sys.argv)[1]
with open(sets_to_estimate_file, 'r', encoding='utf-8') as sets_to_estimate:
    sets_to_estimate = sets_to_estimate.readlines()
sets_to_estimate = [x.strip() for x in sets_to_estimate]

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

aprun = ParallelExecutor(n_jobs=6)

directory = '/nfs/maciej/mcdoi/independent-cascade/'

with open(directory+'estimated_a+predict', 'r', encoding='utf-8') as file:
    estimated = file.readlines()
estimated = set([x.strip() for x in estimated])

with open(directory+'not_estimated_a', 'r+', encoding='utf-8') as file:
    not_estimated = file.readlines()
not_estimated = set([x.strip() for x in not_estimated])

# batch_sizes = [604800]# [43200, 86400, 604800] # (1h), 12h, 24h, 7d
# batch_sizes.reverse()

# with open(directory + 'sets_to_omit', 'r', encoding='utf-8') as sets_to_omit:
#     sets_to_omit = sets_to_omit.readlines()
# sets_to_omit = set([x.strip() for x in sets_to_omit])
#
# with open(directory + 'not_estimated', 'r', encoding='utf-8') as not_estimated:
#     not_estimated = not_estimated.readlines()
# not_estimated = set([x.strip() for x in not_estimated])


def save_results(result: Results, dir, num_predictions):
    for iter in range(num_predictions):
        matrix = result.get_result(iter).matrix
        file_name = dir + '/result_' + str(iter) + '.pickle'
        os.makedirs(os.path.dirname(file_name), exist_ok=True)
        with open(file_name, 'wb') as file:
            pickle.dump(matrix, file)

def proceed_dataset(dataset, estimated):
    for history_length in np.arange(1, 31, 1):
        estimate_a_and_predict(dataset+'/history_'+str(history_length), 7, estimated)

def estimate_a_and_predict(path_dataset_history, num_predictions, estimated):
    if path_dataset_history not in estimated:
        flag=False
        if os.path.isfile(path_dataset_history + '/data_obj.pickle'):
            with open(path_dataset_history + '/data_obj.pickle', 'rb') as f:
                d = pickle.load(f)
            flag = True
        else:
            if sum(1 for line in open(path_dataset_history + '/event_log', 'r', encoding='utf-8')) > 0:
                event_log = pd.read_csv(path_dataset_history + '/event_log', header=None)
                edges = pd.read_csv(os.path.dirname(path_dataset_history) + '/edges', header=None)
                d = Data()
                d.load_data_data_frame(event_log, edges)
                with open(path_dataset_history + '/data_obj.pickle', 'wb') as f:
                    pickle.dump(d, f)
                flag=True
            else:
                with open(directory + '/not_estimated_a', 'a+', encoding='utf-8') as file:
                    file.write(path_dataset_history + '\n')
        if flag:
            m = IndependentCascade()
            m.fit(d)
            new_path_dataset_history = path_dataset_history.split('/')
            new_path_dataset_history[4] = 'independent-cascade'
            new_path_dataset_history = '/'+os.path.join(*new_path_dataset_history)
            file_name = new_path_dataset_history + '/adjacency.pickle'
            os.makedirs(os.path.dirname(file_name), exist_ok=True)
            with open(file_name, 'wb') as adjacency_file:
                pickle.dump(m.adjacency.matrix, adjacency_file)
            result = m.predict(num_predictions)
            save_results(result, new_path_dataset_history, num_predictions)
            with open(directory+'estimated_a+predict', 'a+', encoding='utf-8') as handle:
                handle.write(new_path_dataset_history + '\n')


# def make_dataset_history_paths():
#     paths = []
#     for dat in next(os.walk(directory))[1]:
#         for history_length in np.arange(1, 31, 1):
#             paths.append(directory+dat+'/history_'+str(history_length))
#     return paths

if __name__ == '__main__':
    aprun(bar='txt')(delayed(proceed_dataset)(dat, estimated.union(not_estimated)) for dat in sets_to_estimate)
    # for dat in sets_to_estimate:
    #     proceed_dataset(dat, estimated.union(not_estimated))








