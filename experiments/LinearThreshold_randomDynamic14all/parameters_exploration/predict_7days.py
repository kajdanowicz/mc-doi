import pickle
import os
import sys
sys.path.append('/home/maciek/pyCharmProjects/mc-doi')
from model.results import Results
import pandas as pd
from data.data import Data
from model.parameters import ContagionCorrelation, Adjacency
from model.multi_contagion_models import MultiContagionDynamicLinearThresholdModel as MCDOI
import numpy as np

sets_to_predict_file = list(sys.argv)[1]
with open(sets_to_predict_file, 'r', encoding='utf-8') as sets_to_predict:
    sets_to_predict = sets_to_predict.readlines()
sets_to_predict = [x.strip() for x in sets_to_predict]

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

aprun = ParallelExecutor(n_jobs=18)

directory = '/nfs/maciej/mcdoi/negative-random-activation/'

with open(directory+'predicted_7days', 'r', encoding='utf-8') as file:
    predicted = file.readlines()
predicted = [x.strip() for x in predicted]


def save_results(result: Results, dir, num_predictions):
    for iter in range(num_predictions):
        matrix = result.get_result(iter).matrix
        file_name = dir + '/result_' + str(iter) + '.pickle'
        os.makedirs(dir, exist_ok=True)
        with open(file_name, 'wb') as file:
            pickle.dump(matrix, file)


def predict(path, num_predictions, predicted):
    if path not in predicted:
        if os.path.isfile(os.path.dirname(os.path.dirname(path))+'/data_obj.pickle'):
            with open(os.path.dirname(os.path.dirname(path))+'/data_obj.pickle', 'rb') as f:
                d = pickle.load(f)
        else:
            edges_dir = os.path.dirname(os.path.dirname(os.path.dirname(path))) + '/edges'
            edges = pd.read_csv(edges_dir, header = None)
            event_log_dir = os.path.dirname(os.path.dirname(path))
            event_log = pd.read_csv(event_log_dir+ '/event_log', header=None)
            d = Data()
            d.load_data_data_frame(event_log, edges)
            with open(event_log_dir+'/data_obj.pickle', 'wb') as f:
                pickle.dump(d, f)
            # print(event_log_dir)
        with open(os.path.dirname(os.path.dirname(path)) + '/contagion.pickle', 'rb') as file:
            cc = pickle.load(file)
        with open(os.path.dirname(os.path.dirname(path)) + '/adjacency.pickle', 'rb') as file:
            a = pickle.load(file)
        with open(path + '/threshold.pickle', 'rb') as f:
            t = pickle.load(f)
        m = MCDOI()
        m.assign_contagions_correlation_matrix(cc)
        m.assign_adjacency_matrix(a)
        m.assign_thresholds_matrix(t)
        m.fill_state_matrix(d)
        result = m.predict(num_predictions)
        new_path = path.split('/')
        new_path[4] = 'negative-random-activation'
        save_results(result, '/'+os.path.join(*new_path), num_predictions)
        with open(directory+'predicted_7days', 'a+', encoding='utf-8') as handle:
            handle.write(path + '\n')

# def make_dataset_history_paths():
#     paths = []
#     for dat in next(os.walk(directory))[1]:
#         for history_length in np.arange(1, 31, 1):
#             paths.append(directory+dat+'/history_'+str(history_length))
#     return paths

if __name__ == '__main__':
    aprun(bar='txt')(delayed(predict)(path, 7, predicted) for path in sets_to_predict)








