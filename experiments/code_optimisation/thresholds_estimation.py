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

directory = '/datasets/mcdoi/louvain/'

set = 'louvain_58_72/history_27'

def save_results(result: Results, dir, num_predictions):
    for iter in range(num_predictions):
        matrix = result.get_result(iter).matrix
        file_name = dir + '/result_' + str(iter) + '.pickle'
        os.makedirs(os.path.dirname(file_name), exist_ok=True)
        with open(file_name, 'wb') as file:
            pickle.dump(matrix, file)


def estimate_t_and_predict(path_dataset_history, batch_type, batch_sizes, num_predictions):
    edges = pd.read_csv(os.path.dirname(path_dataset_history) + '/edges', header = None)
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
        file_name = path_dataset_history + '/' + batch_type + '/size_' + str(batch_size) + '/threshold.pickle'
        # os.makedirs(os.path.dirname(file_name), exist_ok=True)
        # with open(file_name, 'wb') as threshold_file:
        #     pickle.dump(m.thresholds.matrix, threshold_file)
        result = m.predict(num_predictions)
        # save_results(result, path_dataset_history + '/' + batch_type + '/size_' + str(batch_size), num_predictions)
        # with open(directory+'estimated_t+predict', 'a+', encoding='utf-8') as handle:
        #     handle.write(path_dataset_history + '/' + batch_type + '/size_' + str(batch_size) + '\n')

if __name__ == '__main__':
    estimate_t_and_predict(directory+set, 'time', [43200], 3)