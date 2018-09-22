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
sets_to_estimate_file = list(sys.argv)[1]
with open(sets_to_estimate_file, 'r', encoding='utf-8') as sets_to_estimate:
    sets_to_estimate = sets_to_estimate.readlines()
sets_to_estimate = [x.strip() for x in sets_to_estimate]

with open(directory+'estimated_cc+a', 'r+', encoding='utf-8') as file:
    estimated = file.readlines()
estimated = set([x.strip() for x in estimated])

with open(directory+'not_estimated_cc+a', 'r+', encoding='utf-8') as file:
    not_estimated = file.readlines()
not_estimated = set([x.strip() for x in not_estimated])

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

def write_to_logger(args):
    logging.basicConfig(filename='./' + datetime.now().strftime("%Y-%m-%d_%H_%M_%S") + '.log', level=logging.DEBUG,
                        format='%(asctime)s %(levelname)s %(name)s %(message)s')
    logger = logging.getLogger(__name__)
    logger.error(args, exc_info=True)

def send_email():
    import smtplib
    content = ("Done!")
    mail = smtplib.SMTP('smtp.gmail.com', 587)
    mail.ehlo()
    mail.starttls()
    mail.login('python2266@gmail.com', 'GXjj5ahH')
    mail.sendmail('python2266@gmail.com', 'falkiewicz.maciej@gmail.com', content)
    mail.close()

def save_results(result: Results, dir, num_predictions):
    for iter in range(num_predictions):
        matrix = result.get_result(iter).matrix
        file_name = dir + 'result_' + str(iter) + '.pickle'
        os.makedirs(os.path.dirname(file_name), exist_ok=True)
        with open(file_name, 'wb') as file:
            pickle.dump(matrix, file)

def save_parameters(m: MCDOI, dir):
    file_name = dir + 'adjacency.pickle'
    os.makedirs(os.path.dirname(file_name), exist_ok=True)
    with open(file_name, 'wb') as adjacency_file:
        pickle.dump(m.adjacency.matrix, adjacency_file)
    file_name = dir + 'contagion.pickle'
    os.makedirs(os.path.dirname(file_name), exist_ok=True)
    with open(file_name, 'wb') as contagion_file:
        pickle.dump(m.contagion_correlation.matrix, contagion_file)
    file_name = dir + 'threshold.pickle'
    os.makedirs(os.path.dirname(file_name), exist_ok=True)
    with open(file_name, 'wb') as threshold_file:
        pickle.dump(m.thresholds.matrix, threshold_file)


def estimate_and_predict(d, dir, batch_type, batch_size, num_predictions):
    try:
        m = MCDOI()
        m.fit(d, batch_type = batch_type, batch_size = batch_size)
        result = m.predict(num_predictions)
        save_results(result, dir + batch_type + '/size_' + str(batch_size) + '/', num_predictions)
        save_parameters(m, dir + batch_type + '/size_' + str(batch_size) + '/')
    except Exception as err:
        write_to_logger(err.args)
        print(err.args)
        exit(1)
    finally:
        send_email()


def proceed_with_history(history_length, directory, dataset, edges):
    dir = directory + dataset + '/history_' + str(history_length)
    if sum(1 for line in open(dir + '/event_log', 'r', encoding='utf-8')) > 0:
        event_log = pd.read_csv(dir + '/event_log', header=None)
        if len(event_log.iloc[:, 2].unique()) <= 2500:
            d = Data()
            d.load_data_data_frame(event_log, copy(edges))
            cc = ContagionCorrelation()
            cc.estimate(d)
            contagion_file_name = dir + '/contagion.pickle'
            os.makedirs(os.path.dirname(contagion_file_name), exist_ok=True)
            with open(contagion_file_name, 'wb') as contagion_file:
                pickle.dump(cc.matrix, contagion_file)
            a = Adjacency()
            a.estimate(d)
            adjacency_file_name = dir + '/adjacency.pickle'
            os.makedirs(os.path.dirname(adjacency_file_name), exist_ok=True)
            with open(adjacency_file_name, 'wb') as adjacency_file:
                pickle.dump(a.matrix, adjacency_file)
        else:
            with open(directory+'not_estimated', 'a+', encoding='utf-8') as file:
                file.write(dataset + '/history_' + str(history_length) + '\n')
    else:
        with open(directory + 'not_estimated', 'a+', encoding='utf-8') as file:
            file.write(dataset + '/history_' + str(history_length) + '\n')


with open(directory + 'sets_to_omit', 'r+', encoding='utf-8') as sets_to_omit:
    sets_to_omit = sets_to_omit.readlines()

sets_to_omit = set([x.strip() for x in sets_to_omit])

with open(directory + 'histories_to_omit', 'r+', encoding='utf-8') as histories_to_omit:
    histories_to_omit = histories_to_omit.readlines()

histories_to_omit = set([x.strip() for x in histories_to_omit])

# for dataset in tqdm(next(os.walk(directory))[1]):
#     if dataset not in sets_to_omit:
#         open(directory+'not_estimated', 'w', encoding='utf-8').close()
#         dir = directory + dataset
#         edges = pd.read_csv(dir+'/edges', header=None)
#         aprun(bar='None')(delayed(proceed_with_history)(history_length, directory, dataset, edges) for history_length in np.arange(1,31,1))

# single dataset passed by subdirectory name. All 30 histories proceeded in single thread
# def proceed_dataset(dataset, sets_to_omit):
#     if dataset not in sets_to_omit:
#         dir = directory + dataset
#         edges = pd.read_csv(dir + '/edges', header=None)
#         for history_length in np.arange(1, 31, 1):
#             proceed_with_history(history_length, directory, dataset, edges)

def proceed_with_history_path(path_dataset_history, edges):
    if sum(1 for line in open(path_dataset_history + '/event_log', 'r', encoding='utf-8')) > 0:
        event_log = pd.read_csv(path_dataset_history + '/event_log', header=None)
        d = Data()
        d.load_data_data_frame(event_log, edges)
        data_obj_file_name = path_dataset_history + '/data_obj.pickle'
        os.makedirs(os.path.dirname(data_obj_file_name), exist_ok=True)
        with open(data_obj_file_name, 'wb') as f:
            pickle.dump(d, f)
        contagion_dict_file_name = path_dataset_history + '/contagion_dict.pickle'
        os.makedirs(os.path.dirname(contagion_dict_file_name), exist_ok=True)
        with open(contagion_dict_file_name, 'wb') as contagion_dict_file:
            pickle.dump(d.contagion_id_dict, contagion_dict_file)
        cc = ContagionCorrelation()
        cc.estimate(d)
        contagion_file_name = path_dataset_history + '/contagion.pickle'
        os.makedirs(os.path.dirname(contagion_file_name), exist_ok=True)
        with open(contagion_file_name, 'wb') as contagion_file:
            pickle.dump(cc.matrix, contagion_file)
        a = Adjacency()
        a.estimate(d)
        adjacency_file_name = path_dataset_history + '/adjacency.pickle'
        os.makedirs(os.path.dirname(adjacency_file_name), exist_ok=True)
        with open(adjacency_file_name, 'wb') as adjacency_file:
            pickle.dump(a.matrix, adjacency_file)
        with open(directory + 'estimated_cc+a', 'a+', encoding='utf-8') as handle:
            handle.write(path_dataset_history + '\n')
    else:
        with open(directory + 'not_estimated_c+aa', 'a+', encoding='utf-8') as file:
            file.write(path_dataset_history + '\n')

# specific history from specific dataset passed by path.
def proceed_dataset_history_path(path_dataset_history, sets_to_omit, histories_to_omit):
    if path_dataset_history.split('/')[5] not in sets_to_omit:
        if path_dataset_history.split('/')[5]+'/'+path_dataset_history.split('/')[6] not in histories_to_omit:
            edges = pd.read_csv(os.path.dirname(path_dataset_history) + '/edges', header=None)
            proceed_with_history_path(path_dataset_history, edges)

def diff(first, second):
    second = set(second)
    return [item for item in first if item not in second]

def make_dataset_history_paths(sets_to_estimate):
    paths = []
    for set in sets_to_estimate:
        for history_length in np.arange(1, 31, 1):
            paths.append(set+'/history_'+str(history_length))
    return paths


if __name__ == '__main__':
    aprun(bar='txt')(delayed(proceed_dataset_history_path)(dat, sets_to_omit, histories_to_omit) for dat in diff(sets_to_estimate,estimated.union(not_estimated)))
# for dat in make_dataset_history_paths():
#     proceed_dataset_history_path(dat, sets_to_omit)
    # d = Data()
    # d.load_data(dir)
    # if d.num_contagions <= 25000:
    #     for batch_size in tqdm(batch_sizes[0:1]):
    #         estimate_and_predict(d, dir, 'time', batch_size, 3)
    # else:
    #     print('Number of contagions in "' + dataset + '" is equal to ' + str(d.num_contagions) + ', it is too much.')
    # print(dataset + ' done!')
