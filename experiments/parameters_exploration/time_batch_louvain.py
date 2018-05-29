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


directory = '/datasets/mcdoi/louvain/'

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

aprun = ParallelExecutor(n_jobs=12)

batch_sizes = [60, 3600, 43200, 86400, 604800]


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
    event_log = pd.read_csv(dir + '/event_log')
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


for dataset in tqdm(next(os.walk(directory))[1]):
    dir = directory + dataset
    edges = pd.read_csv(dir+'/edges')
    aprun(bar='None')(delayed(proceed_with_history)(history_length, directory, dataset, edges) for history_length in np.arange(1,31,1))





    # d = Data()
    # d.load_data(dir)
    # if d.num_contagions <= 25000:
    #     for batch_size in tqdm(batch_sizes[0:1]):
    #         estimate_and_predict(d, dir, 'time', batch_size, 3)
    # else:
    #     print('Number of contagions in "' + dataset + '" is equal to ' + str(d.num_contagions) + ', it is too much.')
    # print(dataset + ' done!')
