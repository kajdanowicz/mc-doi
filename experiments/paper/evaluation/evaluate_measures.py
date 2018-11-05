import pandas as pd
import os
import sys
sys.path.append('/home/maciek/pyCharmProjects/mc-doi')
from matplotlib import pyplot as plt
import csv
import numpy as np
import pickle
from data.data import Data
from collections import defaultdict
import functools
import itertools
import copy
from sklearn.metrics import confusion_matrix
from tqdm import tqdm

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

aprun = ParallelExecutor(n_jobs=22)

sets_to_evaluate_file = list(sys.argv)[1]
with open(sets_to_evaluate_file, 'r', encoding='utf-8') as sets_to_evaluate:
    sets_to_evaluate = sets_to_evaluate.readlines()
sets_to_evaluate = [x.strip() for x in sets_to_evaluate]

start_time = 1332565200
end_time = 1335416399
duration_24h_in_sec = 60*60*24
time_grid = np.arange(start_time+duration_24h_in_sec,end_time+duration_24h_in_sec,duration_24h_in_sec)

model = list(sys.argv)[2]

directory = '/nfs/maciej/mcdoi/paper/'+model+'/'

evaluated = set()
for batch_size in [3600, 43200, 86400, 604800]:
    if os.path.isfile(directory + 'evaluation/evaluated_'+str(batch_size)):
        with open(directory + 'evaluation/evaluated_'+str(batch_size), 'r', encoding='utf-8') as file:
            e = file.readlines()
        evaluated.update([x.strip() for x in e])
    else:
        os.makedirs(directory+'evaluation',exist_ok=True)
        open(directory + 'evaluation/evaluated_'+str(batch_size), 'w+', encoding='utf-8').close()

def diff(first, second):
    second = set(second)
    return [item for item in first if item not in second]

def evaluate(path, iter_length, model):
    new_path = path.split('/')
    new_path[4] = 'paper/' + model
    new_path = '/' + os.path.join(*new_path)
    history = int(path.split('/')[6].split('_')[1])
    batch_size = int(path.split('/')[8].split('_')[1])
    with open(os.path.dirname(os.path.dirname(os.path.dirname(path)))+'/edges', 'r', encoding='utf-8') as f:
        edges = pd.read_csv(f, header=None, names=[Data.user_1, Data.user_2])

    user_dict = defaultdict(functools.partial(next, itertools.count()))
    edges[Data.user_1] = edges[Data.user_1].apply(lambda x: user_dict[x])
    edges[Data.user_2] = edges[Data.user_2].apply(lambda x: user_dict[x])

    with open(os.path.dirname(os.path.dirname(os.path.dirname(path)))+'/event_log', 'r', encoding='utf-8') as f:
        whole_event_log = pd.read_csv(f, header=None, names=[Data.time_stamp, Data.user, Data.contagion])
    whole_event_log.user = whole_event_log.user.apply(lambda x: user_dict[x])

    with open(os.path.dirname(os.path.dirname(path))+'/data_obj.pickle', 'rb') as f:
        d=pickle.load(f)

    with open(os.path.dirname(os.path.dirname(path))+'/contagion_dict.pickle', 'rb') as f:
        contagion_dict=pickle.load(f)

    org_contagion_dict = copy.copy(contagion_dict)

    max_contagion_id = max(contagion_dict.values())

    rev_contagion_dict = {v:k for k,v in contagion_dict.items()}

    whole_event_log[Data.contagion_id] = whole_event_log[Data.contagion].apply(lambda x: contagion_dict[x])
    whole_event_log=whole_event_log[whole_event_log[Data.contagion_id]<=max_contagion_id]

    I_beginning = np.full((d.num_users, d.num_contagions), False, dtype=bool)
    for index, row in whole_event_log[whole_event_log[Data.time_stamp]<=time_grid[history-1]].iterrows():
        I_beginning[row[Data.user]][row[Data.contagion_id]] = True

    indicators = []
    I = np.full((d.num_users, d.num_contagions), False, dtype=bool)
    for i in range(1,min(7,33-history)+1):
        for index, row in whole_event_log[whole_event_log[Data.time_stamp]<=time_grid[history-1] + i * iter_length].iterrows():
            I[row[Data.user]][row[Data.contagion_id]] = True
        indicators.append(I)
        I = copy.deepcopy(I)

    event_log_train = whole_event_log[whole_event_log['ts'] <= time_grid[history - 1]].drop_duplicates(subset=['contagion', 'user'], keep='first')
    # event_log_train = event_log_train.groupby(by=['user']).count()['ts']

    results = []
    for i in range(0, 7):
        with open(new_path + '/result_' + str(i) + '.pickle', 'rb') as result:
            # res = (pickle.load(result)==1)
            res = pickle.load(result)
            results.append(res)

    for i in range(1,min(7,33-history)+1):
        contagion_fscore(batch_size, d, i, indicators, new_path, results, rev_contagion_dict)
        contagion_fscore_diff(I_beginning, batch_size, d, i, indicators, new_path, results, rev_contagion_dict)
        contagion_fractions(batch_size, d, history, i, iter_length, new_path, results, whole_event_log)
        contagion_fractions_diff(batch_size, d, event_log_train, history, i, iter_length, new_path, results,
                                 whole_event_log)
        # contagion_jaccard(batch_size, d, i, indicators, new_path, results, rev_contagion_dict)
        # contagion_jaccard_diff(I_beginning, batch_size, d, i, indicators, new_path, results, rev_contagion_dict)
        # fscore(batch_size, d, i, indicators, new_path, results)
        # fscore_diff(I_beginning, batch_size, d, i, indicators, new_path, results)
        # fractions_diff(batch_size, org_contagion_dict, event_log_train, history, i, iter_length, new_path, results, whole_event_log)
        # fractions(batch_size, org_contagion_dict, history, i, iter_length, new_path, results, whole_event_log)
        # jaccard(batch_size, d, i, indicators, new_path, results)
        # jaccard_diff(I_beginning, batch_size, d, i, indicators, new_path, results)

    with open(directory + 'evaluation/evaluated_'+str(batch_size), 'a', encoding='utf-8') as file:
        file.write(path + '\n')


def jaccard_diff(I_beginning, batch_size, d, i, indicators, new_path, results):
    open(new_path + '/jaccard_diff_' + str(i - 1), 'w', encoding='utf-8').close()
    result_diff = np.logical_xor(results[i - 1], I_beginning)
    real_diff = np.logical_xor(indicators[i - 1], I_beginning)
    for user in range(d.num_users):
        set_from_prediction = np.where(result_diff[user, :])
        set_real = np.where(real_diff[user, :])
        intersection = np.intersect1d(set_from_prediction, set_real)
        union = np.union1d(set_from_prediction, set_real)
        if union.size == 0:
            with open(new_path + '/jaccard_diff_' + str(i - 1), 'a', encoding='utf-8') as file:
                file.write(str(user) + ',' + str(1) + '\n')
        else:
            with open(new_path + '/jaccard_diff_' + str(i - 1), 'a', encoding='utf-8') as file:
                file.write(str(user) + ',' + str(intersection.size / union.size) + '\n')
    with open(directory + 'evaluation/jaccard_diff_' + str(batch_size), 'a+', encoding='utf-8') as file:
        file.write(new_path + '/jaccard_diff_' + str(i - 1) + '\n')


def jaccard(batch_size, d, i, indicators, new_path, results):
    open(new_path + '/jaccard_' + str(i - 1), 'w', encoding='utf-8').close()
    for user in range(d.num_users):
        set_from_prediction = np.where(results[i - 1][user, :])
        set_real = np.where(indicators[i - 1][user, :])
        intersection = np.intersect1d(set_from_prediction, set_real)
        union = np.union1d(set_from_prediction, set_real)
        if union.size == 0:
            with open(new_path + '/jaccard_' + str(i - 1), 'a', encoding='utf-8') as file:
                file.write(str(user) + ',' + str(1) + '\n')
        else:
            with open(new_path + '/jaccard_' + str(i - 1), 'a', encoding='utf-8') as file:
                file.write(str(user) + ',' + str(intersection.size / union.size) + '\n')
    with open(directory + 'evaluation/jaccard_' + str(batch_size), 'a+', encoding='utf-8') as file:
        file.write(new_path + '/jaccard_' + str(i - 1) + '\n')


def fractions(batch_size, contagion_dict, history, i, iter_length, new_path, results, whole_event_log):
    open(new_path + '/fractions_' + str(i - 1), 'w', encoding='utf-8').close()
    e = whole_event_log[whole_event_log['ts'] <= time_grid[history - 1] + i * iter_length].drop_duplicates(
        subset=['contagion', 'user'], keep='first')
    e = e.groupby(by=['contagion']).count()['ts']
    for key, value in contagion_dict.items():
        with open(new_path + '/fractions_' + str(i - 1), 'a', encoding='utf-8') as file:
            file.write(key + ',' + str(e.loc[key] / results[0].shape[0]) + ',' + str(
                np.sum(results[i - 1], axis=0)[value] / results[0].shape[0]) + '\n')
    with open(directory + 'evaluation/fractions_' + str(batch_size), 'a+', encoding='utf-8') as file:
        file.write(new_path + '/fractions_' + str(i - 1) + '\n')


def fractions_diff(batch_size, contagion_dict, event_log_train, history, i, iter_length, new_path, results, whole_event_log):
    e_org = event_log_train.groupby(by=['contagion']).count()['ts']
    open(new_path + '/fractions_diff_' + str(i - 1), 'w', encoding='utf-8').close()
    e = whole_event_log[whole_event_log['ts'] <= time_grid[history - 1] + i * iter_length].drop_duplicates(
        subset=['contagion', 'user'], keep='first')
    e = e.groupby(by=['contagion']).count()['ts']
    for key, value in contagion_dict.items():
        with open(new_path + '/fractions_diff_' + str(i - 1), 'a', encoding='utf-8') as file:
            file.write(key + ',' + str((e.loc[key] - e_org.get(key,0)) / results[0].shape[0]) + ',' + str((np.sum(results[i - 1], axis=0)[value] - e_org.get(key,0)) / results[0].shape[0]) + '\n')
    with open(directory + 'evaluation/fractions_diff_' + str(batch_size), 'a+', encoding='utf-8') as file:
        file.write(new_path + '/fractions_diff_' + str(i - 1) + '\n')


def fscore_diff(I_beginning, batch_size, d, i, indicators, new_path, results):
    open(new_path + '/fscore_diff_' + str(i - 1), 'w', encoding='utf-8').close()
    result_diff = np.logical_xor(results[i - 1], I_beginning)
    real_diff = np.logical_xor(indicators[i - 1], I_beginning)
    for user in range(d.num_users):
        with open(new_path + '/fscore_diff_' + str(i - 1), 'a', encoding='utf-8') as file:
            score = confusion_matrix(real_diff[user, :], result_diff[user, :], labels=[0, 1]).ravel()
            file.write(str(user) + ',' + str(score[0]) + ',' + str(score[1]) + ',' + str(score[2]) + ',' + str(
                score[3]) + '\n')
    with open(directory + 'evaluation/fscore_diff_' + str(batch_size), 'a+', encoding='utf-8') as file:
        file.write(new_path + '/fscore_diff_' + str(i - 1) + '\n')


def fscore(batch_size, d, i, indicators, new_path, results):
    open(new_path + '/fscore_' + str(i - 1), 'w', encoding='utf-8').close()
    for user in range(d.num_users):
        with open(new_path + '/fscore_' + str(i - 1), 'a', encoding='utf-8') as file:
            score = confusion_matrix(indicators[i - 1][user, :], results[i - 1][user, :], labels=[0, 1]).ravel()
            file.write(str(user) + ',' + str(score[0]) + ',' + str(score[1]) + ',' + str(score[2]) + ',' + str(
                score[3]) + '\n')
    with open(directory + 'evaluation/fscore_' + str(batch_size), 'a+', encoding='utf-8') as file:
        file.write(new_path + '/fscore_' + str(i - 1) + '\n')


def contagion_jaccard_diff(I_beginning, batch_size, d, i, indicators, new_path, results, rev_contagion_dict):
    open(new_path + '/contagion_jaccard_diff_' + str(i - 1), 'w', encoding='utf-8').close()
    result_diff = np.logical_xor(results[i - 1], I_beginning)
    real_diff = np.logical_xor(indicators[i - 1], I_beginning)
    for contagion_id in range(d.num_contagions):
        set_from_prediction = np.where(result_diff[:, contagion_id])
        set_real = np.where(real_diff[:, contagion_id])
        intersection = np.intersect1d(set_from_prediction, set_real)
        union = np.union1d(set_from_prediction, set_real)
        if union.size == 0:
            with open(new_path + '/contagion_jaccard_diff_' + str(i - 1), 'a', encoding='utf-8') as file:
                file.write(rev_contagion_dict[contagion_id] + ',' + str(1) + '\n')
        else:
            with open(new_path + '/contagion_jaccard_diff_' + str(i - 1), 'a', encoding='utf-8') as file:
                file.write(rev_contagion_dict[contagion_id] + ',' + str(intersection.size / union.size) + '\n')
    with open(directory + 'evaluation/contagion_jaccard_diff_' + str(batch_size), 'a+', encoding='utf-8') as file:
        file.write(new_path + '/contagion_jaccard_diff_' + str(i - 1) + '\n')


def contagion_jaccard(batch_size, d, i, indicators, new_path, results, rev_contagion_dict):
    open(new_path + '/contagion_jaccard_' + str(i - 1), 'w', encoding='utf-8').close()
    for contagion_id in range(d.num_contagions):
        set_from_prediction = np.where(results[i - 1][:, contagion_id])
        set_real = np.where(indicators[i - 1][:, contagion_id])
        intersection = np.intersect1d(set_from_prediction, set_real)
        union = np.union1d(set_from_prediction, set_real)
        if union.size == 0:
            with open(new_path + '/contagion_jaccard_' + str(i - 1), 'a', encoding='utf-8') as file:
                file.write(rev_contagion_dict[contagion_id] + ',' + str(1) + '\n')
        else:
            with open(new_path + '/contagion_jaccard_' + str(i - 1), 'a', encoding='utf-8') as file:
                file.write(rev_contagion_dict[contagion_id] + ',' + str(intersection.size / union.size) + '\n')
    with open(directory + 'evaluation/contagion_jaccard_' + str(batch_size), 'a+', encoding='utf-8') as file:
        file.write(new_path + '/contagion_jaccard_' + str(i - 1) + '\n')


def contagion_fractions_diff(batch_size, d, event_log_train, history, i, iter_length, new_path, results,
                             whole_event_log):
    e_org = event_log_train.groupby(by=['user']).count()['ts']
    open(new_path + '/contagion_fractions_diff_' + str(i - 1), 'w', encoding='utf-8').close()
    e = whole_event_log[whole_event_log['ts'] <= time_grid[history - 1] + i * iter_length].drop_duplicates(
        subset=['contagion', 'user'], keep='first')
    e = e.groupby(by=['user']).count()['ts']
    for user in range(d.num_users):
        with open(new_path + '/contagion_fractions_diff_' + str(i - 1), 'a', encoding='utf-8') as file:
            file.write(
                str(user) + ',' + str((e.get(user, 0) - e_org.get(user, 0)) / d.num_contagions) + ',' + str(
                    (np.sum(results[i - 1], axis=1)[user] - e_org.get(user, 0)) / d.num_contagions) + '\n')
    with open(directory + 'evaluation/contagion_fractions_diff_' + str(batch_size), 'a+', encoding='utf-8') as file:
        file.write(new_path + '/contagion_fractions_diff_' + str(i - 1) + '\n')


def contagion_fractions(batch_size, d, history, i, iter_length, new_path, results, whole_event_log):
    open(new_path + '/contagion_fractions_' + str(i - 1), 'w', encoding='utf-8').close()
    e = whole_event_log[whole_event_log['ts'] <= time_grid[history - 1] + i * iter_length].drop_duplicates(
        subset=['contagion', 'user'], keep='first')
    e = e.groupby(by=['user']).count()['ts']
    for user in range(d.num_users):
        with open(new_path + '/contagion_fractions_' + str(i - 1), 'a', encoding='utf-8') as file:
            file.write(str(user) + ',' + str(e.get(user, 0) / d.num_contagions) + ',' + str(
                np.sum(results[i - 1], axis=1)[user] / d.num_contagions) + '\n')
    with open(directory + 'evaluation/contagion_fractions_' + str(batch_size), 'a+', encoding='utf-8') as file:
        file.write(new_path + '/contagion_fractions_' + str(i - 1) + '\n')


def contagion_fscore_diff(I_beginning, batch_size, d, i, indicators, new_path, results, rev_contagion_dict):
    open(new_path + '/contagion_fscore_diff_' + str(i - 1), 'w', encoding='utf-8').close()
    result_diff = np.logical_xor(results[i - 1], I_beginning)
    real_diff = np.logical_xor(indicators[i - 1], I_beginning)
    for contagion_id in range(d.num_contagions):
        with open(new_path + '/contagion_fscore_diff_' + str(i - 1), 'a', encoding='utf-8') as file:
            score = confusion_matrix(real_diff[:, contagion_id], result_diff[:, contagion_id], labels=[0, 1]).ravel()
            file.write(rev_contagion_dict[contagion_id] + ',' + str(score[0]) + ',' + str(score[1]) + ',' + str(
                score[2]) + ',' + str(score[3]) + '\n')
    with open(directory + 'evaluation/contagion_fscore_diff_' + str(batch_size), 'a+', encoding='utf-8') as file:
        file.write(new_path + '/contagion_fscore_diff_' + str(i - 1) + '\n')


def contagion_fscore(batch_size, d, i, indicators, new_path, results, rev_contagion_dict):
    open(new_path + '/contagion_fscore_' + str(i - 1), 'w', encoding='utf-8').close()
    for contagion_id in range(d.num_contagions):
        with open(new_path + '/contagion_fscore_' + str(i - 1), 'a', encoding='utf-8') as file:
            score = confusion_matrix(indicators[i - 1][:, contagion_id], results[i - 1][:, contagion_id]).ravel()
            file.write(rev_contagion_dict[contagion_id] + ',' + str(score[0]) + ',' + str(score[1]) + ',' + str(
                score[2]) + ',' + str(score[3]) + '\n')
    with open(directory + 'evaluation/contagion_fscore_' + str(batch_size), 'a+', encoding='utf-8') as file:
        file.write(new_path + '/contagion_fscore_' + str(i - 1) + '\n')


if __name__ == '__main__':
    paths = diff(sets_to_evaluate,evaluated)
    # for path in tqdm(paths):
    #     evaluate(path,86400,model)
    aprun(bar='txt')(delayed(evaluate)(path,86400,model) for path in paths)