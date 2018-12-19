import pandas as pd
from os import listdir
from os.path import isfile, join
import re
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


def df_empty(columns, dtypes, index=None):
    df = pd.DataFrame(index=index)
    for c,d in zip(columns, dtypes):
        df[c] = pd.Series(dtype=d)
    return df


def measure_aggregation(argument, row):
    df = row
    #     ['tag','TN', 'FP', 'FN', 'TP']
    if argument == 'contagion_f1':
        return 2 * df[4] / (2 * df[4] + df[2] + df[3])
    elif argument == 'contagion_precision':
        return df[4] / (df[4] + df[2])
    elif argument == 'contagion_recall':
        return df[4] / (df[4] + df[3])

    elif argument == 'contagion_f1_diff':
        return 2 * df[4] / (2 * df[4] + df[2] + df[3])
    elif argument == 'contagion_precision_diff':
        return df[4] / (df[4] + df[2])
    elif argument == 'contagion_recall_diff':
        return df[4] / (df[4] + df[3])

    elif argument == 'contagion_f1_fillna_diff':
        return (2 * df[4] / (2 * df[4] + df[2] + df[3])).fillna(1)
    elif argument == 'contagion_precision_fillna_diff':
        return (df[4] / (df[4] + df[2])).fillna(1)
    elif argument == 'contagion_recall_fillna_diff':
        return (df[4] / (df[4] + df[3])).fillna(1)

    elif argument == 'contagion_f1_laplace_diff':
        return (2 * df[4]+1) / (2 * df[4] + df[2] + df[3]+1)
    elif argument == 'contagion_precision_laplace_diff':
        return (df[4]+1) / (df[4] + df[2] + 1)
    elif argument == 'contagion_recall_laplace_diff':
        return (df[4]+1) / (df[4] + df[3] + 1)

    elif argument == 'contagion_f1_reversed_diff':
        return 2 * df[1] / (2 * df[1] + df[3] + df[2])
    elif argument == 'contagion_f1_average_diff':
        return 2 * (df[4] + df[1]) / (2 * (df[4] + df[1]) + (df[2] + df[3]) + (df[3] + df[2]))

    elif argument == 'contagion_fractions_diff':
        return df[1] - df[2]  # real - predicted


def proceed_normal_model(model, batch_size, history):
    results = df_empty(
        columns=['community_id', 'community_size', 'model', 'train_len', 'prediction_period', 'measure_name',
                 'measure_value','contagion'],
        dtypes=['int', 'int', 'str', 'int', 'int', 'str', 'float', 'float','str'])
    base_path = '/nfs/maciej/mcdoi/paper/' + model + '/evaluation/'
    onlyfiles = [f for f in listdir(base_path) if
                 (isfile(join(base_path, f)) & f.endswith(str(batch_size)) & (('contagion_fractions_diff' in f) or ('contagion_fscore' in f) or ('contagion_fscore_diff' in f)))]
    for file_name in onlyfiles:
        with open(join(base_path, file_name), 'r') as current_file:
            for line in [f for f in current_file if ('history_'+str(history) in f)]:
                line = line.strip()
                with open(line, 'r') as file_with_results:
                    df = pd.read_csv(file_with_results, header=None)
                line = line.split('/')
                community_id = int(line[6].split('_')[1])
                community_size = int(line[6].split('_')[2])
                train_len = int(line[7].split('_')[1])
                prediction_period = int(line[-1].split('_')[-1])
                measure_name = '_'.join(line[-1].split('_')[0:-1])
                if 'fscore_diff' in measure_name:
                    for measure in ['f1','precision','recall','f1_fillna','precision_fillna','recall_fillna','f1_laplace','precision_laplace','recall_laplace','f1_average', 'f1_reversed']:
                        results_measure = pd.DataFrame()
                        measure_name_new = measure_name.replace('fscore',measure)
                        results_measure['measure_value'] = measure_aggregation(measure_name_new, df)
                        results_measure['community_id'] = community_id
                        results_measure['community_size'] = community_size
                        results_measure['model'] = model
                        results_measure['train_len'] = train_len
                        results_measure['prediction_period'] = prediction_period
                        results_measure['measure_name'] = measure_name_new
                        results_measure['contagion'] = df[0]
                        results = pd.concat([results, results_measure], ignore_index = True, sort=False)
                elif 'fscore' in measure_name:
                    for measure in ['f1','precision','recall']:
                        results_measure = pd.DataFrame()
                        measure_name_new = measure_name.replace('fscore',measure)
                        results_measure['measure_value'] = measure_aggregation(measure_name_new, df)
                        results_measure['community_id'] = community_id
                        results_measure['community_size'] = community_size
                        results_measure['model'] = model
                        results_measure['train_len'] = train_len
                        results_measure['prediction_period'] = prediction_period
                        results_measure['measure_name'] = measure_name_new
                        results_measure['contagion'] = df[0]
                        results = pd.concat([results, results_measure], ignore_index = True, sort=False)
                else:
                    results_measure = pd.DataFrame()
                    results_measure['measure_value'] = measure_aggregation(measure_name, df)
                    results_measure['community_id'] = community_id
                    results_measure['community_size'] = community_size
                    results_measure['model'] = model
                    results_measure['train_len'] = train_len
                    results_measure['prediction_period'] = prediction_period
                    results_measure['measure_name'] = measure_name
                    results_measure['contagion'] = df[0]
                    results = pd.concat([results, results_measure], ignore_index=True, sort=False)
    return results


def proceed_ic_model(history):
    results = df_empty(
        columns=['community_id', 'community_size', 'model', 'train_len', 'prediction_period', 'measure_name',
                 'measure_value','contagion'],
        dtypes=['int', 'int', 'str', 'int', 'int', 'str', 'float', 'float','str'])
    base_path = '/nfs/maciej/mcdoi/paper/' + 'independent-cascade' + '/evaluation/'
    onlyfiles = [f for f in listdir(base_path) if
                 (isfile(join(base_path, f)) & (('contagion_fractions_diff' in f) or ('contagion_fscore' in f) or ('contagion_fscore_diff' in f)))]
    for file_name in onlyfiles:
        with open(join(base_path, file_name), 'r') as current_file:
            for line in [f for f in current_file if ('history_'+str(history) in f)]:
                line = line.strip()
                with open(line, 'r') as file_with_results:
                    df = pd.read_csv(file_with_results, header=None)
                line = line.split('/')
                community_id = int(line[6].split('_')[1])
                community_size = int(line[6].split('_')[2])
                train_len = int(line[7].split('_')[1])
                prediction_period = int(line[-1].split('_')[-1])
                measure_name = '_'.join(line[-1].split('_')[0:-1])
                if 'fscore_diff' in measure_name:
                    for measure in ['f1','precision','recall','f1_fillna','precision_fillna','recall_fillna','f1_laplace','precision_laplace','recall_laplace','f1_average', 'f1_reversed']:
                        results_measure = pd.DataFrame()
                        measure_name_new = measure_name.replace('fscore',measure)
                        results_measure['measure_value'] = measure_aggregation(measure_name_new, df)
                        results_measure['community_id'] = community_id
                        results_measure['community_size'] = community_size
                        results_measure['model'] = 'independent-cascade'
                        results_measure['train_len'] = train_len
                        results_measure['prediction_period'] = prediction_period
                        results_measure['measure_name'] = measure_name_new
                        results_measure['contagion'] = df[0]
                        results = pd.concat([results, results_measure], ignore_index = True, sort=False)
                elif 'fscore' in measure_name:
                    for measure in ['f1','precision','recall']:
                        results_measure = pd.DataFrame()
                        measure_name_new = measure_name.replace('fscore',measure)
                        results_measure['measure_value'] = measure_aggregation(measure_name_new, df)
                        results_measure['community_id'] = community_id
                        results_measure['community_size'] = community_size
                        results_measure['model'] = 'independent-cascade'
                        results_measure['train_len'] = train_len
                        results_measure['prediction_period'] = prediction_period
                        results_measure['measure_name'] = measure_name_new
                        results_measure['contagion'] = df[0]
                        results = pd.concat([results, results_measure], ignore_index = True, sort=False)
                else:
                    results_measure = pd.DataFrame()
                    results_measure['measure_value'] = measure_aggregation(measure_name, df)
                    results_measure['community_id'] = community_id
                    results_measure['community_size'] = community_size
                    results_measure['model'] = 'independent-cascade'
                    results_measure['train_len'] = train_len
                    results_measure['prediction_period'] = prediction_period
                    results_measure['measure_name'] = measure_name
                    results_measure['contagion'] = df[0]
                    results = pd.concat([results, results_measure], ignore_index=True, sort=False)
    return results


models = ['correlated-linear-dynamic-threshold','correlated-linear-threshold','linear-dynamic-threshold','linear-threshold','linear-threshold-random','linear-threshold-random-single-theta']


aprun = ParallelExecutor(n_jobs=len(models))

if __name__ == '__main__':
    list_of_dataframes = aprun(bar='txt')(delayed(proceed_normal_model)(model, '86400', 10) for model in models)
    list_of_dataframes.append(proceed_ic_model(10))
    final_data_frame = pd.concat(list_of_dataframes, ignore_index=True, sort=False)
    final_data_frame.to_pickle('/nfs/maciej/mcdoi/paper/results_dataframes/macro_averaging_dataframe.pickle')



