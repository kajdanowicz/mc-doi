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
    if argument == 'contagion_f1_micro':
        return 2*df[4].sum()/(2*df[4].sum()+df[2].sum()+df[3].sum())
    elif argument == 'contagion_precision_micro':
        return df[4].sum()/(df[4].sum()+df[2].sum())
    elif argument == 'contagion_recall_micro':
        return df[4].sum()/(df[4].sum()+df[3].sum())

    elif argument == 'contagion_f1_micro_diff':
        return 2*df[4].sum()/(2*df[4].sum()+df[2].sum()+df[3].sum())
    elif argument == 'contagion_precision_micro_diff':
        return df[4].sum()/(df[4].sum()+df[2].sum())
    elif argument == 'contagion_recall_micro_diff':
        return df[4].sum()/(df[4].sum()+df[3].sum())

    elif argument == 'contagion_f1_reversed_micro_diff':
        return 2*df[1].sum()/(2*df[1].sum()+df[3].sum()+df[2].sum())
    elif argument == 'contagion_f1_average_micro_diff':
        return 2*(df[4].sum()+df[1].sum())/(2*(df[4].sum()+df[1].sum())+(df[2].sum()+df[3].sum())+(df[3].sum()+df[2].sum()))

    elif argument == 'contagion_fractions_diff_MAE':
        return (np.abs(df[1]-df[2])).mean()  # real - predicted
    elif argument == 'contagion_fractions_diff_RMSE':
        return np.sqrt(np.mean((df[1]-df[2])**2))  # real - predicted


def proceed_normal_model(model, batch_size, history):
    results = df_empty(
        columns=['community_id', 'community_size', 'model', 'train_len', 'prediction_period', 'measure_name',
                 'measure_value'],
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
                    for measure in ['f1_micro','precision_micro','recall_micro','f1_average_micro', 'f1_reversed_micro']:
                        measure_name_new = measure_name.replace('fscore',measure)
                        results.loc[len(results) + 1] = [community_id, community_size, model, train_len, prediction_period, measure_name_new, measure_aggregation(measure_name_new, df)]
                elif 'fscore' in measure_name:
                    for measure in ['f1_micro','precision_micro','recall_micro']:
                        measure_name_new = measure_name.replace('fscore',measure)
                        results.loc[len(results) + 1] = [community_id, community_size, model, train_len, prediction_period, measure_name_new, measure_aggregation(measure_name_new, df)]
                elif 'contagion_fractions_diff' in measure_name:
                    for measure in ['contagion_fractions_diff_MAE', 'contagion_fractions_diff_RMSE']:
                        measure_name_new = measure
                        results.loc[len(results) + 1] = [community_id, community_size, model, train_len, prediction_period, measure_name_new, measure_aggregation(measure_name_new, df)]
    return results


def proceed_ic_model(history):
    results = df_empty(
        columns=['community_id', 'community_size', 'model', 'train_len', 'prediction_period', 'measure_name',
                 'measure_value'],
        dtypes=['int', 'int', 'str', 'int', 'int', 'str', 'float', 'float','str'])
    model = 'independent-cascade'
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
                    for measure in ['f1_micro','precision_micro','recall_micro','f1_average_micro', 'f1_reversed_micro']:
                        measure_name_new = measure_name.replace('fscore',measure)
                        results.loc[len(results) + 1] = [community_id, community_size, model, train_len, prediction_period, measure_name_new, measure_aggregation(measure_name_new, df)]
                elif 'fscore' in measure_name:
                    for measure in ['f1_micro','precision_micro','recall_micro']:
                        measure_name_new = measure_name.replace('fscore',measure)
                        results.loc[len(results) + 1] = [community_id, community_size, model, train_len, prediction_period, measure_name_new, measure_aggregation(measure_name_new, df)]
                elif 'contagion_fractions_diff' in measure_name:
                    for measure in ['contagion_fractions_diff_MAE', 'contagion_fractions_diff_RMSE']:
                        measure_name_new = measure
                        results.loc[len(results) + 1] = [community_id, community_size, model, train_len,
                                                         prediction_period, measure_name_new,
                                                         measure_aggregation(measure_name_new, df)]
    return results


models = ['correlated-linear-dynamic-threshold','correlated-linear-threshold','linear-dynamic-threshold','linear-threshold','linear-threshold-random','linear-threshold-random-single-theta']


aprun = ParallelExecutor(n_jobs=len(models))

if __name__ == '__main__':
    list_of_dataframes = aprun(bar='txt')(delayed(proceed_normal_model)(model, '86400', 10) for model in models)
    list_of_dataframes.append(proceed_ic_model(10))
    final_data_frame = pd.concat(list_of_dataframes, ignore_index=True, sort=False)
    final_data_frame.to_pickle('/nfs/maciej/mcdoi/paper/results_dataframes/micro_averaging_dataframe.pickle')



