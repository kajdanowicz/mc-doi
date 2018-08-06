import sys
import os
sys.path.append('/home/maciek/pyCharmProjects/mc-doi')
import pickle
from tqdm import tqdm
import numpy as np
import pandas as pd
import csv
from matplotlib import pyplot as plt
import seaborn as sns; sns.set(color_codes=True)
from scipy import stats
from sklearn.metrics import mean_squared_error

directory = '/nfs/maciej/mcdoi/dynamic-linear-threshold/'

num_users = 100

threshold = 0

def mse(x, y):
    return mean_squared_error(x, y)

def histogram(list_of_directories,batch_size):
    abs_diffs = dict()
    org = dict()
    pred = dict()
    for i in range(7):
        abs_diffs[i] = []
        org[i] = []
        pred[i] = []
    for d in list_of_directories:
        if int(d.split('/')[5].split('_')[2]) >=num_users:
            i = int(d[-1])
            with open(d, 'r', encoding='utf-8') as file:
                spamreader = csv.reader(file, delimiter=',', quotechar='"')
                for row in spamreader:
                    if (float(row[1])!=threshold) and (float(row[2])!=threshold):
                        org[i].append(float(row[1]))
                        pred[i].append(float(row[2]))
    for i in range(7):
        if len(org[i])>=2:
            os.makedirs(directory + 'histograms/fractions_diff_' + str(num_users), exist_ok=True)
            plt.figure()
            ax = sns.jointplot(org[i], pred[i], kind='reg', stat_func=mse)
            plt.xlabel('Inc. freq. from data')
            plt.ylabel('Inc. freq. from pred')
            plt.savefig(directory+'histograms/fractions_diff_' + str(num_users)+'/jointplot_batch_'+str(batch_size)+'_pred_'+str(i)+'.png', dpi=72)
            plt.close('all')


if __name__ == '__main__':
    for batch_size in tqdm([86400, 604800]):  # tqdm([3600, 43200, 86400, 604800])
        with open(directory + 'frequencies/fractions_diff_' + str(batch_size), 'r', encoding='utf-8') as file:
            e = file.readlines()
        evaluated = set([x.strip() for x in e])
        histogram(evaluated,batch_size)