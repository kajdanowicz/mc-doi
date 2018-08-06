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

directory = '/nfs/maciej/mcdoi/louvain/'

num_users = 100

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
                    abs_diffs[i].append(np.absolute(float(row[1]) - float(row[2])))
                    org[i].append(float(row[1]))
                    pred[i].append(float(row[2]))
    for i in range(7):
        os.makedirs(directory + 'histograms/fractions_diff_' + str(num_users), exist_ok=True)
        plt.figure()
        ax = sns.kdeplot(org[i], pred[i], shade=True)
        plt.xlabel('Inc. freq. from data')
        plt.ylabel('Inc. freq. from pred')
        plt.savefig(directory+'histograms/fractions_diff_' + str(num_users)+'/kde_batch_'+str(batch_size)+'_pred_'+str(i)+'.png', dpi=72)
        plt.close('all')


if __name__ == '__main__':
    for batch_size in tqdm([3600, 43200, 86400, 604800]):
        with open(directory + 'frequencies/fractions_diff_' + str(batch_size), 'r', encoding='utf-8') as file:
            e = file.readlines()
        evaluated = set([x.strip() for x in e])
        histogram(evaluated,batch_size)
