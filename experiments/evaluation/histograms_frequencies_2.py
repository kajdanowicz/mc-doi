import sys
import os
sys.path.append('/home/maciek/pyCharmProjects/mc-doi')
import pickle
from tqdm import tqdm
import numpy as np
import pandas as pd
import csv
from matplotlib import pyplot as plt

directory = '/nfs/maciej/mcdoi/louvain/'

def histogram(list_of_directories,batch_size):
    abs_diffs = dict()
    org = dict()
    pred = dict()
    for i in range(7):
        abs_diffs[i] = []
        org[i] = []
        pred[i] = []
    for d in list_of_directories:
        i = int(d[-1])
        with open(d, 'r', encoding='utf-8') as file:
            spamreader = csv.reader(file, delimiter=',', quotechar='"')
            for row in spamreader:
                abs_diffs[i].append(np.absolute(float(row[1]) - float(row[2])))
                org[i].append(float(row[1]))
                pred[i].append(float(row[2]))
    for i in range(7):
        plt.figure(figsize=(18, 6))
        plt.subplot(1,3,1)
        plt.hist(abs_diffs[i], bins=50)
        plt.title('Absolute differences')
        plt.subplot(1,3,2)
        plt.hist(org[i], bins=50)
        plt.title('Fractions from data')
        plt.subplot(1,3,3)
        plt.hist(pred[i], bins=50)
        plt.title('Fractions from pred')
        plt.savefig(directory+'histograms/fractions/batch_'+str(batch_size)+'_pred_'+str(i)+'.png', dpi=72)
        plt.close('all')
        plt.figure()
        plt.scatter(x=org[i],y=pred[i], marker='o')
        plt.plot([(0,0),(1,1)],'r')
        plt.xlabel('Freq. from data')
        plt.ylabel('Freq. from pred')
        plt.savefig(directory+'histograms/fractions/scatter_batch_'+str(batch_size)+'_pred_'+str(i)+'.png', dpi=72)
        plt.close('all')


if __name__ == '__main__':
    for batch_size in tqdm([3600, 43200, 86400, 604800]):
        with open(directory + 'frequencies/frequencies_' + str(batch_size), 'r', encoding='utf-8') as file:
            e = file.readlines()
        evaluated = set([x.strip() for x in e])
        histogram(evaluated,batch_size)
