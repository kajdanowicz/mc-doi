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
    abs_diffs[0] = []
    abs_diffs[1] = []
    abs_diffs[2] = []
    org = dict()
    org[0] = []
    org[1] = []
    org[2] = []
    for d in list_of_directories:
        for i in range(3):
            if batch_size==604800:
                if os.path.isfile(d+'/frequencies_'+str(i)):
                    with open(d+'/frequencies_'+str(i), 'r', encoding='utf-8') as file:
                        spamreader = csv.reader(file, delimiter=',', quotechar='"')
                        for row in spamreader:
                            abs_diffs[i].append(np.absolute(float(row[1])-float(row[2])))
                            org[i].append(float(row[1]))
            else:
                with open(d + '/frequencies_' + str(i), 'r', encoding='utf-8') as file:
                    spamreader = csv.reader(file, delimiter=',', quotechar='"')
                    for row in spamreader:
                        abs_diffs[i].append(np.absolute(float(row[1]) - float(row[2])))
                        org[i].append(float(row[1]))
    for i in range(3):
        plt.figure()
        plt.subplot(1,2,1)
        plt.hist(abs_diffs[i], bins=50)
        plt.title('Absolute differences')
        plt.subplot(1,2,2)
        plt.hist(org[i], bins=50)
        plt.title('Frequencies from data')
        plt.savefig(directory+'histograms/batch_'+str(batch_size)+'_pred_'+str(i)+'.png', dpi=300)


if __name__ == '__main__':
    for batch_size in tqdm([3600, 43200, 86400, 604800]):
        with open(directory + 'frequencies/frequencies_' + str(batch_size), 'r', encoding='utf-8') as file:
            e = file.readlines()
        evaluated = set([x.strip() for x in e])
        histogram(evaluated,batch_size)