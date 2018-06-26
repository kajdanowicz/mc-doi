import sys
import os
sys.path.append('/home/maciek/pyCharmProjects/mc-doi')
import pickle
from tqdm import tqdm
import numpy as np
import pandas as pd
import csv

directory = '/datasets/mcdoi/louvain/'

evaluated = set()
for batch_size in [3600, 43200, 86400, 604800]:
    with open(directory + 'frequencies/frequencies_'+str(batch_size), 'r', encoding='utf-8') as file:
        e = file.readlines()
    evaluated.update([x.strip() for x in e])

def histogram(list_of_directories):
    abd_diffs = []
    for directory in list_of_directories:
        with open(directory, 'r', encoding='utf-8', newline='') as file:
            spamreader = csv.reader(file, delimiter=',', quotechar='|')
            for row in spamreader:
                print(row)

if __name__ == '__main__':
    for batch_size in [3600, 43200, 86400, 604800]:
        with open(directory + 'frequencies/frequencies_' + str(batch_size), 'r', encoding='utf-8') as file:
            e = file.readlines()
        evaluated = set([x.strip() for x in e])
        histogram(evaluated)
