import pandas as pd
import pickle
import os
from tqdm import tqdm

directory = '/nfs/maciej/mcdoi/corrs/'



for filename in tqdm(os.listdir(directory)):
    with open(directory+filename, 'rb') as f:
        list_of_tuples = pickle.load(f)
    for tuple in list_of_tuples:
        with open(directory+'correlations', 'a+', encoding='utf-8') as f:
            f.write(tuple[0]+','+tuple[1]+','+str(tuple[2])+','+tuple[3]+','+tuple[4]+'\n')