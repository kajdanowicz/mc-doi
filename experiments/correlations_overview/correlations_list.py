import pandas as pd
import pickle
from tqdm import tqdm as tqdm

with open('/nfs/maciej/mcdoi/louvain/estimated_cc+a', 'r', encoding='utf-8') as f:
    estimated = f.readlines()
estimated = [x.strip() for x in estimated]

from multiprocessing import Pool

def work_sub(string):
    container = []
    if string.split('_')[-1] == '30':
        with open(string+'/contagion.pickle', 'rb') as f:
            matrix = pickle.load(f)
        with open(string+'/contagion_dict.pickle', 'rb') as f:
            d = pickle.load(f)
        for key, value in d.items():
            for key1, value1 in d.items():
                if value<value1:
                    container.append((key,key1,str(matrix[value, value1]),string.split('_')[-3],string.split('_')[-2].split('/')[0]))
        with open('/nfs/maciej/mcdoi/corrs/correlations_'+string.split('_')[-3]+'.pickle', 'wb') as f:
            pickle.dump(container, f)

params = estimated

n_proc = 20

use_parallel = True
if use_parallel:
    pool = Pool(processes=n_proc, maxtasksperchild=1)
    try:
        l = list(tqdm(pool.imap( work_sub, params, chunksize=1 ), total=len(params)))
        pool.close()
    except KeyboardInterrupt:
        print('got ^C while pool mapping, terminating the pool')
        pool.terminate()
        print('pool is terminated')
    except Exception as e:
        print('got exception: %r, terminating the pool' % (e,))
        pool.terminate()
        print('pool is terminated')
    finally:
        pool.join()
else:
    l = [work_sub(param) for param in params]