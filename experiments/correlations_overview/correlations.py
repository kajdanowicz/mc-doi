import pandas as pd
import pickle
from tqdm import tqdm as tqdm

with open('/nfs/maciej/mcdoi/louvain/estimated_cc+a', 'r', encoding='utf-8') as f:
    estimated = f.readlines()
estimated = [x.strip() for x in estimated]

from multiprocessing import Pool

def work_sub(string):
    if string.split('_')[-1] == '30':
        with open(string+'/contagion.pickle', 'rb') as f:
            matrix = pickle.load(f)
        with open(string+'/contagion_dict.pickle', 'rb') as f:
            d = pickle.load(f)
        rev_d = {v: k for k, v in d.items()}
        for i in range(matrix.shape[0]):
            for j in range(i+1,matrix.shape[0]):
                with open('/nfs/maciej/mcdoi/correlations_reversed', 'a+', encoding='utf-8') as f:
                    f.write(rev_d[i]+','+rev_d[j]+','+str(matrix[i,j])+','+string.split('_')[-3]+','+string.split('_')[-2].split('/')[0]+'\n')
#                 df.append({'tag1': rev_d[i], 'tag2': rev_d[j], 'correlation': matrix[i,j], 'communityID': string.split('_')[-3], 'communitySize': string.split('_')[-2].split('/')[0]})

params = list(reversed(estimated))

n_proc = 12

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