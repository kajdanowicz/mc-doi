import pandas as pd
import numpy as np
from collections import defaultdict
from matplotlib import pyplot as plt
from tqdm import tqdm
directory = '/nfs/maciej/mcdoi/louvain/'

event_log = pd.read_csv('/nfs/maciej/twitter/Prediction_of_Viral_Memes_on_Twitter/event_log', header=None)
event_log.columns = ['ts', 'user', 'contagion']
num_users = len(event_log['user'].unique())
event_log = event_log.drop_duplicates(subset=['contagion', 'user'], keep='first')

start_time = 1332565200
end_time = 1335416399
duration_24h_in_sec = 60 * 60 * 24
time_grid = np.arange(start_time + duration_24h_in_sec, end_time + duration_24h_in_sec, duration_24h_in_sec)

d = defaultdict(lambda: 0)

for count, time in tqdm(enumerate(time_grid,1)):
    e = event_log[event_log['ts'] <= time].groupby(by=['contagion']).count()['ts']
    diffs = []
    non_zero = 0
    tags = 0
    for tag in e.index.unique():
        diffs.append((e[tag]-d[tag])/num_users)
        tags += 1
        if e[tag]-d[tag]>0:
            non_zero+=1
        d[tag] = e[tag]
    plt.figure(figsize=(12,6))
    plt.hist(diffs, range=(0,1), bins=50)
    plt.axvline(np.mean(diffs), color='r', linestyle='dashed', linewidth=1)
    plt.axvline(np.median(diffs), color='g', linestyle='dashed', linewidth=1)
    plt.title('Fractions increase in day '+str(count)+', ' + str(non_zero) + ' increases out of ' + str(tags) + ' tags')
    plt.savefig(directory + 'data-overview/fractions_diff/day_' + str(count)+ '.png', dpi=72)
    plt.close('all')
    plt.figure(figsize=(12,6))
    plt.hist(diffs, range=(0,max(diffs)), bins=50)
    plt.axvline(np.mean(diffs), color='r', linestyle='dashed', linewidth=1)
    plt.axvline(np.median(diffs), color='g', linestyle='dashed', linewidth=1)
    plt.title('Fractions increase in day '+str(count)+', ' + str(non_zero) + ' increases out of ' + str(tags) + ' tags')
    plt.savefig(directory + 'data-overview/fractions_diff/day_' + str(count)+ '_with_max.png', dpi=72)
    plt.close('all')
