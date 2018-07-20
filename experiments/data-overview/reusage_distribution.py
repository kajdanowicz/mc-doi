import pandas as pd
import numpy as np
from collections import defaultdict
from matplotlib import pyplot as plt
from tqdm import tqdm
directory = '/nfs/maciej/mcdoi/louvain/'

event_log = pd.read_csv('/nfs/maciej/twitter/Prediction_of_Viral_Memes_on_Twitter/event_log', header=None)
event_log.columns = ['ts', 'user', 'contagion']
num_users = len(event_log['user'].unique())
# event_log = event_log.drop_duplicates(subset=['contagion', 'user'], keep='first')

start_time = 1332565200
end_time = 1335416399
duration_24h_in_sec = 60 * 60 * 24
time_grid = np.arange(start_time + duration_24h_in_sec, end_time + duration_24h_in_sec, duration_24h_in_sec)

d = defaultdict(lambda: 0)

for count, time in tqdm(enumerate(time_grid,1)):
    e = event_log[event_log['ts'] <= time].groupby(by=['contagion','user']).count()['ts']
    usage = list(e['ts'].values)
    plt.figure(figsize=(12,6))
    plt.hist(usage, range=(1, max(usage)), bins=50)
    plt.axvline(np.mean(list(usage)), color='r', linestyle='dashed', linewidth=1)
    plt.axvline(np.median(list(usage)), color='g', linestyle='dashed', linewidth=1)
    plt.title('Count of (contagion,user) pairs in day '+str(count))
    plt.savefig(directory + 'data-overview/reusage/day_' + str(count)+ '_with_max.png', dpi=72)
    plt.close('all')
