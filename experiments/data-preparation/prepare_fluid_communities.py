import pandas as pd
import pickle
import os
from tqdm import tqdm

edges = pd.read_csv('/nfs/maciej/twitter/Prediction_of_Viral_Memes_on_Twitter/edges')

with open('/nfs/maciej/twitter/Prediction_of_Viral_Memes_on_Twitter/fluid_communities.pickle', 'rb') as handle:
    communities = pickle.load(handle)

event_log = pd.read_csv('/nfs/maciej/twitter/Prediction_of_Viral_Memes_on_Twitter/event_log')
event_log.columns = ['ts','user','contagion']

directory = '/datasets/mcdoi/fluid/'
counter = 0

for community in tqdm(communities):
    size = len(community)
    edges_file_name = directory + 'fluid_' + str(counter) + '_' + str(size) + '/edges'
    os.makedirs(os.path.dirname(edges_file_name), exist_ok=True)
    with open(edges_file_name, 'w', encoding='utf-8') as handle:
        row_mask_edges = edges.isin(community).all(1)
        edges[row_mask_edges].to_csv(handle, header=False, index=False)
    event_log_file_name = directory + 'fluid_' + str(counter) + '_' + str(size) + '/event_log'
    os.makedirs(os.path.dirname(event_log_file_name), exist_ok=True)
    with open(event_log_file_name, 'w', encoding='utf-8') as handle:
        row_mask_event_log = event_log['user'].isin(community)
        event_log[row_mask_event_log].to_csv(handle, header=False, index=False, columns = ['ts','user','contagion'])
    counter+=1
