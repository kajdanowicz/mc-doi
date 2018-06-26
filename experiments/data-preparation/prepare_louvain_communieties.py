import pandas as pd
import pickle
import os
from tqdm import tqdm
import networkx as nx

edges = pd.read_csv('/nfs/maciej/twitter/Prediction_of_Viral_Memes_on_Twitter/edges', header = None)
edges.columns = ['user1','user2']

with open('/nfs/maciej/twitter/Prediction_of_Viral_Memes_on_Twitter/louvain_communities.pickle', 'rb') as handle:
    communities = pickle.load(handle)

event_log = pd.read_csv('/nfs/maciej/twitter/Prediction_of_Viral_Memes_on_Twitter/event_log', header = None)
event_log.columns = ['ts','user','contagion']

directory = '/nfs/maciej/mcdoi/louvain/'
counter = 0

network = nx.from_pandas_edgelist(edges, 'user1', 'user2')

for community in tqdm(communities):
    community_network = network.subgraph(community)
    connected_component_community = sorted(nx.connected_components(community_network), key=len, reverse=True)[0]
    size = len(connected_component_community)
    edges_file_name = directory + 'louvain_' + str(counter) + '_' + str(size) + '/edges'
    os.makedirs(os.path.dirname(edges_file_name), exist_ok=True)
    with open(edges_file_name, 'w', encoding='utf-8') as handle:
        row_mask_edges = edges.isin(connected_component_community).all(1)
        edges[row_mask_edges].to_csv(handle, header=False, index=False)
    event_log_file_name = directory + 'louvain_' + str(counter) + '_' + str(size) + '/event_log'
    os.makedirs(os.path.dirname(event_log_file_name), exist_ok=True)
    with open(event_log_file_name, 'w', encoding='utf-8') as handle:
        row_mask_event_log = event_log['user'].isin(connected_component_community)
        event_log[row_mask_event_log].to_csv(handle, header=False, index=False, columns = ['ts','user','contagion'])
    counter+=1
