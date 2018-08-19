import pandas as pd
import networkx as nx
import community
import pickle

with open('/nfs/maciej/mcdoi/c4/active_follower_real_csv/active_follower_real.csv', 'r', encoding='utf-8') as f:
    df = pd.read_csv(f, header=None, names=['user1','user2'])

graph = nx.from_pandas_edgelist(df,source='user1', target='user2',create_using=nx.Graph())

partition_louvain=community.best_partition(graph)

partition_louvian_groups = [set() for index in range(max(partition_louvain.values())+1)]
for key,value in partition_louvain.items():
    partition_louvian_groups[value].add(key)

with open('/nfs/maciej/mcdoi/c4/louvain_communities.pickle', 'wb') as handle:
    pickle.dump(partition_louvian_groups, handle)



