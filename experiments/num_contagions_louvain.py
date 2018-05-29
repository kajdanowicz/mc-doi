import pandas as pd
import os
import csv

directory = '/datasets/mcdoi/louvain/'

# Save number of unique contagions to files
# for dataset in next(os.walk(directory))[1]:
#     event_log = pd.read_csv(directory+dataset+'/event_log')
#     event_log.columns = ['ts','user','contagion']
#     file_name = directory + dataset + '/number_of_contagions_' + str(len(event_log.contagion.unique()))
#     os.makedirs(os.path.dirname(file_name), exist_ok=True)
#     with open(file_name, 'w', encoding='utf-8') as handle:
#         wr = csv.writer(handle, quoting=csv.QUOTE_ALL)
#         wr.writerow(event_log.contagion.unique())

# List number of unique contagions for each data set
for dataset in next(os.walk(directory))[1]:
    for file in next(os.walk(directory + dataset + '/'))[2]:
        if 'number_of_contagions' in file:
            print(dataset + ' - ' + file.split('_')[3])

