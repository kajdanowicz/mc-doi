import numpy as np
directory = '/nfs/maciej/mcdoi/'

models = ['louvain','correlated-linear-threshold', 'dynamic-linear-threshold', 'linear-threshold','linear-threshold-model','linear-threshold-model-14all','linear-threshold-random-dynamic','linear-threshold-random-dynamic-14all']

list_of_predicted = []
for model in models:
    model_directory = directory+model+'/'
    with open(model_directory+'predicted_7days', 'r', encoding='utf-8') as predicted:
        predicted = predicted.readlines()
    predicted = [x.strip() for x in predicted]
    list_of_predicted.append(predicted)

predicted_4all = list_of_predicted[0]
for predicted in list_of_predicted[1::]:
    predicted_4all = np.intersect1d(predicted_4all,predicted)

print(len(predicted_4all))

open(directory+'predicted_4all', 'w', encoding='utf-8').close()
for predicted in predicted_4all:
    with open(directory+'predicted_4all', 'a', encoding='utf-8') as file:
        file.write(predicted + '\n')