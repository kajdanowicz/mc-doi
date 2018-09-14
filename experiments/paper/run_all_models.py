import os
home_path = '/home/maciek/pyCharmProjects/mc-doi/experiments/paper/'
path_to_datasets = '/nfs/maciej/mcdoi/paper/estimated_cc+a'

models = ['correlated-linear-dynamic-threshold','correlated-linear-threshold','linear-dynamic-threshold','linear-threshold','linear-threshold-random','linear-threshold-random-dynamic','linear-threshold-random-dynamic-single-theta','linear-threshold-random-single-theta','correlated-linear-dynamic-threshold-continuous']

for model in models:
    os.system('sudo python3 '+home_path+model+'/estimate_thresholds_and_predict.py '+path_to_datasets+' '+model)
    # print('sudo python3 '+home_path+model+'/estimate_thresholds_and_predict.py '+path_to_datasets+' '+model)