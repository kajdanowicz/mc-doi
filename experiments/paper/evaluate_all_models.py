import os
script_path = '/home/maciek/pyCharmProjects/mc-doi/experiments/paper/evaluation/evaluate_measures.py'
home_path = '/nfs/maciej/mcdoi/paper/'

# models = ['correlated-linear-dynamic-threshold','correlated-linear-threshold','linear-dynamic-threshold','linear-threshold','linear-threshold-random','linear-threshold-random-dynamic','linear-threshold-random-dynamic-single-theta','linear-threshold-random-single-theta','correlated-linear-dynamic-threshold-continuous']
models = ['linear-dynamic-threshold','linear-threshold','correlated-linear-dynamic-threshold','correlated-linear-threshold','linear-threshold-random','linear-threshold-random-dynamic','linear-threshold-random-dynamic-single-theta','linear-threshold-random-single-theta']

for model in models:
    os.system('sudo python3 '+script_path+' '+home_path+model+'/predicted'+' '+model)
    # print('sudo python3 '+script_path+' '+home_path+model+'/predicted'+' '+model)