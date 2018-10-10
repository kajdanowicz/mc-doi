import os
script_path = '/home/maciek/pyCharmProjects/mc-doi/experiments/paper/evaluation/evaluate_measures_reevaluate.py'
home_path = '/nfs/maciej/mcdoi/paper/'

models = ['correlated-linear-dynamic-threshold','correlated-linear-threshold','linear-dynamic-threshold','linear-threshold','linear-threshold-random','linear-threshold-random-dynamic','linear-threshold-random-dynamic-single-theta','linear-threshold-random-single-theta','correlated-linear-dynamic-threshold-continuous']

for model in models:
    for batch_size in ['3600','43200','86400','604800']:
        os.system('sudo python3 '+script_path+' '+home_path+model+'/evaluation/evaluated_'+batch_size+' '+model)
    # print('sudo python3 '+script_path+' '+home_path+model+'/predicted'+' '+model)