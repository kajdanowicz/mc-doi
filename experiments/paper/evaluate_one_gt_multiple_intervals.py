import os
import sys
script_path = '/home/maciek/pyCharmProjects/mc-doi/experiments/paper/evaluation/evaluate_measures_one_gt.py'
home_path = '/nfs/maciej/mcdoi/paper/'

model = list(sys.argv)[1]

# models = ['correlated-linear-dynamic-threshold','correlated-linear-threshold','linear-dynamic-threshold','linear-threshold','linear-threshold-random','linear-threshold-random-dynamic','linear-threshold-random-dynamic-single-theta','linear-threshold-random-single-theta','correlated-linear-dynamic-threshold-continuous']
# models = ['linear-dynamic-threshold','linear-threshold','correlated-linear-dynamic-threshold','correlated-linear-threshold','linear-threshold-random','linear-threshold-random-dynamic','linear-threshold-random-dynamic-single-theta','linear-threshold-random-single-theta']

# intervals = [3600,7200,10800,21600,43200,86400,129600,172800,259200]
intervals = [86400, 172800, 259200, 345600, 432000, 518400, 604800]

for interval in intervals:
    # os.system('sudo python3 '+script_path+' '+home_path+model+'/predicted'+' '+model+' '+str(interval))
    os.system('python3 '+script_path+' '+home_path+model+'/predicted'+' '+model+' '+str(interval))
    # print('sudo python3 '+script_path+' '+home_path+model+'/predicted'+' '+model+' '+str(interval))