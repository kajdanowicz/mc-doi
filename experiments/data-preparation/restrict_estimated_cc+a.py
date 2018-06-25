import sys
import os
sys.path.append('/home/maciek/pyCharmProjects/mc-doi')
import operator

directory = '/datasets/mcdoi/louvain/'
min_num_users = 200

estimated_cc_a_file = directory + 'estimated_cc+a'
with open(estimated_cc_a_file, 'r', encoding='utf-8') as file:
    estimated_cc_a = file.readlines()
estimated_cc_a = [x.strip() for x in estimated_cc_a]

for e in estimated_cc_a:
    print(e.split('/')[3].split('_')[2])