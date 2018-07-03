import sys
import os
sys.path.append('/home/maciek/pyCharmProjects/mc-doi')
import operator

directory = '/nfs/maciej/mcdoi/louvain/'
min_num_users = 200

estimated_cc_a_file = directory + 'estimated_cc+a'
with open(estimated_cc_a_file, 'r', encoding='utf-8') as file:
    estimated_cc_a = file.readlines()
estimated_cc_a = [x.strip() for x in estimated_cc_a]

in_list_file = directory + 'estimated_cc+a_num_users'
with open(in_list_file, 'r', encoding='utf-8') as file:
    in_list = file.readlines()
in_list = [x.strip() for x in in_list]

num_users = 200

for e in estimated_cc_a:
    if e not in in_list:
        if int(e.split('/')[5].split('_')[2])>=num_users:
            with open(directory+'estimated_cc+a_num_users', 'a+', encoding='utf-8') as handle:
                handle.write(e + '\n')