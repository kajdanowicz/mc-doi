import sys


model = list(sys.argv)[1]
directory = '/nfs/maciej/mcdoi/paper/'+model+'/'
sets_path = list(sys.argv)[2]
with open(sets_path, 'r', encoding='utf-8') as sets:
    sets = sets.readlines()
sets = [x.strip() for x in sets]

batch_sizes = [86400, 604800]

open(directory + 'predicted', 'w', encoding='utf-8').close()
with open(directory + 'predicted', 'a+', encoding='utf-8') as handle:
    for set in sets:
        for batch_size in batch_sizes:
            handle.write(set + '/' + 'time' + '/size_' + str(batch_size) + '\n')
            # print(set + '/' + 'time' + '/size_' + str(batch_size) + '\n')