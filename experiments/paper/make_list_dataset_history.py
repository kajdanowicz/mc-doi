datasets = ['99_5','55_123','101_7','48_2234', '46_720','53_502','31_462','54_222','80_12']

file_directory_name = '/nfs/maciej/mcdoi/paper/estimated_cc+a'

# datasets = ['99_5']
histories = [10,20]
open(file_directory_name, 'w', encoding='utf-8').close()
for s in datasets:
    for history in histories:
        path = '/nfs/maciej/mcdoi/louvain/louvain_'+s+'/history_'+str(history)
        with open(file_directory_name, 'a+', encoding='utf-8') as file:
            file.write(path+ '\n')