# datasets = ['99_5','55_123','101_7','48_2234']

# open('/nfs/maciej/mcdoi/cdlt-with-forgetting/estimated_cc+a', 'w', encoding='utf-8').close()
# for s in datasets:
#     path = '/nfs/maciej/mcdoi/louvain/louvain_'+s+'/history_20/time/size_604800'
#     with open('/nfs/maciej/mcdoi/cdlt-with-forgetting/estimated_t', 'a+', encoding='utf-8') as file:
#         file.write(path+ '\n')

datasets = ['55_123']
open('/nfs/maciej/mcdoi/cdlt-with-forgetting/estimated_cc+a', 'w', encoding='utf-8').close()
for s in datasets:
    path = '/nfs/maciej/mcdoi/louvain/louvain_'+s+'/history_10'
    with open('/nfs/maciej/mcdoi/cdlt-with-forgetting/estimated_cc+a', 'a+', encoding='utf-8') as file:
        file.write(path+ '\n')