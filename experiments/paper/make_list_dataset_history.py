datasets = ['99_5','55_123','101_7','48_2234', '46_720','53_502','31_462','54_222','80_12','18_1819','24_2135','41_2265','42_2430','23_2536','16_2685','28_3065','11_3357','33_3824','8_3955','29_4554','37_4648','32_4904','26_2878','12_2888','50_4630']
#
file_directory_name = '/nfs/maciej/mcdoi/paper/estimated_cc+a'

# datasets = ['99_5']
histories = [10]#,11,12,13,14,15,16,17,18,19,20]
open(file_directory_name, 'w', encoding='utf-8').close()
for s in datasets:
    for history in histories:
        path = '/nfs/maciej/mcdoi/louvain/louvain_'+s+'/history_'+str(history)
        with open(file_directory_name, 'a+', encoding='utf-8') as file:
            file.write(path+ '\n')