import os

models = ['cdlt-with-forgetting']#['correlated-linear-threshold', 'dynamic-linear-threshold', 'linear-threshold','linear-threshold-model','linear-threshold-model-14all','linear-threshold-random-dynamic','linear-threshold-random-dynamic-14all']
batch_sizes = ['3600', '43200','86400','604800']
# 'hamming_diff_','contagion_hamming_','hamming_'
measures = ['contagion_jaccard_','contagion_fractions_diff_','contagion_fractions_','contagion_fscore_','contagion_fscore_diff_','fscore_','fscore_diff_','fractions_diff_', 'frequencies_','contagion_jaccard_diff_','jaccard_','jaccard_diff_']

for model in models:
    for batch_size in batch_sizes:
        directory = '/nfs/maciej/mcdoi/'+model+'/'
        for measure in measures:
            os.makedirs(directory+'frequencies/', exist_ok=True)
            open(directory+'frequencies/'+measure+batch_size, 'w', encoding='utf-8').close()