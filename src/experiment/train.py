import sys
sys.path.append('../')
import AE, VAE, IForest, OCSVM, Whisper, KMeans
from itertools import product
import os
import glob
from global_var import *


datasets = {
    # 'cicids_custom': [
    #     'Tuesday',
    #     'Wednesday',
    #     'Thursday',
    #     'Friday'
    # ],
    # 'toniot_custom': [
    #     'backdoor',
    #     'ddos',
    #     'dos',
    #     'injection',
    #     'mitm',
    #     'password',
    #     'runsomware',
    #     'scanning',
    #     'xss'
    # ],
    # 'cicids_improved': [
    #     'tuesday',
    #     'wednesday',
    #     'thursday',
    #     'friday'
    # ],
    'cse_improved': [
        'server1',
        'server2'
    ]
}

models = {
    'AE': AE,
    'VAE': VAE,
    'IForest': IForest,
    'OCSVM': OCSVM,
    # 'Whisper': Whisper,
    # 'KMeans': KMeans,
}


def clear_norm():
    files = glob.glob(os.path.join(NORMALIZER_DIR, '*.norm'))
    for file in files:
        try:
            os.remove(file)
        except:
            pass
        

def clear_result(m):
    if os.path.exists(os.path.join(RESULT_DIR, f'{m}.csv')):
        os.remove(os.path.join(RESULT_DIR, f'{m}.csv'))


if __name__ == '__main__':
    # clear_norm()
    for m in models:
        # clear_result(m)
        for d in datasets:
            for sub in datasets[d]:
                print(f'training {m} using {d}-{sub}')
                models[m].train_process(d, sub)
                print('\n')
                models[m].test_process(d, sub)
