import sys
import os
sys.path.append('/home/maciek/pyCharmProjects/mc-doi')
from datetime import datetime
import logging
from model.multi_contagion_models import MultiContagionDynamicThresholdModel as MCDOI
from data.data import Data
from model.results import Results
import pickle
from tqdm import tqdm
import numpy as np
import pandas as pd
from data.data import Data
from model.parameters import ContagionCorrelation, Adjacency



directory = '/datasets/mcdoi/louvain/'

# print(next(os.walk(directory))[1]) # all subdirectories in directory

batch_sizes = [60, 3600, 43200, 86400, 604800]


def write_to_logger(args):
    logging.basicConfig(filename='./' + datetime.now().strftime("%Y-%m-%d_%H_%M_%S") + '.log', level=logging.DEBUG,
                        format='%(asctime)s %(levelname)s %(name)s %(message)s')
    logger = logging.getLogger(__name__)
    logger.error(args, exc_info=True)


def send_email():
    import smtplib
    content = ("Done!")
    mail = smtplib.SMTP('smtp.gmail.com', 587)
    mail.ehlo()
    mail.starttls()
    mail.login('python2266@gmail.com', 'GXjj5ahH')
    mail.sendmail('python2266@gmail.com', 'falkiewicz.maciej@gmail.com', content)
    mail.close()


def save_results(result: Results, dir, num_predictions):
    for iter in range(num_predictions):
        matrix = result.get_result(iter).matrix
        file_name = dir + 'result_' + str(iter) + '.pickle'
        os.makedirs(os.path.dirname(file_name), exist_ok=True)
        with open(file_name, 'wb') as file:
            pickle.dump(matrix, file)

def save_parameters(m: MCDOI, dir):
    file_name = dir + 'adjacency.pickle'
    os.makedirs(os.path.dirname(file_name), exist_ok=True)
    with open(file_name, 'wb') as adjacency_file:
        pickle.dump(m.adjacency.matrix, adjacency_file)
    file_name = dir + 'contagion.pickle'
    os.makedirs(os.path.dirname(file_name), exist_ok=True)
    with open(file_name, 'wb') as contagion_file:
        pickle.dump(m.contagion_correlation.matrix, contagion_file)
    file_name = dir + 'threshold.pickle'
    os.makedirs(os.path.dirname(file_name), exist_ok=True)
    with open(file_name, 'wb') as threshold_file:
        pickle.dump(m.thresholds.matrix, threshold_file)


def estimate_and_predict(d, dir, batch_type, batch_size, num_predictions):
    try:
        m = MCDOI()
        m.fit(d, batch_type = batch_type, batch_size = batch_size)
        result = m.predict(num_predictions)
        save_results(result, dir + batch_type + '/size_' + str(batch_size) + '/', num_predictions)
        save_parameters(m, dir + batch_type + '/size_' + str(batch_size) + '/')
    except Exception as err:
        write_to_logger(err.args)
        print(err.args)
        exit(1)
    finally:
        send_email()


for dataset in ['louvain_46_720']:#tqdm(next(os.walk(directory))[1]):
    dir = directory + dataset
    edges = pd.read_csv(dir+'/edges')
    for history_length in tqdm(np.arange(1,31,1)):
        dir = directory + dataset +'/history_'+str(history_length)
        event_log = pd.read_csv(dir+'/event_log')
        d = Data()
        d.load_data_data_frame(event_log,edges)
        cc = ContagionCorrelation()
        cc.estimate(d)
        print(cc.num_contagions_)
        a = Adjacency()
        a.estimate(d)
        print(a.num_users_)



    # d = Data()
    # d.load_data(dir)
    # if d.num_contagions <= 25000:
    #     for batch_size in tqdm(batch_sizes[0:1]):
    #         estimate_and_predict(d, dir, 'time', batch_size, 3)
    # else:
    #     print('Number of contagions in "' + dataset + '" is equal to ' + str(d.num_contagions) + ', it is too much.')
    # print(dataset + ' done!')
