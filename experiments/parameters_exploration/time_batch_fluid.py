import sys
import os
sys.path.append('/home/maciek/pyCharmProjects/mc-doi')
from datetime import datetime
import logging
from model.multi_contagion_models import MultiContagionDynamicThresholdModel as MCDOI
from data.data import Data
from model.results import Results
import pickle


directory = '/datasets/mcdoi/fluid/'

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
        with open(file_name, 'wb') as file:
            pickle.dump(matrix, file)

def save_parameters(m: MCDOI, dir):
    with open(dir, 'wb') as adjacency_file:
        pickle.dump(m.adjacency.matrix, adjacency_file)
    with open(dir, 'wb') as contagion_file:
        pickle.dump(m.contagion_correlation.matrix, contagion_file)
    with open(dir, 'wb') as threshold_file:
        pickle.dump(m.thresholds.matrix, threshold_file)


def estimate_and_predict(dir, batch_type, batch_size, num_predictions):
    try:
        d = Data()
        d.load_data(dir)
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


for dataset in next(os.walk(directory))[1][:1]:
    estimate_and_predict(directory+dataset+'/', 'time', 604800, 3)
