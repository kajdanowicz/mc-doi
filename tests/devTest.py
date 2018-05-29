import sys
from uuid import getnode as get_mac

if get_mac() == 2485377892363:
    sys.path.append('/home/maciek/pyCharmProjects/mc-doi')

import logging
from datetime import datetime
import pickle
import numpy as np

from model.multi_contagion_models import MultiContagionDynamicThresholdModel as MCDOI
from data.data import Data
import config.config as config

from model.parameters import Threshold
from model.parameters import Adjacency
from model.parameters import ContagionCorrelation

mode = 'Testing'

if get_mac() == 2485377892363:
    directory = config.remote['directory' + mode]
else:
    directory = config.local['directory' + mode]


def writeToLogger(args):
    logging.basicConfig(filename='./' + datetime.now().strftime("%Y-%m-%d_%H_%M_%S") + '.log', level=logging.DEBUG,
                        format='%(asctime)s %(levelname)s %(name)s %(message)s')
    logger = logging.getLogger(__name__)
    logger.error(args, exc_info=True)


def send_email():
    if get_mac() == 2485377892363:
        import smtplib
        content = ("Done!")
        mail = smtplib.SMTP('smtp.gmail.com', 587)
        mail.ehlo()
        mail.starttls()
        mail.login('python2266@gmail.com', 'GXjj5ahH')
        mail.sendmail('python2266@gmail.com', 'falkiewicz.maciej@gmail.com', content)
        mail.close()

def main():
    directory = '/datasets/mcdoi/louvain/louvain_46_720/'
    try:
        d = Data()
        d.load_data(directory)
        m = MCDOI()
        m.fit(d, batch_type = 'time', batch_size = 86400)
        m.predict(3) #predict(3) zwraca 63 aktywacje
    except Exception as err:
        writeToLogger(err.args)
        print(err.args)
        exit(1)
    finally:
        send_email()


if __name__ == "__main__":
    main()
