import sys
from uuid import getnode as get_mac
if get_mac() == 2485377892363:
    sys.path.append('/home/maciek/pyCharmProjects/mc-doi')

import numpy as np
import logging
from datetime import datetime
import pickle

import pandas as pd

from model.model import Model
from data.data import Data
import config.config as config

from model.threshold import Threshold
from model.adjacency import Adjacency
from model.contagion_correlation import ContagionCorrelation

mode = 'Testing'

if get_mac() == 2485377892363:
    directory=config.remote['directory'+mode]
elif get_mac() == 215977245577188:
    directory = config.localPC['directory' + mode]
else:
    directory=config.local['directory'+mode]

def writeToLogger(args):
    logging.basicConfig(filename='./' + datetime.now().strftime("%Y-%m-%d_%H_%M_%S") + '.log', level=logging.DEBUG,
                        format='%(asctime)s %(levelname)s %(name)s %(message)s')
    logger = logging.getLogger(__name__)
    logger.error(args, exc_info=True)

def sendEmail():
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
    try:
        # d = Data.from_pickle('./')
        d = Data()
        d.load_data(directory)
        # d.sample_edges(fraction=0.001)
        # d.restrict_event_log(maxNumContagions=25)
        # print(d.num_users, d.num_contagions, d.numEvents)
        # d.addGraph()
        # d.construct_event_log_grouped()
        # validationData = d.prepare_test_data(fraction = 0.8)
        m = Model()
        m.fit(d, 'volume', 2)
        # m = Model.from_pickle('./')
        m.predict(3)
    except Exception as err:
        writeToLogger(err.args)
        print(err.args)
        exit(1)
    finally:
        sendEmail()

def main1():
    try:
        d = Data()
        d.load_data(directory)
        d.sample_edges(fraction=0.001)
        d.restrict_event_log(max_num_contagions=25)
        print(d.verify_users_correct(d.event_log, d.edges))
        d.add_graph()
        d.construct_event_log_grouped()
        d.toPickle('./')
    except Exception as err:
        writeToLogger(err.args)

if __name__ == "__main__":
    main()

def test(t):
    if t:
        print('Passed')
    else:
        print('Failed')

def testDeleteUsers(userList):
    d=Data.data()
    d.load_data(directory=directory)
    u=d.delete_users(userList)
    print(u.values())

def testLoadDataFile(directory):
    # passed
    d=Data.data()
    return d.load_data_file(directory)

def testLoadDataDataFrame(eventLogDF,edgesDF):
    d=Data.data()
    return d.load_data_data_frame(eventLogDF, edgesDF)

def testRestrictEventLogMinOccurences(directory, minOccurs = 40000):
    # TODO Design this tests
    d=Data.data()
    d.load_data(directory=directory)
    print('Before restriction:',d.numEvents,'num_contagions:',d.numContagions)
    d.restrict_event_log_min_occurences(minOccurs)
    print('After restriction:', d.numEvents,'num_contagions:',d.numContagions)

def testEstimateContagionCorrelationMatrix(directory,minOccurs=40000):
    # TODO Design this tests
    m=Model.model()
    d=Data.data()
    d.load_data(directory)
    d.restrict_event_log_min_occurences(minOccurs)
    m.estimate_contagion_correlation_matrix(d)
    print(m.contagionCorrelationMatrix)
    test(m.verifyContagionCorrelationMatrixSymetry())

def testCreationOfRandomCorrelationMatrix(size=100):
    cm = Model.CorrelationMatrix()
    print(cm.generateRandomCorrelationMatrix(size))

def testSymetry(cm):
    print(cm.testSymetry())

def testLoadData(fileName):
    frame = events.EventsData(fileName=fileName)
    print(frame.data.shape)

def testEstimateCorrelationMatrixFromData(fileName, minOccur = 10):
    frame = events.EventsData(fileName=fileName)
    cm = Model.CorrelationMatrix()
    cm.estimateCorrelationMatrixFromData(frame.getEventsMinOccurences(minOccur))
    print(cm.correlationMatrix)
    testSymetry(cm)