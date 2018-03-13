import sys
from uuid import getnode as get_mac
if get_mac() == 2485377892363:
    sys.path.append('/home/maciek/pyCharmProjects/mc-doi')

import numpy as np
import scipy
import logging
from datetime import datetime
import pickle

import pandas as pd

from model.model import model
from data.data import data
import config.config as config

from model.tMatrix import tMatrix
from model.aMatrix import aMatrix
from model.ccMatrix import ccMatrix

mode = ''

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
        d = data()
        d.loadData(directory)
        d.addGraph()
        d.toPickle(directory)
        print('Pickle dumped')
        # d=data.fromPickle(directory)
        d.sample(0.01)
        print(d.numContagions)
        cm=ccMatrix()
        cm.estimate(d)
        # am=aMatrix.aMatrix()
        # am.estimate(d)
        # m=model.model()
        # m.fit(d,'volume',100)
    except Exception as err:
        writeToLogger(err.args)
        exit(1)
    finally:
        sendEmail()

def main1():
    try:
        d = data.data()
        d.loadData(directory)
        d.restrictEventLogMinOccurences(40000)
        d.eventLog.to_csv('restrictedEventLog',header=False,index=False)
        d.edges.to_csv('restrictedEdges', header=False, index=False)
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
    d=data.data()
    d.loadData(directory=directory)
    u=d.deleteUsers(userList)
    print(u.values())

def testLoadDataFile(directory):
    # passed
    d=data.data()
    return d.loadDataFile(directory)

def testLoadDataDataFrame(eventLogDF,edgesDF):
    d=data.data()
    return d.loadDataDataFrame(eventLogDF,edgesDF)

def testRestrictEventLogMinOccurences(directory, minOccurs = 40000):
    # TODO Design this test
    d=data.data()
    d.loadData(directory=directory)
    print('Before restriction:',d.numEvents,'numContagions:',d.numContagions)
    d.restrictEventLogMinOccurences(minOccurs)
    print('After restriction:', d.numEvents,'numContagions:',d.numContagions)

def testEstimateContagionCorrelationMatrix(directory,minOccurs=40000):
    # TODO Design this test
    m=model.model()
    d=data.data()
    d.loadData(directory)
    d.restrictEventLogMinOccurences(minOccurs)
    m.estimateContagionCorrelationMatrix(d)
    print(m.contagionCorrelationMatrix)
    test(m.verifyContagionCorrelationMatrixSymetry())

def testCreationOfRandomCorrelationMatrix(size=100):
    cm = model.CorrelationMatrix()
    print(cm.generateRandomCorrelationMatrix(size))

def testSymetry(cm):
    print(cm.testSymetry())

def testLoadData(fileName):
    frame = events.EventsData(fileName=fileName)
    print(frame.data.shape)

def testEstimateCorrelationMatrixFromData(fileName, minOccur = 10):
    frame = events.EventsData(fileName=fileName)
    cm = model.CorrelationMatrix()
    cm.estimateCorrelationMatrixFromData(frame.getEventsMinOccurences(minOccur))
    print(cm.correlationMatrix)
    testSymetry(cm)