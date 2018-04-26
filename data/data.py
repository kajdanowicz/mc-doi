import functools
import itertools
import pickle
from collections import defaultdict

import networkx as nx
import numpy as np
import pandas as pd


class Data:
    def __init__(self):

        self.event_log = None
        self.edges = None
        self.num_users = None
        self.num_contagions = None
        self.num_events = None
        self.contagion_id_dict = None
        self.graph = None

    def add_graph(self):
        if self.graph is None:
            self.graph = nx.from_pandas_edgelist(self.edges, 'user1', 'user2')

    def load_data_file(self, directory, file_names=('event_log', 'edges')):
        """

        :param directory:
        :param file_names:
        :return:
        """
        event_log_df = pd.read_csv(directory + file_names[0],header=None)
        event_log_df.columns = ['ts', 'user', 'contagion']
        edges_df = pd.read_csv(directory + file_names[1],header=None)
        edges_df.columns = ['user1', 'user2']
        if Data.verify_users_correct(event_log_df, edges_df):
            self.event_log = event_log_df
            self.edges = edges_df
            self.reindex_users()
            self.add_contagion_id()
            self.num_contagions = max(self.contagion_id_dict.values()) + 1
            self.num_events = self.event_log.shape[0]
            self.sort_data()
            return True
        else:
            return False
            # review

    def load_data_data_frame(self, event_log_df, edges_df):
        """

        :param pd.DataFrame event_log_df: Data frame containing event log
        :param pd.DataFrame edges_df: Data frame containing
        :return: If Data have been loaded successfully
        :rtype: bool
        """
        event_log_df.columns = ['ts', 'user', 'contagion']
        edges_df.columns = ['user1', 'user2']
        if Data.verify_users_correct(event_log_df, edges_df):
            self.event_log = event_log_df
            self.edges = edges_df
            self.reindex_users()
            self.add_contagion_id()
            self.num_contagions = max(self.contagion_id_dict.values()) + 1
            self.num_events = self.event_log.shape[0]
            self.sort_data()
            return True
        else:
            return False
            # review

    @staticmethod
    def verify_users_correct(event_log_df, edges_df):
        if set(event_log_df['user']).issubset(edges_df['user1'].append(edges_df['user2'])):
            return True
        else:
            return False

    # TODO find faster way
    # review

    def verify_data_correct(self):
        if not Data.verify_users_correct(self.event_log, self.edges):
            return False
        elif not self.num_events == self.event_log.shape[0]:
            return False
        elif not self.num_contagions == len(self.event_log['contagions'].unique()):
            return False
        elif not self.num_users == len(np.union1d(self.edges['user1'], self.edges['user2'])):
            return False
        else:
            return True

    # review

    def sort_data(self):
        if 'contagion_id' in self.event_log.columns:
            self.event_log.sort_values(by=['contagion_id', 'ts'], inplace=True)
        else:
            self.event_log.sort_values(by=['contagion', 'ts'], inplace=True)

    def load_data(self, directory=None, event_log_df=None, edges_df=None, file_names=('event_log', 'edges')):
        ''' Loads Data to class Data instance from the source that depends on given arguments'''
        if directory is not None:
            if not self.load_data_file(directory, file_names):
                return False
        elif (event_log_df is not None) and (edges_df is not None):
            if not self.load_data_data_frame(event_log_df, edges_df):
                return False
        else:
            return False
        print('Data loaded')
        return True
        # review

    def edge_exists(self, user1, user2):
        if self.edges[(self.edges['user1'].isin([user1, user2])) & (self.edges['user2'].isin([user1, user2]))].empty:
            return False
        else:
            return True

    def load_data_min_occurrence(self, min_occurs, directory=None, event_log_df=None, edges_df=None):
        """ Loads Data to class Data instance from the source that depends on given arguments
        Only contagions appearing in min_occurs events are loaded"""
        if not self.load_data(directory, event_log_df, edges_df):
            return False
        self.restrict_event_log_min_occurences(min_occurs)
        return True
        # review

    def restrict_event_log_min_occurences(self, min_occurs, max_num_contagions=None):
        """ Restricts events in self to that, which contains contagions appearing in the Data min_occurs times."""
        # TODO Use delete_contagions to obtain this
        temp = self.event_log.groupby(by='contagion').count().reset_index()[['contagion', 'ts']]
        if max_num_contagions is None:
            series = temp[(temp['ts'] >= min_occurs)]['contagion']
            self.event_log = self.event_log[self.event_log['contagion'].isin(series)]
            self.update_event_log()
        else:
            series = temp[(min_occurs <= temp['ts'])].sort_values(by='ts').iloc[:max_num_contagions]['contagion']
            self.event_log = self.event_log[self.event_log['contagion'].isin(series)]
            self.update_event_log()
            # review

    def restrict_event_log_max_occurences(self, max_occurs, max_num_contagions=None):
        """ Restricts events in self to that, which contains contagions appearing in the Data minOccurs times."""
        # TODO Use delete_contagions to obtain this
        temp = self.event_log.groupby(by='contagion').count().reset_index()[['contagion', 'ts']]
        if max_num_contagions is None:
            series = temp[(temp['ts'] <= max_occurs)]['contagion']
            self.event_log = self.event_log[self.event_log['contagion'].isin(series)]
            self.update_event_log()
        else:
            series = temp[(temp['ts'] <= max_occurs)].sort_values(by='ts', ascending=False).iloc[:max_num_contagions][
                'contagion']
            self.event_log = self.event_log[self.event_log['contagion'].isin(series)]
            self.update_event_log()
            # review

    def restrict_event_log_min_max_occurences(self, min_occurs, max_occurs):
        temp = self.event_log.groupby(by='contagion').count().reset_index()[['contagion', 'ts']]
        series = temp[(min_occurs <= temp['ts']) & (temp['ts'] <= max_occurs)]['contagion']
        self.event_log = self.event_log[self.event_log['contagion'].isin(series)]
        self.update_event_log()

    def restrict_event_log_max_num_contagions(self, max_num_contagions):
        temp = self.event_log.groupby(by='contagion').count().reset_index()[['contagion', 'ts']]
        series = temp.sort_values(by='ts', ascending=False).iloc[:max_num_contagions]['contagion']
        self.event_log = self.event_log[self.event_log['contagion'].isin(series)]
        self.update_event_log()

    def restrict_event_log(self, max_occurs=None, min_occurs=None, max_num_contagions=None):
        """ """
        # TODO Use delete_contagions to obtain this
        if (max_occurs is not None) and (min_occurs is None):
            self.restrict_event_log_max_occurences(max_occurs, max_num_contagions)
        elif (max_occurs is not None) and (min_occurs is not None) and (min_occurs <= max_occurs):
            self.restrict_event_log_min_max_occurences(min_occurs, max_occurs)
        elif (max_occurs is None) and (min_occurs is not None):
            self.restrict_event_log_min_occurences(min_occurs, max_num_contagions)
        elif (max_occurs is None) and (min_occurs is None) and (max_num_contagions is not None):
            self.restrict_event_log_max_num_contagions(max_num_contagions)
            # review

    def delete_users(self, user_list):
        self.edges.drop(self.edges[(self.edges['user1'].isin(user_list)) | (self.edges['user2'].isin(user_list))].index,
                        inplace=True)
        self.event_log.drop(self.event_log[self.event_log['user'].isin(user_list)].index, inplace=True)
        self.update_event_log()
        self.remove_from_graph(user_list)
        self.reindex_users()
        # review

    def keep_users(self, user_list):
        '''
        "private" method
        Does not check if contagions have at least one event
        '''
        row_mask = self.edges.isin(user_list).any(1)
        self.edges = self.edges[row_mask]
        # self.eventLog = self.eventLog[self.eventLog['user'].isin(user_list)]
        # self.numEvents = self.eventLog.shape[0]

    def drop_edge(self, edge=None, user_1=None, user_2=None):
        if (user_1 is None) & (user_2 is None):
            self.edges.drop(self.edges[((self.edges['user_1'] == edge[0]) & (self.edges['user_2'] == edge[1])) | (
                (self.edges['user_1'] == edge[1]) & (self.edges['user_2'] == edge[0]))].index, inplace=True)
        elif (edge is None) & (user_1 is not None) & (user_2 is not None):
            self.edges.drop(self.edges[(self.edges['user_1'].isin([user_1, user_2])) & (
                self.edges['user_2'].isin([user_1, user_2]))].index, inplace=True)
        else:
            return False
            # question Should an unconnected user be deleted?
            # review

    def delete_edges(self, edges_list):
        # question Should an unconnected user be deleted?
        for edge in edges_list:
            self.drop_edge(edge)
            # review

    def delete_contagions(self, contagion_list):
        self.event_log.drop(self.event_log[self.event_log['contagion'].isin(contagion_list)].index, inplace=True)
        self.num_events = self.event_log.shape[0]
        self.num_contagions = len(self.event_log['contagion'].unique())
        # review

    def delete_contagions_by_id(self, contagion_id_list):
        if 'contagion_id' in self.event_log.columns:
            self.event_log.drop(self.event_log[self.event_log['contagion_id'].isin(contagion_id_list)].index, inplace=True)
            self.num_events = self.event_log.shape[0]
            self.num_contagions = len(self.event_log['contagion_id'].unique())
            # review

    def add_contagion_id(self):
        if 'contagion_id' not in self.event_log.columns:
            t = defaultdict(functools.partial(next, itertools.count()))
            self.event_log = self.event_log.assign(contagion_id=self.event_log['contagion'].map(t))
            self.contagion_id_dict = t
        else:
            pass
            # review

    def construct_event_log_grouped(self):
        if 'event_id' not in self.event_log.columns:
            t = defaultdict(functools.partial(next, itertools.count()))
            self.event_log = self.event_log.assign(
                event_id=self.event_log.apply(lambda row: t[(row['user'], row['ts'])], axis=1, reduce=True))
            # review

    def to_csv(self, directory=''):
        self.event_log.to_csv(directory + 'event_log', header=False, index=False)
        self.edges.to_csv(directory + 'edges', header=False, index=False)

    def sample_events(self, fraction):
        self.event_log = self.event_log.sample(frac=fraction)
        self.update_event_log()
        self.num_events = self.event_log.shape[0]

    def sample_edges(self, fraction, number=None):
        if number is None:
            self.edges = self.edges.sample(frac=fraction)
        else:
            self.edges = self.edges.sample(frac=float(number) / self.edges.shape[0])
        self.keep_events_of_users(set(self.edges.user1.unique()).union(self.edges.user2.unique()))
        self.reindex_users()
        self.update_event_log()

    def toPickle(self, directory=''):
        pickle.dump(self, open(directory + 'Data.p', 'wb'))

    def restrictUsersToActive(self):
        activeUsers = self.event_log.user.unique()
        self.keep_users(activeUsers)
        if self.graph is not None:
            self.restrict_graph(activeUsers)
        self.reindex_users()

    def restrictUsersTo(self, userList):
        self.keep_users(userList)
        if self.graph is not None:
            self.restrict_graph(userList)
        self.keep_events_of_users(userList)
        self.reindex_users()

    def delete_events_of_users(self, userList):
        self.event_log.drop(self.event_log[self.event_log['user'].isin(userList)].index, inplace=True)

    def keep_events_of_users(self, userList):
        self.event_log = self.event_log[self.event_log.user.isin(userList)]

    def restrict_graph(self, userList):
        self.graph.remove_nodes_from(np.setdiff1d(self.graph.nodes(), userList))

    def remove_from_graph(self, user_list):
        self.graph.remove_nodes_from(user_list)

    def reindex_users_in_event_log(self, dictionary):
        self.event_log.user = self.event_log.user.map(dictionary)

    def reindex_contagion_id(self, dictionary):
        self.event_log = self.event_log.assign(contagion_id=self.event_log['contagion'].map(dictionary))

    def update_event_log(self):
        if 'contagion_id' in self.event_log.columns:
            t = defaultdict(functools.partial(next, itertools.count()))
            self.reindex_contagion_id(t)
            self.num_contagions = max(t.values()) + 1
            self.contagion_id_dict = t
        else:
            self.num_contagions = len(self.event_log.contagion.unique())
        self.num_events = self.event_log.shape[0]

    def reindex_users_in_graph(self, dictionary):
        self.graph = nx.relabel_nodes(self.graph, dictionary, copy=True)

    def reindex_users(self):
        t = defaultdict(functools.partial(next, itertools.count()))
        self.edges.user1 = self.edges.user1.map(t)
        self.edges.user2 = self.edges.user2.map(t)
        self.reindex_users_in_event_log(t)
        if self.graph is not None:
            self.reindex_users_in_graph(t)
        self.num_users = max(t.values()) + 1

    def prepare_test_data(self, fraction):
        min_time = self.event_log.ts.min()
        max_time = self.event_log.ts.max()
        stopping_time = float(max_time - min_time) * fraction + min_time
        tmp = self.event_log[self.event_log.ts > stopping_time]
        self.event_log = self.event_log[self.event_log.ts <= stopping_time]
        self.num_events = self.event_log.shape[0]
        return tmp

    @staticmethod
    def from_pickle(directory):
        return pickle.load(open(directory + 'Data.p', 'rb'))
