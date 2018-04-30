import functools
import itertools
import pickle
from collections import defaultdict

import networkx as nx
import numpy as np
import pandas as pd


class Data:
    # TODO Finish refactoring. Done: user_1, user_2, time_stamp
    time_stamp = 'ts'
    user = 'user'
    contagion = 'contagion'
    contagion_id = 'contagion_id'
    user_1 = 'user_1'
    user_2 = 'user_2'
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
            self.graph = nx.from_pandas_edgelist(self.edges, Data.user_1, Data.user_2)

    def load_data_file(self, directory, file_names=('event_log', 'edges')):
        # TODO extract column names as class attributes?
        event_log_df = pd.read_csv(directory + file_names[0],header=None, names=[Data.time_stamp, Data.user, Data.contagion])
        edges_df = pd.read_csv(directory + file_names[1],header=None, names=[Data.user_1, Data.user_2])
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

    def load_data_data_frame(self, event_log_df: pd.DataFrame, edges_df: pd.DataFrame) -> bool:
        event_log_df.columns = [Data.time_stamp, Data.user, Data.contagion]
        edges_df.columns = [Data.user_1, Data.user_2]
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
        if set(event_log_df['user']).issubset(edges_df[Data.user_1].append(edges_df[Data.user_2])):
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
        elif not self.num_users == len(np.union1d(self.edges[Data.user_1], self.edges[Data.user_2])):
            return False
        else:
            return True

    # review

    def sort_data(self):
        if 'contagion_id' in self.event_log.columns:
            self.event_log.sort_values(by=[Data.contagion_id, Data.time_stamp], inplace=True)
        else:
            self.event_log.sort_values(by=[Data.contagion, Data.time_stamp], inplace=True)

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

    def edge_exists(self, user_1, user_2):
        if self.edges[(self.edges[Data.user_1].isin([user_1, user_2])) & (self.edges[Data.user_2].isin([user_1, user_2]))].empty:
            return False
        else:
            return True

    def load_data_min_occurrence(self, min_occurs, directory=None, event_log_df=None, edges_df=None):
        """ Loads Data to class Data instance from the source that depends on given arguments
        Only contagions appearing in min_occurs events are loaded"""
        if not self.load_data(directory, event_log_df, edges_df):
            return False
        self.restrict_event_log_min_occurrence(min_occurs)
        return True
        # review

    def restrict_event_log_min_occurrence(self, min_occurs, max_num_contagions=None):
        """ Restricts events in self to that, which contains contagions appearing in the Data min_occurs times."""
        # TODO Use delete_contagions to obtain this
        temp = self.event_log.groupby(by=Data.contagion).count().reset_index()[[Data.contagion, Data.time_stamp]]
        if max_num_contagions is None:
            series = temp[(temp[Data.time_stamp] >= min_occurs)][Data.contagion]
            self.event_log = self.event_log[self.event_log[Data.contagion].isin(series)]
            self.update_event_log()
        else:
            series = temp[(min_occurs <= temp[Data.time_stamp])].sort_values(by=Data.time_stamp).iloc[:max_num_contagions][Data.contagion]
            self.event_log = self.event_log[self.event_log[Data.contagion].isin(series)]
            self.update_event_log()
            # review

    def restrict_event_log_max_occurrence(self, max_occurs, max_num_contagions=None):
        """ Restricts events in self to that, which contains contagions appearing in the Data minOccurs times."""
        # TODO Use delete_contagions to obtain this
        temp = self.event_log.groupby(by=Data.contagion).count().reset_index()[[Data.contagion, Data.time_stamp]]
        if max_num_contagions is None:
            series = temp[(temp[Data.time_stamp] <= max_occurs)][Data.contagion]
            self.event_log = self.event_log[self.event_log[Data.contagion].isin(series)]
            self.update_event_log()
        else:
            series = temp[(temp[Data.time_stamp] <= max_occurs)].sort_values(by=Data.time_stamp, ascending=False).iloc[:max_num_contagions][
                Data.contagion]
            self.event_log = self.event_log[self.event_log[Data.contagion].isin(series)]
            self.update_event_log()
            # review

    def restrict_event_log_min_max_occurrence(self, min_occurs, max_occurs):
        temp = self.event_log.groupby(by=Data.contagion).count().reset_index()[[Data.contagion, Data.time_stamp]]
        series = temp[(min_occurs <= temp[Data.time_stamp]) & (temp[Data.time_stamp] <= max_occurs)][Data.contagion]
        self.event_log = self.event_log[self.event_log[Data.contagion].isin(series)]
        self.update_event_log()

    def restrict_event_log_max_num_contagions(self, max_num_contagions):
        temp = self.event_log.groupby(by=Data.contagion).count().reset_index()[[Data.contagion, Data.time_stamp]]
        series = temp.sort_values(by=Data.time_stamp, ascending=False).iloc[:max_num_contagions][Data.contagion]
        self.event_log = self.event_log[self.event_log[Data.contagion].isin(series)]
        self.update_event_log()

    def restrict_event_log(self, max_occurs=None, min_occurs=None, max_num_contagions=None):
        """ """
        # TODO Use delete_contagions to obtain this
        if (max_occurs is not None) and (min_occurs is None):
            self.restrict_event_log_max_occurrence(max_occurs, max_num_contagions)
        elif (max_occurs is not None) and (min_occurs is not None) and (min_occurs <= max_occurs):
            self.restrict_event_log_min_max_occurrence(min_occurs, max_occurs)
        elif (max_occurs is None) and (min_occurs is not None):
            self.restrict_event_log_min_occurrence(min_occurs, max_num_contagions)
        elif (max_occurs is None) and (min_occurs is None) and (max_num_contagions is not None):
            self.restrict_event_log_max_num_contagions(max_num_contagions)
            # review

    def delete_users(self, user_list):
        self.edges.drop(self.edges[(self.edges[Data.user_1].isin(user_list)) | (self.edges[Data.user_2].isin(user_list))].index,
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
            self.edges.drop(self.edges[((self.edges[Data.user_1] == edge[0]) & (self.edges[Data.user_2] == edge[1])) | (
                (self.edges[Data.user_1] == edge[1]) & (self.edges[Data.user_2] == edge[0]))].index, inplace=True)
        elif (edge is None) & (user_1 is not None) & (user_2 is not None):
            self.edges.drop(self.edges[(self.edges[Data.user_1].isin([user_1, user_2])) & (
                self.edges[Data.user_2].isin([user_1, user_2]))].index, inplace=True)
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

    def keep_contagions(self, contagion_list):
        self.event_log.drop(self.event_log[~self.event_log['contagion'].isin(contagion_list)].index, inplace=True)
        self.num_events = self.event_log.shape[0]
        self.num_contagions = len(self.event_log['contagion'].unique())

    def restrict_contagions_to_present(self):
        self.keep_contagions(self.event_log['contagion'])
        self.update_event_log()

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
                event_id=self.event_log.apply(lambda row: t[(row[Data.user], row[Data.time_stamp])], axis=1, reduce=True))
            # review

    def to_csv(self, directory=''):
        self.event_log.to_csv(directory + 'event_log', header=False, index=False, columns = [Data.time_stamp,Data.user,Data.contagion])
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
        self.keep_events_of_users(set(self.edges[Data.user_1].unique()).union(self.edges[Data.user_2].unique()))
        self.reindex_users()
        self.update_event_log()

    def to_pickle(self, directory=''):
        pickle.dump(self, open(directory + 'Data.p', 'wb'))

    def restrict_users_to_active(self):
        active_users = self.event_log.user.unique()
        self.keep_users(active_users)
        if self.graph is not None:
            self.restrict_graph(active_users)
        self.reindex_users()

    def restrict_users_to(self, user_list):
        self.keep_users(user_list)
        if self.graph is not None:
            self.restrict_graph(user_list)
        self.keep_events_of_users(user_list)
        self.reindex_users()

    def delete_events_of_users(self, userList):
        self.event_log.drop(self.event_log[self.event_log['user'].isin(userList)].index, inplace=True)

    def keep_events_of_users(self, userList):
        self.event_log = self.event_log[self.event_log.user.isin(userList)]
        self.num_events = self.event_log.shape[0]

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
        self.edges[Data.user_1] = self.edges[Data.user_1].map(t)
        self.edges[Data.user_2] = self.edges[Data.user_2].map(t)
        self.reindex_users_in_event_log(t)
        if self.graph is not None:
            self.reindex_users_in_graph(t)
        self.num_users = max(t.values()) + 1

    def prepare_test_data(self, fraction):
        min_time = self.event_log[Data.time_stamp].min()
        max_time = self.event_log[Data.time_stamp].max()
        stopping_time = float(max_time - min_time) * fraction + min_time
        tmp = self.event_log[self.event_log[Data.time_stamp] > stopping_time]
        self.event_log = self.event_log[self.event_log[Data.time_stamp] <= stopping_time]
        self.num_events = self.event_log.shape[0]
        return tmp

    @staticmethod
    def from_pickle(directory):
        return pickle.load(open(directory + 'Data.p', 'rb'))

    def get_neighbors(self,user: int) -> set:
        user_set = set()
        row_mask = self.edges.isin([user]).any(1)
        user_set.update(set(self.edges[row_mask][Data.user_1]))
        user_set.update(set(self.edges[row_mask][Data.user_2]))
        return user_set

    def snowball_sampling(self,initial_user_id: int) -> set:
        set_of_users = set()
        set_of_users.update(self.get_neighbors(initial_user_id))
        # temporary_set = set()
        # for user in set_of_users:
        #     temporary_set.update(self.get_neighbors(user))
        # set_of_users.update(temporary_set)
        return set_of_users

