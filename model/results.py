import copy

class Results():

    def __init__(self):
        self.list_single_iter_results = []
        self.num_results = 0

    def add_result(self, result):
        self.list_single_iter_results.append(copy.copy(result))
        self.num_results += 1

    def get_result(self, result_num):
        # TODO Implement exceptions
        if result_num >= self.num_results:
            return False
        else:
            return self.list_single_iter_results[result_num]

class SingleIterResult():

    def __init__(self):
        self.matrix = None
        self.num_users = None
        self.num_contagions = None