import copy

class results():

    def __init__(self):
        self.list = []
        self.num_results = 0

    def add_result(self, result):
        self.list.append(copy.copy(result))
        self.num_results += 1

    def get_result(self, result_num):
        if result_num >= self.num_results:
            return False
        else:
            return self.list[result_num]