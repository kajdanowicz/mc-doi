class tVector():

    def __init__(self):
        self.vector=None
        self.numUsers=None

    def estimate(self,data):
        #TODO Implement
        indykatory_est = []
        I = np.full((data.numUsers, data.numContagions), False, dtype=bool)
        for i in range(history):
            for index, row in event_log[event_log['ts'] == i].iterrows():
                I[row['userNEW'], row['tagID']] = True
            indykatory_est.append(I)
            I = copy.deepcopy(I)