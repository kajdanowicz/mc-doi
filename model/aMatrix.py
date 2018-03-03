from collections import defaultdict

class aMatrix():
    def __init__(self):
        self.matrix=None
        self.numUsers=None

    def estimate(self,data):
        # TODO Implement
        # TODO Analyse uniqueness of (user, action)
        currentContagion=None
        a_u=defaultdict(lambda: 0)
        a_v2u = defaultdict(lambda: 0)
        a_vandu = defaultdict(lambda: 0)
        tau_vu = defaultdict(lambda: 0)
        data.addContagionID()
        currentTable = []
        for index, row in data.eventLog.iterrows():
            if row['contagionID'] != currentContagion:
                currentTable = []
                currentContagion = row['contagionID']
            a_u[row['user']]+=1
            #parents=[]
            for tuple in currentTable:
                if data.edgeExists(tuple[0],row['user']) & row['ts']>tuple[1]:
                    a_v2u[(tuple[0],row['user'])]+=1
                    tau_vu[(tuple[0],row['user'])]+=(row['ts']-tuple[1])
                    #parents.append(tuple(0))
                a_vandu[(tuple[0],row['user'])]+=1
                a_vandu[(row['user'],tuple[0])]+=1
            #for v in parents:
                #update credit_v,u
            currentTable.append((row['user'],row['ts']))
        print(a_u)
        print(a_v2u)





        pass