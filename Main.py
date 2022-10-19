import networkx as nx
import pickle
import random

from viz import *


Ve, Vp = [], []
for i in range(50):
    print (i)

    G = nx.read_gml('networks/recommend_group/reco' + str(i) + '.gml')
    # G = nx.read_gml('networks/Ecoli.gml')

    N = sorted(list(G.nodes()))
    Ra = 0.00000001
    # Ra = 0.0001

    R = {tuple([0, 0]): [], tuple([0, 1]): [], tuple([1, 0]): [], tuple([1, 1]): []}
    L = []

    i = 0
    for u in N:
        i = i + 1

        for v in N:
            for w in N:

                # if random.uniform(0, 1) >= Ra:
                #     continue

                V = [int(G.has_edge(u, v)), int(G.has_edge(v, w)), int(G.has_edge(u, w))]
                L.append(V)

                R[tuple([V[0], V[1]])].append(V[2])

        # print (i, float(i) / len(G), {key: [float(len([val for val in R[key] if val == 1])), (len(R[key]) + 1)]
        #                            for key in R.keys()})
        #
        # print (i, float(i) / len(G), {key: float(len([val for val in R[key] if val == 1])) / float(len(R[key]) + 1)
        #                            for key in R.keys()}, '\n')

        # if i % 10 == 0:
        # pickle.dump(R, open('R.p', 'wb'))
        # pickle.dump(L, open('L.p', 'wb'))
        # a = accuracy(L)
        # L = []

    a, p = accuracy(L)
    Ve.append(a)
    Vp.append(p)

    print (np.mean(Ve), np.std(Ve))
    print (np.mean(Vp), np.std(Vp))

    # pickle.dump(R, open('R.p', 'wb'))
    # pickle.dump(L, open('L.p', 'wb'))
