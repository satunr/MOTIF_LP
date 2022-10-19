import random
import pickle
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt

from sklearn import metrics
from mpl_toolkits.mplot3d import Axes3D



def prior_sign(G, s):
    H = G.to_undirected()

    F = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    Fn = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]

    # Ordering is u -> v (0); v -> w (1); u -> w (2)

    i = 0

    L = [[], [], []]
    for u in G.nodes():
        for v in G.nodes():

            l = list(H.neighbors(u)) + list(H.neighbors(v))
            l = list(set(l))

            for w in l:

                if G.has_edge(u, v) and G.has_edge(v, w):
                    if '-' in G[u][v]['weight'] and '-' in G[v][w]['weight']:
                        F[8] += 1
                        if G.has_edge(u, w) and s in G[u][w]['weight']:
                            F[12] += 1
                    else:
                        Fn[8] += 1
                        if G.has_edge(u, w) and s in G[u][w]['weight']:
                            Fn[12] += 1

                    if '-' in G[u][v]['weight'] and '+' in G[v][w]['weight']:
                        F[9] += 1
                        if G.has_edge(u, w) and s in G[u][w]['weight']:
                            F[12] += 1
                    else:
                        Fn[9] += 1
                        if G.has_edge(u, w) and s in G[u][w]['weight']:
                            Fn[12] += 1

                    if '+' in G[u][v]['weight'] and '-' in G[v][w]['weight']:
                        F[10] += 1
                        if G.has_edge(u, w) and s in G[u][w]['weight']:
                            F[12] += 1
                    else:
                        Fn[10] += 1
                        if G.has_edge(u, w) and s in G[u][w]['weight']:
                            Fn[12] += 1

                    if '+' in G[u][v]['weight'] and '+' in G[v][w]['weight']:
                        F[11] += 1
                        if G.has_edge(u, w) and s in G[u][w]['weight']:
                            F[12] += 1
                    else:
                        Fn[11] += 1
                        if G.has_edge(u, w) and s in G[u][w]['weight']:
                            Fn[12] += 1

                if G.has_edge(v, w) and G.has_edge(u, w):
                    if '-' in G[v][w]['weight'] and '-' in G[u][w]['weight']:
                        F[0] += 1
                        if G.has_edge(u, v) and s in G[u][v]['weight']:
                            F[12] += 1
                    else:
                        Fn[0] += 1
                        if G.has_edge(u, v) and s in G[u][v]['weight']:
                            Fn[12] += 1

                    if '-' in G[v][w]['weight'] and '+' in G[u][w]['weight']:
                        F[1] += 1
                        if G.has_edge(u, v) and s in G[u][v]['weight']:
                            F[12] += 1
                    else:
                        Fn[1] += 1
                        if G.has_edge(u, v) and s in G[u][v]['weight']:
                            Fn[12] += 1

                    if '+' in G[v][w]['weight'] and '-' in G[u][w]['weight']:
                        F[2] += 1
                        if G.has_edge(u, v) and s in G[u][v]['weight']:
                            F[12] += 1
                    else:
                        Fn[2] += 1
                        if G.has_edge(u, v) and s in G[u][v]['weight']:
                            Fn[12] += 1

                    if '+' in G[v][w]['weight'] and '+' in G[u][w]['weight']:
                        F[3] += 1
                        if G.has_edge(u, v) and s in G[u][v]['weight']:
                            F[12] += 1
                    else:
                        Fn[3] += 1
                        if G.has_edge(u, v) and s in G[u][v]['weight']:
                            Fn[12] += 1

                if G.has_edge(u, v) and G.has_edge(u, w):
                    if '-' in G[u][v]['weight'] and '-' in G[u][w]['weight']:
                        F[4] += 1
                        if G.has_edge(v, w) and s in G[v][w]['weight']:
                            F[12] += 1
                    else:
                        Fn[4] += 1
                        if G.has_edge(v, w) and s in G[v][w]['weight']:
                            Fn[12] += 1

                    if '-' in G[u][v]['weight'] and '+' in G[u][w]['weight']:
                        F[5] += 1
                        if G.has_edge(v, w) and s in G[v][w]['weight']:
                            F[12] += 1
                    else:
                        Fn[5] += 1
                        if G.has_edge(v, w) and s in G[v][w]['weight']:
                            Fn[12] += 1

                    if '+' in G[u][v]['weight'] and '-' in G[u][w]['weight']:
                        F[6] += 1
                        if G.has_edge(v, w) and s in G[v][w]['weight']:
                            F[12] += 1
                    else:
                        Fn[6] += 1
                        if G.has_edge(v, w) and s in G[v][w]['weight']:
                            Fn[12] += 1

                    if '+' in G[u][v]['weight'] and '+' in G[u][w]['weight']:
                        F[7] += 1
                        if G.has_edge(v, w) and s in G[v][w]['weight']:
                            F[12] += 1
                    else:
                        Fn[7] += 1
                        if G.has_edge(v, w) and s in G[v][w]['weight']:
                            Fn[12] += 1

        i = i + 1
    return F, Fn


def score_sign(Gt, Gp, mode, kappa = 0.001):
    C, Cn = {}, {}
    i = 0
    for x in list(Gp.nodes()):

        i = i + 1
        for y in list(Gp.nodes()):
            if x == y or (x, y) in Gt.edges():
                continue

            C[(x, y)], Cn[(x, y)] = 0, 0

            for z in list(Gt.nodes()):
                if z == x or z == y:
                    continue

                if (x, z) in Gt.edges() and (y, z) in Gt.edges():
                    if mode == 0:
                        if '-' in G[y][z]['weight'] and '-' in G[x][z]['weight']:
                            C[(x, y)] += 1
                        else:
                            Cn[(x, y)] += 1

                    if mode == 1:
                        if '-' in G[y][z]['weight'] and '+' in G[x][z]['weight']:
                            C[(x, y)] += 1
                        else:
                            Cn[(x, y)] += 1

                    if mode == 2:
                        if '+' in G[y][z]['weight'] and '-' in G[x][z]['weight']:
                            C[(x, y)] += 1
                        else:
                            Cn[(x, y)] += 1

                    if mode == 3:
                        if '+' in G[y][z]['weight'] and '+' in G[x][z]['weight']:
                            C[(x, y)] += 1
                        else:
                            Cn[(x, y)] += 1

                if (z, x) in Gt.edges() and (z, y) in Gt.edges():
                    if mode == 4:
                        if '-' in G[z][y]['weight'] and '-' in G[z][x]['weight']:
                            C[(x, y)] += 1
                        else:
                            Cn[(x, y)] += 1

                    if mode == 5:
                        if '-' in G[z][y]['weight'] and '+' in G[z][x]['weight']:
                            C[(x, y)] += 1
                        else:
                            Cn[(x, y)] += 1

                    if mode == 6:
                        if '+' in G[z][y]['weight'] and '-' in G[z][x]['weight']:
                            C[(x, y)] += 1
                        else:
                            Cn[(x, y)] += 1

                    if mode == 7:
                        if '+' in G[z][y]['weight'] and '+' in G[z][x]['weight']:
                            C[(x, y)] += 1
                        else:
                            Cn[(x, y)] += 1

                if (x, z) in Gt.edges() and (z, y) in Gt.edges():
                    if mode == 8:
                        if '-' in G[x][z]['weight'] and '-' in G[z][y]['weight']:
                            C[(x, y)] += 1
                        else:
                            Cn[(x, y)] += 1
                    if mode == 9:
                        if '-' in G[x][z]['weight'] and '+' in G[z][y]['weight']:
                            C[(x, y)] += 1
                        else:
                            Cn[(x, y)] += 1
                    if mode == 10:
                        if '+' in G[x][z]['weight'] and '-' in G[z][y]['weight']:
                            C[(x, y)] += 1
                        else:
                            Cn[(x, y)] += 1
                    if mode == 11:
                        if '+' in G[x][z]['weight'] and '+' in G[z][y]['weight']:
                            C[(x, y)] += 1
                        else:
                            Cn[(x, y)] += 1

    C = {key: (C[key] + kappa) / (max(C.values()) + kappa) for key in C.keys()}
    Cn = {key: (Cn[key] + kappa) / (max(Cn.values()) + kappa) for key in Cn.keys()}
    return C, Cn


def sample(G, perc):
    E = list(G.edges())

    Et, Ep = [], []
    for i in range(len(E)):

        if random.uniform(0, 1) < perc:
            Et.append(E[i])
        else:
            Ep.append(E[i])

    Gt = create_graph(G, Et)
    Gp = create_graph(G, Ep)
    return Gt, Gp


def create_graph(G, E):
    H = nx.DiGraph()
    H.add_nodes_from(list(G.nodes()))

    for (u, v) in E:
        H.add_edge(u, v, weight = G[u][v]['weight'])
    return H


def find_auc(G, S, s):

    Y = []
    pred = []
    for key in S.keys():
        if key in G.edges() and s in G[key[0]][key[1]]['weight']:
            # print ('**')
            Y.append(1)
        else:
            Y.append(0)

        pred.append(S[key])

    return metrics.roc_auc_score(Y, pred)


def jaccard_remove(G, s):
    E = [(u, v) for (u, v) in G.edges() if s not in G[u][v]['weight']]
    G.remove_edges_from(E)
    return G.to_undirected()



kappa = 0.001
# k = 1.0
# K = [10000.0, 1000.0, 100.0, 10.0, 1.0, 0.1, 0.01, 0.001, 0.0001]
K = [1.0]
A4, A2 = [], []

for i in range(50):
    G = nx.read_gml('networks/biological_group/biological' + str(i) + '.gml')
    print (len(G), len(G.edges()))

    Gt, Gp = sample(G, 0.9)
    print (i, len(Gt), len(Gp), len(Gt.edges()), len(Gp.edges()))

    # Gt = pickle.load(open('networks/biological_group/biological_train' + str(i) + '.gml', 'rb'))
    # Gp = pickle.load(open('networks/biological_group/biological_test' + str(i) + '.gml', 'rb'))

    F, Fn = prior_sign(Gt, '+')

    CList, CnList = [], []
    for j in range(12):
        C, Cn = score_sign(Gt, Gp, j)
        # print (C)

        CList.append(C)
        CnList.append(Cn)

    for k in K:
        S4 = {}
        for e in CList[0].keys():
            S4[e] = 0
            for j in range(12):
                S4[e] += float(F[12]) / float(F[j] + kappa) * CList[j][e] + \
                         k * float(Fn[12]) / float(Fn[j] + kappa) * CnList[j][e]

        a4 = find_auc(Gp, S4, '+')
        # print (k, a4)

        A4.append(a4)
        print(np.mean(A4), np.std(A4))

    S2 = {}
    I = jaccard_remove(Gt, '+')
    preds = nx.jaccard_coefficient(I, [(u, v) for u in Gp.nodes() for v in Gp.nodes()
                                       if u != v and (u, v) not in Gt.edges()])

    for u, v, p in preds:
        S2[(u, v)] = p

    a2 = find_auc(Gp, S2, '+')
    A2.append(a2)
    print(np.mean(A2), np.std(A2))
    print ('\n')




