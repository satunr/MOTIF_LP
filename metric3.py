import networkx as nx
import pickle
import random
import numpy as np
from sklearn import metrics
from triad import *


def gen(G, si):

    H = G.to_undirected()
    dsum = float(sum([H.degree(u) for u in H.nodes()]))

    # New subgraph
    g = nx.DiGraph()

    # Pick random seed as a starting node
    ra = np.random.choice(G.nodes(), 10, p = [float(H.degree(u))/float(dsum) for u in G.nodes()])
    r = random.choice(ra)
    # print (ra, r)

    g.add_node(r)

    while len(g) < si:
        nset = []
        # List of all neighbors of nodes in g
        for u in g.nodes():
            l = H.neighbors(u)
            nset.extend(l)

        # The nodes in 'nset' are not already present in g
        nset = [u for u in nset if u not in g.nodes()]

        if len(nset) > 0:
            r = random.choice(nset)
        else:
            r = random.choice(list(G.nodes()))

        N = list(g.nodes())
        for u in N:
            if G.has_edge(u, r):
                g.add_edge(u, r)

            if G.has_edge(r, u):
                g.add_edge(r, u)

    h = g.to_undirected()
    if nx.number_connected_components(h) > 1:
        print ("ALARM---")

    return g


def score(G, mode, kappa = 0.001):
    C = {}
    i = 0
    for x in list(G.nodes()):
        # if i % 20 == 0:
        #     print (mode, float(i) / len(G))

        i = i + 1
        for y in G.nodes():
            if x == y:
                continue
            C[(x, y)] = 0
            for z in G.nodes():
                if z == x or z == y:
                    continue

                if mode == 0:
                    if (x, z) in G.edges() and (y, z) in G.edges():
                        C[(x, y)] += 1

                elif mode == 1:
                    if (z, x) in G.edges() and (z, y) in G.edges():
                        C[(x, y)] += 1

                elif mode == 2:
                    if (x, z) in G.edges() and (z, y) in G.edges():
                        C[(x, y)] += 1

    C = {key: (C[key] + kappa) / (max(C.values()) + kappa) for key in C.keys()}
    pickle.dump(C, open('C' + str(mode) + '.p', 'wb'))
    # return C


def score2(Gt, Gp, mode, kappa = 0.001):
    C = {}
    i = 0
    for x in list(Gp.nodes()):

        i = i + 1
        for y in list(Gp.nodes()):
            if x == y or (x, y) in Gt.edges():
                continue

            C[(x, y)] = 0

            for z in list(Gt.nodes()):
                if z == x or z == y:
                    continue

                if mode == 0:
                    if (x, z) in Gt.edges() and (y, z) in Gt.edges():
                        C[(x, y)] += 1

                elif mode == 1:
                    if (z, x) in Gt.edges() and (z, y) in Gt.edges():
                        C[(x, y)] += 1

                elif mode == 2:
                    if (x, z) in Gt.edges() and (z, y) in Gt.edges():
                        C[(x, y)] += 1

    C = {key: (C[key] + kappa) / (max(C.values()) + kappa) for key in C.keys()}
    # pickle.dump(C, open('C' + str(mode) + '.p', 'wb'))
    return C


def score3(Gt, Gp, mode, kappa = 0.001):
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

                if mode == 0:
                    if (x, z) in Gt.edges() and (y, z) in Gt.edges():
                        C[(x, y)] += 1
                    else:
                        Cn[(x, y)] += 1

                elif mode == 1:
                    if (z, x) in Gt.edges() and (z, y) in Gt.edges():
                        C[(x, y)] += 1
                    else:
                        Cn[(x, y)] += 1

                elif mode == 2:
                    if (x, z) in Gt.edges() and (z, y) in Gt.edges():
                        C[(x, y)] += 1
                    else:
                        Cn[(x, y)] += 1

    C = {key: (C[key] + kappa) / (max(C.values()) + kappa) for key in C.keys()}
    Cn = {key: (Cn[key] + kappa) / (max(Cn.values()) + kappa) for key in Cn.keys()}

    # pickle.dump(C, open('C' + str(mode) + '.p', 'wb'))
    return C, Cn


def find_auc(G, S):

    Y = []
    pred = []
    for key in S.keys():

        if key in G.edges():
            Y.append(1)
        else:
            Y.append(0)

        pred.append(S[key])

    return metrics.roc_auc_score(Y, pred)


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
    H.add_edges_from(E)
    return H


'''
G = nx.read_gml('networks/Ecoli.gml')
r, repeat, A1, A2 = 0, 10, [], []
while r < repeat:

    H = gen(G, 400)
    pickle.dump(H, open('H.p', 'wb'))
    print (r, len(H), len(H.edges()))

    I = H.to_undirected()

    S2 = {}
    preds = nx.jaccard_coefficient(I, [(u, v) for u in I.nodes() for v in I.nodes() if u != v])

    for u, v, p in preds:
        S2[(u, v)] = p

    score(H, 0)
    print ('End of score 0')

    score(H, 1)
    print ('End of score 1')

    score(H, 2)
    print ('End of score 2')

    C0 = pickle.load(open('C0.p', 'rb'))
    C1 = pickle.load(open('C1.p', 'rb'))
    C2 = pickle.load(open('C2.p', 'rb'))

    S1 = {}
    m = 0
    for e in C0.keys():
        S1[e] = C0[e] * C1[e] * C2[e]
        if S1[e] > m:
            m = S1[e]

    a1 = find_auc(H, S1)
    A1.append(a1)
    print (np.mean(A1), np.std(A1))

    a2 = find_auc(H, S2)
    A2.append(a2)
    print (np.mean(A2), np.std(A2))

    r = r + 1
'''

A1, A2, A3, A4 = [], [], [], []
kappa = 0.001
K = 0.01

for i in range(50):
    G = nx.read_gml('networks/biological_group/biological-4-' + str(i) + '.gml')
    Gt, Gp = sample(G, 0.9)
    print (i, len(Gt), len(Gp), len(Gt.edges()), len(Gp.edges()))

    pickle.dump(Gt, open('networks/biological_group/mouse_train' + str(i) + '.gml', 'wb'))
    pickle.dump(Gp, open('networks/biological_group/mouse_test' + str(i) + '.gml', 'wb'))

    F, Fn = prior(Gt)

    # C0 = score2(Gt, Gp, 0)
    # C1 = score2(Gt, Gp, 1)
    # C2 = score2(Gt, Gp, 2)

    C0, Cn0 = score3(Gt, Gp, 0)
    C1, Cn1 = score3(Gt, Gp, 1)
    C2, Cn2 = score3(Gt, Gp, 2)

    S1, S3, S4 = {}, {}, {}
    m = 0

    for e in C0.keys():
        # print (C0[e], C1[e], C2[e], F)
        # print (Cn0[e], Cn1[e], Cn2[e], Fn, '\n')

        S1[e] = (C0[e] + C1[e] + C2[e]) / 3.0
        S3[e] = float(F[3]) / float(F[0] + kappa) * C0[e] + float(F[3]) / float(F[1] + kappa) *\
                C1[e] + float(F[3]) / float(F[2] + kappa) * C2[e]

        S4[e] = float(F[3]) / float(F[0] + kappa) * C0[e] + K * float(Fn[3]) / float(Fn[0] + kappa) * Cn0[e] + \
                float(F[3]) / float(F[1] + kappa) * C1[e] + K * float(Fn[3]) / float(Fn[1] + kappa) * Cn1[e] + \
                float(F[3]) / float(F[2] + kappa) * C2[e] + K * float(Fn[3]) / float(Fn[2] + kappa) * Cn2[e]

        # S4[e] = float(F[3]) / float(F[0] + kappa) * C0[e] + 0.01 * float(Fn[3]) / float(Fn[0] + kappa) * Cn0[e] + \
        #         float(F[3]) / float(F[1] + kappa) * C1[e] + 0.01 * float(Fn[3]) / float(Fn[1] + kappa) * Cn1[e] + \
        #         float(F[3]) / float(F[2] + kappa) * C2[e] + 0.01 * float(Fn[3]) / float(Fn[2] + kappa) * Cn2[e]

        if S1[e] > m:
            m = S1[e]

    a1 = find_auc(Gp, S1)
    A1.append(a1)
    print (np.mean(A1), np.std(A1))

    a3 = find_auc(Gp, S3)
    A3.append(a3)
    print (np.mean(A3), np.std(A3))

    a4 = find_auc(Gp, S4)
    A4.append(a4)
    print (np.mean(A4), np.std(A4))

    S2 = {}
    I = Gt.to_undirected()
    preds = nx.jaccard_coefficient(I, [(u, v) for u in Gp.nodes() for v in Gp.nodes()
                                       if u != v and (u, v) not in Gt.edges()])

    for u, v, p in preds:
        S2[(u, v)] = p

    a2 = find_auc(Gp, S2)
    A2.append(a2)
    print (np.mean(A2), np.std(A2), '\n')
