import networkx as nx
import pickle
import random
import numpy as np
from sklearn import metrics


def prior(G):
    H = G.to_undirected()
    F = [0, 0, 0, 0]
    Fn = [0, 0, 0, 0]

    i = 0

    L = [[], [], []]
    for u in G.nodes():
        for v in G.nodes():

            l = list(H.neighbors(u)) + list(H.neighbors(v))
            l = list(set(l))

            for w in l:

                if G.has_edge(u, v) and G.has_edge(v, w):
                    F[2] += 1
                    if G.has_edge(u, w):
                        F[3] += 1
                else:
                    Fn[2] += 1
                    if G.has_edge(u, w):
                        Fn[3] += 1

                if G.has_edge(u, v) and G.has_edge(u, w):
                    F[1] += 1
                    if G.has_edge(v, w):
                        F[3] += 1
                else:
                    Fn[1] += 1
                    if G.has_edge(v, w):
                        Fn[3] += 1

                if G.has_edge(v, w) and G.has_edge(u, w):
                    F[0] += 1
                    if G.has_edge(u, v):
                        F[3] += 1.0
                else:
                    Fn[0] += 1
                    if G.has_edge(u, v):
                        Fn[3] += 1

        # print(float(i) / float(len(G)), F, [len(each) for each in L])
        i = i + 1

    return F, Fn


def self_loop(G):
    L = list(G.nodes())
    C = 0

    for i in range(len(L) - 1):
        for j in range(i + 1, len(L)):
            if G.has_edge(L[i], L[j]) and G.has_edge(L[j], L[i]):
                C = C + 1

    return float(C) / float(len(G.edges()))

# G = nx.read_gml('networks/Ecoli.gml')
# G = nx.erdos_renyi_graph(n = 50, p = 0.2, directed = True)
#
# F = prior(G)
# print (F)

'''
V = []
for i in range(50):
    G = nx.read_gml('networks/friendship_group/wiki' + str(i) + '.gml')

    v = self_loop(G)
    V.append(v)

    print (i, np.mean(V), np.std(V))
'''
