import networkx as nx
import pickle
import random
import numpy as np
from sklearn import metrics


def motif(G):
    M = []
    C = {e: 0 for e in list(G.edges())}
    eC = {e: [0, 0, 0] for e in list(G.edges())}

    for u in sorted(G.nodes()):
        # print (float(u) / len(G))

        if G.out_degree(u) < 2:
            continue
        for v in sorted(G.nodes()):
            if G.in_degree(v) < 1 or G.in_degree(v) < 1:
                continue
            for w in sorted(G.nodes()):
                if G.in_degree(w) < 2:
                    continue

                if G.has_edge(u, v) and G.has_edge(v, w) and G.has_edge(u, w):
                    M.append([u, v, w])

                    C, eC = augment(C, (u, v), eC, 0)
                    C, eC = augment(C, (v, w), eC, 1)
                    C, eC = augment(C, (u, w), eC, 2)

        # print('Fraction of edges with non-zero motif centrality: ', float(len([e for e in G.edges() if C[e] > 0]))
        #       / float(len(G.edges())))

    return M, C, eC


def augment(C, e, eC, mode):
    C[e] += 1
    if mode == 0:
        eC[e] = [eC[e][0] + 1, eC[e][1], eC[e][2]]
    elif mode == 1:
        eC[e] = [eC[e][0], eC[e][1] + 1, eC[e][2]]
    else:
        eC[e] = [eC[e][0], eC[e][1], eC[e][2] + 1]

    return C, eC


def sparse(G):
    e = len(G.edges())
    n = len(G.nodes())

    return float(e) / float(n * (n - 1))


def merge(Gt, Gp):
    G = nx.DiGraph()
    G.add_nodes_from(list(set(list(Gt.nodes()) + list(Gp.nodes()))))
    G.add_edges_from(list(Gt.edges()) + list(Gp.edges()))

    return G


V = []
for i in range(50):
    Gt = pickle.load(open('networks/friendship_group/wiki_train' + str(i) + '.gml', 'rb'))
    Gp = pickle.load(open('networks/friendship_group/wiki_test' + str(i) + '.gml', 'rb'))

    G = merge(Gt, Gp)

    H = nx.erdos_renyi_graph(n = len(G), p = sparse(G), directed = True)
    n = len(H)
    M, _, _ = motif(H)

    d = float(len(M)) / float(n * (n - 1) * (n - 2))
    # print (i, len(G.edges()), len(H.edges()), len(M), np.mean(V), np.std(V))

    # d = float(len(H.edges()))
    V.append(d)
    print (np.mean(V))

# E. coli       0.028681025762156004 0.007397258674587788 887.24 9.644377604692749e-07
# Yeast         0.03092695080710109 0.009199741667191228 907.92 1.1282949129462114e-06
# Metabolic     0.10766085297047794 0.01267681153644922 1689.94 6.812417229691814e-06
# Citation      1.0486747010101025 0.48015574174682446 5168.62 0.00022113009060776786
# Email         2.8071151223088746 0.24150662218716015 8724.62 0.000918102100214735
# Twitter       0.0115847556606881 0.0115983962514816 528.58 3.0900915056152877e-07
# Wikipedia     0.6117847178532356 0.12733235835568305 4028.6 9.379288156644445e-05
# Reco          0.057677059673331865 0.031015765052598586 1231.84 2.9007953431647627e-06
# Mouse         0.03958829525833174 0.007653102024340399 1023.48 1.5697365566055382e-06
# Human         0.05999494250588503 0.009297150800384348 1263.42 2.8364496120550983e-06
