import networkx as nx
import pickle
import numpy as np
import matplotlib.pyplot as plt
import random
from scipy.stats import variation


def motif(G):
    M = []
    C = {e: 0 for e in list(G.edges())}
    eC = {e: [0, 0, 0] for e in list(G.edges())}
    i = 0
    for u in sorted(G.nodes()):
        # print (float(i) / len(G))
        i = i + 1

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


def freq(C, h = 1000):

    m = 0
    f = [0 for _ in range(h)]
    for u in C.keys():
        f[C[u]] += 1

        if C[u] > m:
            m = C[u]
    return f[: m + 1]


def reorder(X, Y, Z):
    for i in range(len(X) - 1):
        for j in range(i + 1, len(X)):

            if X[i] > X[j]:
                temp = X[i]
                X[i] = X[j]
                X[j] = temp

                temp = Y[i]
                Y[i] = Y[j]
                Y[j] = temp

                temp = Z[i]
                Z[i] = Z[j]
                Z[j] = temp

    return X, Y, Z


def accuracy(L):
    # L = pickle.load(open('L.p', 'rb'))
    # print(L)

    num_a = 0
    den_a = 0
    num_p = 0
    den_p = 0

    for i in range(len(L)):
        den_a = den_a + 1

        if L[i] == [1, 1, 1] or L[i] == [0, 0, 0] or L[i] == [0, 1, 0] or L[i] == [1, 0, 0]:
            num_a = num_a + 1

        if L[i] == [1, 1, 1] or L[i] == [0, 0, 1] or L[i] == [0, 1, 1] or L[i] == [1, 0, 1]:
            den_p = den_p + 1
            if L[i] == [1, 1, 1]:
                num_p = num_p + 1

    return float(num_a) / float(den_a), float(num_p) / float(den_p)


'''
# nG = 200
G = nx.read_gml('networks/Ecoli.gml')
# M, _, _ = motif(G)
# print (float(len(M)) / float(len(G.edges())))

H = nx.erdos_renyi_graph(n = len(G), p = 0.0015, directed = True)
M, _, _ = motif(H)
print (float(len(M)) / float(len(H.edges())))
'''

'''
G = nx.convert_node_labels_to_integers(G, first_label = 0)
M, C, eC = motif(G)
pickle.dump(eC, open('eC.p', 'wb'))
exit(1)

plt.plot([i for i in range(len(eC.keys()))], [eC[u][0] for u in sorted(eC.keys())])
plt.plot([i for i in range(len(eC.keys()))], [eC[u][1] for u in sorted(eC.keys())])
plt.plot([i for i in range(len(eC.keys()))], [eC[u][2] for u in sorted(eC.keys())])
plt.show()
'''

'''
eC = pickle.load(open('eC.p', 'rb'))
width = 0.25

X = [eC[u][2] for u in sorted(eC.keys())]
Y = [eC[u][1] for u in sorted(eC.keys())]
Z = [eC[u][0] for u in sorted(eC.keys())]

X, Y, Z = reorder(X, Y, Z)
print (X)

x = list(set(X))

Y = [np.mean([Y[i] for i in range(len(Y)) if X[i] == val]) for val in x]
Y = [float(val) / float(sum(Y)) for val in Y]
print (len(Y))

Z = [np.mean([Z[i] for i in range(len(Z)) if X[i] == val]) for val in x]
Z = [float(val) / float(sum(Z)) for val in Z]
print (len(Z))

plt.bar(np.array(x) - width / 2, Y, alpha = 0.5, color = 'blue', width = width, label = 'v --> w')
plt.bar(np.array(x) + width / 2, Z, alpha = 0.5, color = 'red', width = width, label = 'u --> v')

plt.xlabel('Number of u --> w relationships')
plt.ylabel('Normalized number of u --> v / v --> w relationships')
plt.legend()
plt.show()
'''

'''
print ('Coefficient of variation: ', variation(list(C.values())))
print ('Fraction of edges with non-zero motif centrality: ', float(len([e for e in G.edges() if C[e] > 0]))
       / float(len(G.edges())))

f = freq(C)
# print (f)

plt.plot([i for i in range(len(f))], f)
plt.show()
'''


