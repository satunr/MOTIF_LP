import networkx as nx
import pickle
import random
import numpy as np
from sklearn import metrics


def reads(fname):

    G = nx.DiGraph()
    f = open(fname, 'r')

    for line in f.readlines():
        # print (line, line.split())
        # u, v, _, _ = line.split()
        # u, v, _ = line.split()
        u, v = line.split()

        G.add_edge(int(u), int(v))
    return G


def motif(G, skip = 1.0):
    M = []
    n = len(G)

    C = {e: 0 for e in list(G.edges())}
    eC = {e: [0, 0, 0] for e in list(G.edges())}

    for u in sorted(G.nodes()):

        if random.uniform(0, 1.0) > 1.0 / float(skip):
            continue
        else:
            print (u, 'Not SKIP')
            print (float(u) / len(G))
            print(n, len(M), float(len(M) * skip) / float(n * (n - 1) * (n - 2)))

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

    return M, C, eC, skip


def augment(C, e, eC, mode):
    C[e] += 1
    if mode == 0:
        eC[e] = [eC[e][0] + 1, eC[e][1], eC[e][2]]
    elif mode == 1:
        eC[e] = [eC[e][0], eC[e][1] + 1, eC[e][2]]
    else:
        eC[e] = [eC[e][0], eC[e][1], eC[e][2] + 1]

    return C, eC


def gen(G, si):

    G = nx.convert_node_labels_to_integers(G, first_label = 0)
    H = G.to_undirected()
    dsum = float(sum([H.degree(u) for u in H.nodes()]))

    # New subgraph
    g = nx.DiGraph()

    # Pick random seed as a starting node
    ra = np.random.choice(G.nodes(), 10, p = [float(H.degree(u))/float(dsum) for u in G.nodes()])
    r = random.choice(ra)
    # print (ra, r)

    g.add_node(int(r))

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
                if 'value' in G[u][r]:
                    g.add_edge(u, r, weight = G[u][r]['value'])
                else:
                    g.add_edge(u, r)

            if G.has_edge(r, u):
                if 'value' in G[r][u]:
                    g.add_edge(r, u, weight = G[r][u]['value'])
                else:
                    g.add_edge(r, u)

    h = g.to_undirected()
    if nx.number_connected_components(h) > 1:
        print ("ALARM---")

    return g


# G = reads('networks/email-Eu-core.txt')
# G = reads('networks/metabolic_edgelist.txt')
# G = read('networks/Florida-bay.txt')
# G = reads('networks/soc-twitter-follows-mun.edges')
# G = reads('networks/cit-HepPh.edges')
G = reads('networks/wiki-Vote.txt')
# G = reads('networks/reco.edges')

# G = nx.read_gml('networks/Ecoli.gml')
# print (len(G))
# exit(1)

# G = nx.read_gml('networks/Yeast.gml')
# G = nx.read_gml('networks/Human-Original.gml')

# Gcc = sorted(nx.connected_components(G.to_undirected()), key = len, reverse = True)
# print ([len(each) for each in Gcc])
# G = G.subgraph(Gcc[0])

# G = nx.convert_node_labels_to_integers(G, first_label = 0)
# print (len(G), len(G.edges()))

n = len(G)
M, _, _, skip = motif(G)
print (n, len(M), float(len(M) * skip) / float(n * (n - 1) * (n - 2)))
exit(1)

# for i in range(50):
#     H = gen(G, 300)
#
#     print (len(M) / float(V**3))
#     print (len(H), len(H.edges()))
#     nx.write_gml(H, 'networks/reco_group/reco' + str(i) + '.gml')
#     nx.write_gml(H, 'networks/biological_group/biological' + str(i) + '.gml')

# M, _, _, skip = motif(G)
# print (len(M))

# E. coli   1565    4313    1.1273775688479392e-06
# Yeast     4441    4115    4.701332322940128e-08
# Human     2862    7388    3.1548111451562546e-07
# Mouse     2456    4103    2.772982313863383e-07
# Email     1005    432643  0.00042749296658200857
# Metabolic 1039    13198   1.1800936312646464e-05