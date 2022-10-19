import networkx as nx
import matplotlib.pyplot as plt
import pickle

def motif(G):
    M = []
    C = {e: 0 for e in list(G.edges())}
    eC = {e: [0, 0, 0] for e in list(G.edges())}

    i = 0
    for u in sorted(G.nodes()):
        print (float(i) / len(G))
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

        print (eC)
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


'''
G = nx.read_gml('networks/Ecoli.gml')
M, C, eC = motif(G)

F = [0 for _ in range(1000)]
for (u, v) in eC.keys():
    F[sum(eC[(u, v)])] += 1

print (F)
pickle.dump(F, open('F.p', 'wb'))
'''

F = pickle.load(open('F.p', 'rb'))
print (F)

plt.loglog([i for i in range(1000)], F, linewidth = 2)
# plt.plot([i for i in range(1000)], F)

plt.xlabel('Edge Motif Centrality (log scale)', fontsize = 12)
plt.ylabel('Frequency (log scale)', fontsize = 12)
plt.tight_layout()

plt.savefig('Dist.png', dpi = 450)
plt.show()

plt.show()
