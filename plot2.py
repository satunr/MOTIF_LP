import matplotlib.pyplot as plt
import numpy as np


Networks = ['Meta', 'Ecoli', 'Yeast', 'Hum', 'Mse', 'Friend', 'Twitter', 'Wiki', 'Cite', 'Reco']
Options = ['Variant 1', 'Variant 2', 'Variant 3', 'Jaccard']
width = 0.2

f = open('res_metric.txt', 'r')
L = f.readlines()
print (L)

Y = {val: [] for val in Options}

for i in range(len(Networks)):
    print (i)

    l = [j for j in range(len(L)) if 'N' + str(i) == L[j].replace('\n', '')][0]

    print (L[l + 2], L[l + 3], L[l + 4], L[l + 5])

    for k in range(len(Options)):
        Y[Options[k]].append(float(L[l + 2 + k].split()[0]))

disp = [0.675, 2, 2, 0.675]
ind = 0
colorlist = ['red', 'blue', 'green', 'black']
for key in Y.keys():
    if ind < 2:
        plt.bar([i - width / disp[ind] for i in range(len(Networks))], np.round(Y[key], 2), width = width,
                color = colorlist[ind], label = key, alpha = 0.5)
    else:
        plt.bar([i + width / disp[ind] for i in range(len(Networks))], np.round(Y[key], 2), width = width,
                color = colorlist[ind], label = key, alpha = 0.5)

    ind = ind + 1

print (Y)

plt.ylim([0.25, 1.2])
plt.ylabel('ROC AUC', fontsize = 12)
plt.xticks([i for i in range(len(Networks))], Networks, rotation = 90)
plt.tight_layout()
plt.legend()
plt.savefig('Metric Variant.png', dpi = 450)
plt.show()

