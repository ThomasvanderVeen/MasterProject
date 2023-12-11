from functions import *
from matplotlib.patches import Rectangle
from matplotlib.lines import Line2D
swing = pickle_open('Data/swing')
stance = pickle_open('Data/swing')

_, _, _, _, _, _, base_perm = get_encoding()

base_perm = base_perm.astype(int)

general = np.zeros((3, 4, 6, 24))

fig, axes = plt.subplots(2, 3)

for k in range(6):
    for i in range(3):
        counts = [0, 0, 0, 0]
        for j in range(112):
            type = base_perm[j, i]-1
            if type >= 0:
                general[i, type, k, counts[type]] = swing[k, j]
                counts[type] += 1


general_average = np.mean(general, axis=3)
general_max = np.max(general, axis=3)-general_average
general_min = general_average-np.min(general, axis=3)
general_std = np.std(general, axis=3)

print(np.max(general_max + general_average))
print(np.min(-general_min + general_average))

x = [0, 1, 2, 3, 6, 7, 8, 9, 12, 13, 14, 15]
LEGS = ['R1', 'R2', 'R3', 'L1', 'L2', 'L3']
colors = ['#e41a1c', '#377eb8', '#4daf4a', '#984ea3', '#ff7f00', '#ffff33', '#a65628']

for k in range(6):
    ax = axes[k // 3, k % 3]
    ax.set_ylim([-0.1, 1.1])
    ax.set_xlim([-0.3, 2.9])
    ax.plot([-2, 5], [0, 0], linestyle='dotted', color='black')
    ax.plot([-2, 5], [1, 1], linestyle='dotted', color='black')
    for j in range(4):
        ax.errorbar(np.array([0, 1, 2])+0.2*j, general_average[:, j, k], yerr=((general_min[:, j, k], general_max[:, j, k])), capsize=3, fmt='.', color=colors[j+1])
        for i in range(3):
            ax.add_patch(Rectangle((i+0.2*j-0.08, general_average[i, j, k]-general_std[i, j, k]), 0.16, 2*general_std[i, j, k], facecolor=colors[j+1]))
        #ax.errorbar(i+0.2*j, general_average[i, j, k], yerr=general_std[i, j, k], capsize=2, fmt='o', color=colors[j])
    if k % 3 != 0:
        ax.set_yticks([])
    if k // 3 == 1:
        ax.set_xticks([.3, 1.3, 2.3])
        labels = [item.get_text() for item in ax.get_xticklabels()]
        labels[:] = ['α', 'β', 'γ']
        ax.set_xticklabels(labels)
    else:
        ax.set_xticks([])
    ax.set_title(LEGS[k])
labels = ['v-', 'v+', 'p-', 'p+']
legend_elements = [Line2D([0], [0], marker='o', color='w', label=labels[i], markerfacecolor=colors[i+1], markersize=7)
                   for i in range(len(labels))]
fig.legend(handles=legend_elements, loc='center right')
plt.show()