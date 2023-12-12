from matplotlib.patches import Rectangle

from plots import *

swing = pickle_open('Data/swing')

_, _, _, _, _, _, base_perm = get_encoding()

swings = np.zeros((3, 4, 6, 24))

for i, j, k in np.ndindex((6, 3, 112)):
    perm_type = base_perm[k, j] - 1
    if perm_type >= 0:
        index_to_update = np.argmax(swings[j, perm_type, i, :] == 0)
        swings[j, perm_type, i, index_to_update] = swing[i, k]

swings_average = np.mean(swings, axis=3)
swings_max = np.max(swings, axis=3) - swings_average
swings_min = swings_average - np.min(swings, axis=3)
swings_std = np.std(swings, axis=3)

x = [0, 1, 2, 3, 6, 7, 8, 9, 12, 13, 14, 15]
LEGS = ['R1', 'R2', 'R3', 'L1', 'L2', 'L3']
colors = ['#e41a1c', '#377eb8', '#4daf4a', '#984ea3', '#ff7f00', '#ffff33', '#a65628']

Y_AXIS_LIMITS = [-0.1, 1.1]
X_AXIS_LIMITS = [-0.3, 2.9]
Y_TICKS = [0, 0.5, 1]
X_TICKS = [.3, 1.3, 2.3]
X_LABELS = ['α', 'β', 'γ']
LEGEND_LABELS = ['v-', 'v+', 'p-', 'p+']

fig, axes = plt.subplots(2, 3)

for k in range(6):
    ax = axes[k // 3, k % 3]
    ax.set_ylim(Y_AXIS_LIMITS)
    ax.set_xlim(X_AXIS_LIMITS)

    for y_val in [0, 0.5, 1]:
        ax.plot([-2, 5], [y_val, y_val], linestyle='dotted', color='black', zorder=0)

    for j in range(4):
        ax.errorbar(np.array([0, 1, 2]) + 0.2 * j, swings_average[:, j, k],
                    yerr=(swings_min[:, j, k], swings_max[:, j, k]), capsize=3, fmt='None', color=colors[j + 1])

        for i in range(3):
            ax.add_patch(Rectangle((i + 0.2 * j - 0.08, swings_average[i, j, k] - swings_std[i, j, k]), 0.16,
                                   2 * swings_std[i, j, k], facecolor=colors[j + 1]))
            ax.add_patch(
                Rectangle((i + 0.2 * j - 0.08, swings_average[i, j, k] - 0.0075), 0.16, 0.015, facecolor='black',
                          zorder=10))

    if k % 3 != 0:
        ax.set_yticks([])
    else:
        ax.set_yticks(Y_TICKS)

    if k // 3 == 1:
        ax.set_xticks(X_TICKS)
        labels = [item.get_text() for item in ax.get_xticklabels()]
        labels[:] = X_LABELS
        ax.set_xticklabels(labels)
    else:
        ax.set_xticks([])
    ax.set_title(LEGS[k])

legend_elements = [
    Line2D([0], [0], marker='o', color='w', label=LEGEND_LABELS[i], markerfacecolor=colors[i + 1], markersize=7)
    for i in range(len(LEGEND_LABELS))]
fig.legend(handles=legend_elements, loc='lower center', ncol=4, bbox_to_anchor=(0.5, -0.09))

fig.tight_layout(pad=0.5)
fig.savefig('Images/swing_stance_comparison', bbox_inches='tight')
