import matplotlib.pylab as pylab
import seaborn as sns
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle

from functions import *

create_directory('Images')
create_directory('Images_PSTH')

params = {'legend.fontsize': 10,
          'figure.figsize': (1.5 * 3.54, 3.54),
          'figure.dpi': 600,
          'axes.labelsize': 15,
          'axes.titlesize': 15,
          'xtick.labelsize': 10,
          'ytick.labelsize': 10}
pylab.rcParams.update(params)

colors = ['#e41a1c', '#377eb8', '#4daf4a', '#984ea3', '#ff7f00', '#FFDB58', '#a65628']
markers = ['o', "^", "v", "*", "+", "x", "s", "p"]


def plot_single_hair(ax, v):
    #plt.style.use('classic')
    fig = plt.figure(1)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Firing frequency (Hz)")
    ax.grid(axis='y')
    ax.minorticks_on()

    for i in range(len(ax.get_lines())):
        ax.get_lines()[i].set_color(colors[i])

    if isinstance(v, int):
        ax.legend(['θ = 15°', 'θ = 23°', 'θ = 34°', 'θ = 46°', 'θ = 60°'], fancybox=False, edgecolor='black')
        ax.set_yticks([0, 40, 80, 120, 160, 200, 240, 280])
        fig.tight_layout(pad=0.5)
        fig.savefig('Images/angle.png')
        fig.savefig('Images/angle.pdf')
    else:
        ax.legend(['ω = 24 °/s', 'ω = 47 °/s', 'ω = 88 °/s', 'ω = 151 °/s', 'ω = 245 °/s'], fancybox=False, edgecolor='black', loc='lower center')
        ax.set_yticks([0, 40, 80, 120, 160])
        fig.tight_layout(pad=0.5)
        fig.savefig('Images/angular_velocity.png')
        fig.savefig('Images/angular_velocity.pdf')

    plt.close(fig)

    return


def plot_heat_map(df):
    heatmap = sns.heatmap(data=df, annot=True, fmt='.3g', cbar_kws={'label': 'mean absolute error (MAE)'})
    heatmap.set(xlabel='b (pV)', ylabel=r'$\tau_w$ (s)')

    plt.tight_layout(pad=0.5)
    plt.savefig('Images/heat_map.png')
    plt.savefig('Images/heat_map.pdf')


def plot_hair_field(ax, name):
    fig = plt.figure(1)
    ax.set_xlabel("joint angle (degrees)")
    ax.set_ylabel("hair angle (degrees)")
    ax.minorticks_on()

    fig.tight_layout(pad=0.5)
    fig.savefig('Images/hair_field_' + name + '.png')
    #fig.savefig('Images/hair_field_' + name + '.pdf')
    fig.clear()

    return


def plot_position_interneuron(ax1, ax2, fig, name):
    ax1.set_xlabel("Time (s)")
    ax1.set_ylabel("Joint angle (degrees)")
    ax1.minorticks_on()
    ax2.minorticks_on()

    if name == 'uni':
        fig.legend(['Exp. data', 'Model'], loc='lower right', fancybox=False, edgecolor='black', bbox_to_anchor=[0.87, 0.15])
    else:
        ax1.set_xlim([0, 3])
        ax2.set_xlim([0, 3])
        fig.legend(loc='lower right', fancybox=False, edgecolor='black', bbox_to_anchor=[0.87, 0.75])

    ax2.set_ylabel("Firing frequency (Hz)")

    fig.tight_layout(pad=0.5)
    fig.savefig('Images/position_interneuron_' + str(name) + '.png')
    fig.savefig('Images/position_interneuron_' + str(name) + '.pdf')
    fig.clear()

    return


def plot_spike_timing(ax1, ax2, fig, n_index):
    ax1.set_xlabel("Time (s)")
    ax1.set_ylabel("Neuron index")
    ax1.set_yticks(np.arange(1, n_index + 1)[::10])
    ax2.set_ylabel("Joint angle (degrees)")

    ax2.minorticks_on()

    fig.tight_layout(pad=0.5)
    fig.savefig('Images/spike_timing_.png')
    fig.savefig('Images/spike_timing_.pdf')
    fig.clear()

    return


def plot_movement_interneuron(ax, fig):
    ax.set_xlabel(r"Angular velocity (degrees$\cdot$s$^{-1}$)")
    ax.set_ylabel("Firing frequency (Hz)")

    ax.set_ylim([0, 120])
    ax.set_xlim([0, 200])

    ax.legend(loc='upper left', fancybox=False, edgecolor='black')

    ax.minorticks_on()
    ax.grid(zorder=0)

    fig.tight_layout(pad=0.5)
    fig.savefig('Images/movement_interneuron.png')
    fig.savefig('Images/movement_interneuron.pdf')
    fig.clear()

    return


def plot_movement_interneuron_network(ax, fig):
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Joint angle (degrees)")

    ax.grid(axis='y', zorder=0)

    ax.minorticks_on()

    ax.legend(['Exp. data', 'Dorsal direction', 'ventral direction'], loc='lower right', fancybox=False, edgecolor='black')

    fig.tight_layout(pad=0.5)
    fig.savefig('Images/movement_interneuron_network.png')
    fig.savefig('Images/movement_interneuron_network.pdf')
    fig.clear()

    return


def plot_movement_binary(ax, ax1, fig):
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Joint angle (degrees)")
    ax1.set_ylabel("Firing frequency (Hz)")

    ax.minorticks_on()
    ax1.minorticks_on()

    fig.legend(loc='upper right', fancybox=False, edgecolor='black', bbox_to_anchor=[0.87, 0.97])

    fig.tight_layout(pad=0.5)
    fig.savefig('Images/movement_binary.png')
    fig.savefig('Images/movement_binary.pdf')
    fig.clear()


def plot_primitive_roc(ax, fig):
    ax.minorticks_on()
    ax.set_xlabel("False positive rate")
    ax.set_ylabel("True positive rate")
    ax.legend(loc='lower right', fancybox=False, edgecolor='black')

    ax.set_xlim([-0.05, 1.05])
    ax.set_ylim([-0.05, 1.05])

    fig.tight_layout(pad=0.5)
    fig.savefig('Images/primitive_ROC.png')
    fig.savefig('Images/primitive_ROC.pdf')
    fig.clear()

    return


def plot_psth(ax, fig, neuron, leg, permutations_name, label):
    ax.set_ylabel('Likelihood of Spiking (%)')
    ax.set_xlabel("Locomotion phase")
    ax.set_xticks([0.375, 1.125])

    labels = [item.get_text() for item in ax.get_xticklabels()]
    labels[:] = ['Swing', 'Stance']
    ax.set_xticklabels(labels)

    ax.minorticks_on()
    ax.xaxis.set_tick_params(which='minor', bottom=False)

    fig.tight_layout(pad=0.5)
    if label == 'primitive':
        fig.savefig(f'Images_PSTH/prim_{permutations_name}_leg_{leg}_neuron_{neuron}')
    elif label == 'position':
        fig.savefig(f'Images_PSTH/pos_{permutations_name}_leg_{leg}')
    elif label == 'velocity':
        fig.savefig(f'Images_PSTH/vel_{permutations_name}_leg_{leg}')
    plt.cla()

    return


def plot_primitive_accuracy(ax, fig, tau_list):
    ax.set_xlabel('τ (ms)', fontsize=15)
    ax.set_ylabel("Balanced accuracy", fontsize=15)
    ax.set_xticks(1000 * tau_list[::2])
    ax.legend(['p-p', 'v-v', 'p-v', 'p-p-v', 'v-v-p', 'p-p-p', 'v-v-v', 'mean'], loc='lower right')
    fig.tight_layout(pad=0.5)
    fig.savefig('Images/primitive_accuracy.png')
    fig.savefig('Images/primitive_accuracy.pdf')

    return


def plot_primitive_weights(ax, fig, tau_list, w_1, w_2):
    ax[0].grid(True, axis='y')
    ax[1].grid(True, axis='y')
    ax[0].set_xticks([])
    ax[1].set_xticks(1000 * tau_list[::2])
    ax[0].set_yticks(1000 * w_1[::2])
    ax[1].set_yticks(1000 * w_2[::2])
    ax[0].set_ylabel('w_pos (mV)', fontsize=15)
    ax[1].set_ylabel('w_vel (mV)', fontsize=15)
    fig.legend(['p-p', 'v-v', 'p-v', 'p-p-v', 'v-v-p', 'p-p-p', 'v-v-v'], loc='upper center',
               bbox_to_anchor=(1.08, 0.75))
    fig.supxlabel('τ [ms]', fontsize=15)
    fig.tight_layout(pad=0.5)
    fig.savefig('Images/primitive_weights.png', bbox_inches='tight')
    fig.savefig('Images/primitive_weights.pdf', bbox_inches='tight')

    return


def plot_climbing_accuracy(fig, ax, name):
    ax.plot()
    ax.legend()
    fig.tight_layout(pad=0.5)
    if name == 'climbing':
        ax.set_ylabel('Balanced Accuracy')
        ax.set_xlabel('w_exc (mV)')
        fig.savefig('Images/climbing_accuracy.png', bbox_inches='tight')
        fig.savefig('Images/climbing_accuracy.pdf', bbox_inches='tight')
    elif name == 'pitch':
        ax.set_ylabel('DTW score')
        ax.set_xlabel('w_up (mV)')
        fig.savefig('Images/pitch_accuracy.png', bbox_inches='tight')
        fig.savefig('Images/pitch_accuracy.pdf', bbox_inches='tight')


def plot_climbing_classifier(fig, ax):
    ax.set_ylim([-20, 75])
    ax.set_xlim([0, 20])
    ax.legend(['body pitch', 'divide', 'spikes', 'climbing'])
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Body Pitch (°)")

    fig.savefig('Images/climbing_classifier.png', bbox_inches='tight')
    fig.savefig('Images/climbing_classifier.pdf', bbox_inches='tight')


def plot_swing_stance(ax, fig, x, legs):
    for i in range(3):
        ax[i].grid(axis='y')
        if i == 2:
            ax[i].set_xticks(x)
            labels = [item.get_text() for item in ax[i].get_xticklabels()]
            labels[:] = legs
            ax[i].set_xticklabels(labels)
        else:
            ax[i].set_xticks([])
    fig.supylabel("$n_{sw}/(n_{sw}+n_{st})$")
    labels = ['None', 'Vel-', 'Vel+', 'Pos-', 'Pos+']
    legend_elements = [Line2D([0], [0], marker='o', color='w', label=labels[i], markerfacecolor=colors[i], markersize=7)
                       for i in range(len(labels))]
    fig.legend(handles=legend_elements, loc='upper center', bbox_to_anchor=(1.12, 0.6))
    fig.tight_layout(pad=0.5)
    fig.savefig('Images/swing_stance.png')
    #fig.savefig('Images/swing_stance.pdf')


def plot_pitch_estimation(ax, fig):
    ax.legend(['Ground Truth', 'Model', 'Moving Average'], loc='lower right')
    ax.set_xlabel("t (s)")
    ax.set_ylabel("Body Pitch (a.u.)")

    fig.tight_layout(pad=0.5)
    fig.savefig('Images/pitch_estimation.png')
    fig.savefig('Images/pitch_estimation.pdf')
