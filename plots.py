from matplotlib.lines import Line2D
import seaborn as sns
import matplotlib.pylab as pylab
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

colors = ['#e41a1c', '#377eb8', '#4daf4a', '#984ea3', '#ff7f00', '#ffff33', '#a65628']


def plot_single_hair(ax, v):
    fig = plt.figure(1)
    ax.set_xlabel("Time [s]")
    ax.set_ylabel("firing rate [imp/s]")

    for i in range(len(ax.get_lines())):
        ax.get_lines()[i].set_color(colors[i])

    if isinstance(v, int):
        ax.legend(['15°', '23°', '34°', '46°', '60°'])
        ax.set_yticks([0, 40, 80, 120, 160, 200, 240, 280])
        fig.tight_layout(pad=0.5)
        fig.savefig('Images/angle.pdf')
    else:
        ax.legend(['24 °/s', '47 °/s', '88 °/s', '151 °/s', '245 °/s'])
        ax.set_yticks([0, 40, 80, 120, 160])
        fig.tight_layout(pad=0.5)
        fig.savefig('Images/angular_velocity.pdf')

    plt.close(fig)

    return


def plot_heat_map(df):
    heatmap = sns.heatmap(data=df, annot=True, fmt='.3g', cbar_kws={'label': 'mean absolute error (MAE)'})
    heatmap.set(xlabel='b', ylabel='V_r')

    plt.tight_layout(pad=0.5)
    plt.savefig('Images/heat_map.pdf')


def plot_hair_field(ax, name):
    fig = plt.figure(1, figsize=(10, 6))
    ax.set_xlabel("joint angle [degrees]")
    ax.set_ylabel("hair angle [degrees]")

    fig.tight_layout(pad=0.5)
    fig.savefig('Images/hair_field_' + name + '.pdf')
    fig.clear()

    return


def plot_position_interneuron(ax1, ax2, fig, name):
    ax1.set_xlabel("time [s]")
    ax1.set_ylabel("firing rate [imp/s]")

    if name == 'uni':
        ax1.legend(['Model Response', 'Exp. Data'], loc='lower right')
    else:
        ax1.legend(['Dorsal Response', 'Ventral Response'], loc='lower right')

    ax2.set_ylabel("Joint Angle [°]")

    fig.tight_layout(pad=0.5)
    fig.savefig('Images/position_interneuron_' + str(name) + '.pdf')
    fig.clear()

    return


def plot_spike_timing(ax1, ax2, fig, n_index):
    ax1.set_xlabel("time [s]")
    ax1.set_ylabel("Neuron index")
    ax1.set_yticks(np.arange(1, n_index + 1)[::10])
    ax2.set_ylabel("Joint Angle [°]")

    fig.tight_layout(pad=0.5)
    fig.savefig('Images/spike_timing_.pdf')
    fig.clear()

    return


def plot_movement_interneuron(ax, fig):
    ax.set_xlabel("Velocity [°/s]")
    ax.set_ylabel("firing rate [imp/s]")

    fig.tight_layout(pad=0.5)
    fig.savefig('Images/movement_interneuron.pdf')
    fig.clear()

    return


def plot_movement_interneuron_network(ax, fig):
    ax.set_xlabel("Time [s]")
    ax.set_ylabel("Joint Angle [°]")

    ax.legend(['Stimulus', 'Down', 'Up'], loc='lower right')

    fig.tight_layout(pad=0.5)
    fig.savefig('Images/movement_interneuron_network.pdf')
    fig.clear()

    return


def plot_movement_binary(ax, ax1, fig):
    ax.set_xlabel("Time [s]")
    ax.set_ylabel("Joint Angle [degrees/s]")
    ax1.set_ylabel("firing rate [imp/s]")

    ax1.legend(['Down', 'Up'], loc='upper right')

    fig.tight_layout(pad=0.5)
    fig.savefig('Images/movement_binary.pdf')
    fig.clear()


def plot_primitive_roc(ax, fig):
    labels = ['pos-pos', 'vel-vel', 'pos-vel', 'pos-pos-vel', 'vel-vel-pos', 'pos-pos-pos', 'vel-vel-vel']

    legend_elements = [Line2D([0], [0], marker='o', color='w', label=labels[i], markerfacecolor=colors[i], markersize=7)
                       for i in range(len(labels))]

    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.legend(handles=legend_elements, loc='lower right')

    fig.tight_layout(pad=0.5)
    fig.savefig('Images/primitive_ROC.pdf')
    fig.clear()

    return


def plot_psth(ax, fig, neuron, leg, permutations_name, label):
    ax.set_ylabel('Likelihood of Spiking')
    fig.text(0.33, 0.04, 'Swing', ha='center')
    fig.text(0.66, 0.04, 'Stance')
    ax.set_xticks([])
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
    ax.set_xlabel('τ [ms]', fontsize=15)
    ax.set_ylabel("Balanced accuracy", fontsize=15)
    ax.set_xticks(1000 * tau_list[::2])
    ax.legend(['p-p', 'v-v', 'p-v', 'p-p-v', 'v-v-p', 'p-p-p', 'v-v-v', 'mean'], loc='lower right')
    fig.tight_layout(pad=0.5)
    fig.savefig('Images/primitive_accuracy.pdf')

    return


def plot_primitive_weights(ax, fig, tau_list, w_1, w_2):
    ax[0].grid(True, axis='y')
    ax[1].grid(True, axis='y')
    ax[0].set_xticks([])
    ax[1].set_xticks(1000 * tau_list[::2])
    ax[0].set_yticks(1000 * w_1[::2])
    ax[1].set_yticks(1000 * w_2[::2])
    ax[0].set_ylabel('w_pos [mV]', fontsize=15)
    ax[1].set_ylabel('w_vel [mV]', fontsize=15)
    fig.legend(['p-p', 'v-v', 'p-v', 'p-p-v', 'v-v-p', 'p-p-p', 'v-v-v'], loc='upper center',
               bbox_to_anchor=(1.08, 0.75))
    fig.supxlabel('τ [ms]', fontsize=15)
    fig.tight_layout(pad=0.5)
    fig.savefig('Images/primitive_weights.pdf', bbox_inches='tight')

    return


def plot_climbing_accuracy(fig, ax, name):
    ax.set_xlabel('w_exc (mV)')
    ax.set_ylabel('Balanced accuracy')
    ax.plot()
    ax.legend()
    fig.tight_layout(pad=0.5)
    if name == 'climbing':
        fig.savefig('Images/climbing_accuracy.pdf', bbox_inches='tight')
    elif name == 'pitch':
        fig.savefig('Images/pitch_accuracy.pdf', bbox_inches='tight')


def plot_climbing_classifier(fig, ax):
    ax.set_ylim([-20, 75])
    ax.set_xlim([0, 20])
    ax.legend(['body pitch', 'divide', 'spikes', 'climbing'])
    ax.set_xlabel("Time [s]")
    ax.set_ylabel("Body Pitch [°]")

    fig.savefig('Images/climbing_classifier.pdf', bbox_inches='tight')


def plot_swing_stance(ax, fig, x, legs):
    ax.set_xticks(x)
    labels = [item.get_text() for item in ax.get_xticklabels()]
    labels[:] = legs
    ax.set_xticklabels(labels)
    ax.set_ylabel("n_sw/(n_sw+n_st)")
    labels = ['None', 'Vel-', 'Vel+', 'Pos-', 'Pos+']
    legend_elements = [Line2D([0], [0], marker='o', color='w', label=labels[i], markerfacecolor=colors[i], markersize=7)
                       for i in range(len(labels))]
    ax.legend(handles=legend_elements, loc='upper center', bbox_to_anchor=(1.12, 0.6))
    fig.tight_layout(pad=0.5)
    fig.savefig('Images/swing_stance.pdf')


def plot_pitch_estimation(ax, fig):
    ax.legend(['model', 'ground_truth'])

    fig.savefig('Images/pitch_estimation.pdf')
