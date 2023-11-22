import matplotlib.pyplot as plt
import os
import numpy as np
from matplotlib.lines import Line2D
import matplotlib
import seaborn as sns

if not os.path.exists("Images"):
    os.makedirs("Images")

colors = ['#c1272d', '#0000a7', '#eecc16', '#008176', '#b3b3b3']
matplotlib.rc('xtick', labelsize=10)
matplotlib.rc('ytick', labelsize=10)

def plot_single_hair(ax, v):
    fig = plt.figure(1, figsize=(10, 6))
    ax.set_xlabel("Time [s]")
    ax.set_ylabel("firing rate [imp/s]")

    for i in range(5):
        ax.get_lines()[i].set_color(colors[i])

    if isinstance(v, int):
        ax.legend(['15°', '23°', '34°', '46°', '60°'])
        ax.set_yticks([0, 40, 80, 120, 160, 200, 240, 280])
        fig.tight_layout(pad=0.5)
        fig.savefig('Images/angle.png')
    else:
        ax.legend(['24 °/s', '47 °/s', '88 °/s', '151 °/s', '245 °/s'])
        ax.set_yticks([0, 40, 80, 120, 160])
        fig.tight_layout(pad=0.5)
        fig.savefig('Images/angular_velocity.png')

    fig.clear()

    return


def plot_heat_map(df):
    heatmap = sns.heatmap(data=df, annot=True, fmt='.3g', cbar_kws={'label': 'mean absolute error (MAE)'})
    heatmap.set(xlabel='b', ylabel='V_r')
    plt.tight_layout(pad=0.5)
    plt.savefig('Images/heat_map')


def plot_hair_field(ax, name):
    fig = plt.figure(1, figsize=(10, 6))
    ax.set_xlabel("joint angle [degrees]")
    ax.set_ylabel("hair angle [degrees]")

    fig.tight_layout(pad=0.5)
    fig.savefig('Images/hair_field_' + name + '.png')
    fig.clear()

    return


def plot_position_interneuron(ax1, ax2, fig, name):

    ax1.set_xlabel("time [s]")
    ax1.set_ylabel("firing rate [imp/s]")

    if name == 'uni':
        fig.legend(['Model Response', 'Exp. Data'])
    else:
        ax1.legend(['Dorsal Response', 'Ventral Response'])

    ax2.set_ylabel("Joint Angle [°]")

    fig.tight_layout(pad=0.5)
    fig.savefig('Images/position_interneuron_' + str(name) + '.png')
    fig.clear()

    return


def plot_spike_timing(ax1, ax2, fig, n_index):

    ax1.set_xlabel("time [s]")
    ax1.set_ylabel("Neuron index")
    ax1.set_yticks(np.arange(1, n_index+1))
    ax2.set_ylabel("Joint Angle [°]")

    fig.tight_layout(pad=0.5)
    fig.savefig('Images/spike_timing_.png')
    fig.clear()

    return


def plot_movement_interneuron(ax, fig):

    ax.set_xlabel("Velocity [°/s]")
    ax.set_ylabel("firing rate [imp/s]")

    fig.tight_layout(pad=0.5)
    fig.savefig('Images/movement_interneuron.png')
    fig.clear()

    return


def plot_movement_interneuron_network(ax, fig):

    ax.set_xlabel("Time [s]")
    ax.set_ylabel("Joint Angle [°]")

    fig.legend(['Stimulus', 'Down', 'Up'])
    fig.tight_layout(pad=0.5)
    fig.savefig('Images/movement_interneuron_network.png')
    fig.clear()

    return


def plot_primitive_ROC(ax, fig):
    colors = ['blue', 'black', 'green', 'yellow', 'orange']
    labels = ['vel-pos', 'vel-vel', 'pos-pos', 'pos-vel-vel', 'vel-pos-pos']

    legend_elements = [Line2D([0], [0], marker='o', color='w', label=labels[i], markerfacecolor=colors[i], markersize=7)
                       for i in range(5)]

    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.legend(handles=legend_elements, loc='lower right')

    fig.tight_layout(pad=0.5)
    fig.savefig('Images/primitive_ROC')
    fig.clear()

    return


def plot_psth(ax, fig, neuron, leg):
    ax.set_ylabel('Likelihood of Spiking')
    fig.text(0.33, 0.04, 'Swing', ha='center')
    fig.text(0.66, 0.04, 'Stance')
    ax.set_xticks([])
    fig.savefig(f'Images_PSTH/neuron_{neuron}_leg_{leg}')
    plt.cla()

    return


def plot_primitive_accuracy(ax, fig, tau_list):
    ax.set_xlabel('τ [ms]', fontsize=15)
    ax.set_ylabel("Matthews correlation", fontsize=15)
    ax.set_xticks(1000*tau_list[::2])
    ax.legend(['v-p', 'v-v', 'p-p', 'p-v-v', 'v-p-p', 'mean'], loc='lower right')
    fig.tight_layout(pad=0.5)
    fig.savefig('Images/primitive_accuracy')

    return


def plot_primitive_weights(ax, fig, tau_list, w_1, w_2):
    ax[0].grid(True, axis='y')
    ax[1].grid(True, axis='y')
    ax[0].set_xticks([])
    ax[1].set_xticks(1000*tau_list[::2])
    ax[0].set_yticks(1000*w_1[::2])
    ax[1].set_yticks(1000*w_2[::2])
    ax[0].set_ylabel('w_pos [mV]', fontsize=15)
    ax[1].set_ylabel('w_vel [mV]', fontsize=15)
    fig.legend(['v-p', 'v-v', 'p-p', 'p-v-v', 'v-p-p'], loc='upper center', bbox_to_anchor=(1.08, 0.75))
    fig.supxlabel('τ [ms]', fontsize=15)
    fig.tight_layout(pad=0.5)
    fig.savefig('Images/primitive_weights', bbox_inches='tight')

    return


def plot_climbing_accuracy(fig, ax):
    ax.set_xlabel('w_exc (mV)')
    ax.set_ylabel('Accuracy')
    ax.plot()
    ax.legend()

    fig.savefig('Images/climbing_accuracy', bbox_inches='tight')


def plot_climbing_classifier(fig, ax):
    ax.set_ylim([-20, 75])
    ax.set_xlim([0, 20])
    ax.legend(['body pitch', 'divide', 'spikes', 'climbing'])
    ax.set_xlabel("Time [s]")
    ax.set_ylabel("Body Pitch [°]")

    fig.savefig('Images/climbing_classifier', bbox_inches='tight')
