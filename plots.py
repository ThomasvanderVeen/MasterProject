import matplotlib.pyplot as plt
import os
import numpy as np
from matplotlib.lines import Line2D

if not os.path.exists("Images"):
    os.makedirs("Images")

colors = ['r', 'k', 'b', 'y', 'g', 'purple']


def plot_single_hair(ax, v):
    fig = plt.figure(1, figsize=(10, 6))
    ax.set_xlabel("Time [s]")
    ax.set_ylabel("firing rate [imp/s]")

    for i in range(5):
        ax.get_lines()[i].set_color(colors[i])

    if isinstance(v, int):
        ax.legend(['15°', '23°', '34°', '46°', '60°'])
        fig.tight_layout(pad=0.5)
        fig.savefig('Images/angle.png')
    else:
        ax.legend(['24 °/s', '47 °/s', '88 °/s', '151 °/s', '245 °/s'])
        fig.tight_layout(pad=0.5)
        fig.savefig('Images/angular_velocity.png')

    fig.clear()

    return


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
        ax1.legend(['Ventral Response', 'Dorsal Response'])

    ax2.set_ylabel("Joint Angle [°]")

    fig.tight_layout(pad=0.5)
    fig.savefig('Images/position_interneuron_' + str(name) + '.png')
    fig.clear()

    return


def plot_spike_timing(ax1, ax2, fig, n_index):

    ax1.set_xlabel("time [s]")
    ax1.set_ylabel("Neuron index")
    ax1.set_yticks(np.arange(n_index))
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

    fig.legend(['Stimulus', 'Up', 'Down'])
    fig.tight_layout(pad=0.5)
    fig.savefig('Images/movement_interneuron_network.png')
    fig.clear()

    return


def plot_primitive_interneuron(ax, fig):
    colors = ['blue', 'black', 'green', 'yellow', 'orange']
    labels = ['vel-pos', 'vel-vel', 'pos-pos', 'pos-vel-vel', 'vel-pos-pos']

    legend_elements = [Line2D([0], [0], marker='o', color='w', label=labels[i], markerfacecolor=colors[i], markersize=7)
                       for i in range(5)]

    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.legend(handles=legend_elements, loc='lower right')

    fig.tight_layout(pad=0.5)
    fig.savefig('Images/primitive_interneuron')
    fig.clear()

    return
