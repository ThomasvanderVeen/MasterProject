from brian2 import *

# Define the Izhikevich model parameters for a bursting neuron
a = 0.02
b = 0.2
c = -65
d = 6

# Neuron model equations
eqs = '''
dv/dt = 0.04*v**2 + 5*v + 140 - u + I : 1
du/dt = a*(b*v - u) : 1
I : 1 (constant)
'''

# Initial conditions
v0 = -70
u0 = b * v0

# Create a bursting neuron group
bursting_neuron = NeuronGroup(1, model=eqs, threshold='v>=30', reset='v=c; u+=d', method='euler')

# Set initial conditions
bursting_neuron.v = v0
bursting_neuron.u = u0

# Set a current input to trigger bursting
bursting_neuron.I = 10

# Monitor the membrane potential
mon = StateMonitor(bursting_neuron, 'v', record=0)

# Run the simulation
run(100*ms)

# Plot the membrane potential
plot(mon.t/ms, mon.v[0])
xlabel('Time (ms)')
ylabel('Membrane Potential')
show()