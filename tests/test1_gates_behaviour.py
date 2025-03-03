# Testo che le gates che ho definito manualmente funzionino come previsto 
# dalle loro tabelle di verità

import numpy as np

import sys
sys.path.append("../scripts/")  # path dove sono contenuti i moduli che importo (la cartella precedente)
from gates import *
from utils import *

n_qubits = 3
qubits_states = [np.array([1,0]), np.array([1,0]), np.array([1,0])] # |000>
initial_state = tensor_product(qubits_states)

control = 0
target = 1

c_Ry = control_r_y(phi = np.pi, n_qubits=n_qubits, control=control, target=target) 
output_state = np.dot(c_Ry, initial_state)
print("Initial state: ", initial_state)
print("Output state: ", output_state)


qubits_states = [np.array([0,1]), np.array([1,0]), np.array([1,0])] # |100>
initial_state = tensor_product(qubits_states)
output_state = np.dot(c_Ry, initial_state)
print("Initial state: ", initial_state)
print("Output state: ", output_state)