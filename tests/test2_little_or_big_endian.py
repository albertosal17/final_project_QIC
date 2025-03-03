import numpy as np
import pandas as pd

from qiskit_aer import AerSimulator
from qiskit import  transpile,QuantumCircuit
from qiskit.visualization import plot_histogram

from qiskit.quantum_info import Statevector  
from qiskit.visualization import plot_bloch_multivector

import sys
sys.path.append("../scripts/")  # path dove sono contenuti i moduli che importo (la cartella precedente)
from gates import *
from utils import *
from quantum_circuit import quantum_circuit_GRN

debug=True

# WE KNOW QISKIT REPRESENTS IN LITTLE ENDIAN




# Reading the stored result for the activation ratio of each gene
act_ratios_df = pd.read_csv('../act_ratios.csv', header=None) 

# Separating the indices from the values
act_ratios = act_ratios_df[1].values[:3]

# Initial theta matrix: 0 for all the off-diagonal elements
theta = np.diag(act_ratios) 

nr_qubits = theta.shape[0]

##########################################
## NUMERICAL 

qc_opt = quantum_circuit_GRN(theta)

# Computing the initial state, with all qubits initialized to state |0>
psi_q0 = np.array([1,0], dtype=complex) #state |0>
psi_in = psi_q0
for ii in range(nr_qubits-1): # tensorizing the state |0> for each qubit
    psi_in = np.kron(psi_in, [1,0])

    
# Computing output state from the circuit
psi_out = np.dot(qc_opt, psi_in)

if debug:
    print('---------------------------------------------\nNumerical:')
    print(psi_out)

##########################################
## QISKIT

qc = QuantumCircuit(nr_qubits,nr_qubits) #ALL QUBITS INITIALIZED IN THE STATE |0> OF THE COMPUTATIONAL BASIS


#Defining the "encoding layer" L_enc
for ii in range(nr_qubits):
    qc.ry(theta[ii,ii], ii) 
qc.barrier() 

#Defining the correlation layers
for kk in range(nr_qubits):
    #Defining the L_k layer: it uses k-th qubit as control
    for pp in range(nr_qubits):
        if pp != kk:
            qc.cry(theta[kk,pp], kk, pp) #Documentation: cry(theta, control, target)
    qc.barrier()

psi_out_qiskit = Statevector(qc).data 
if debug:
    print('---------------------------------------------\nQiskit:')
    print(psi_out_qiskit)
    print('---------------------------------------------')


if not np.allclose(psi_out_qiskit, psi_out):
    print('Vectors are different; the numerical psi_out is surely not little endian')

    psi_out_little = big_to_little_endian_vector_state(psi_out)
    if debug:
        print('---------------------------------------------\nNumerical reordered to little-endian (in the hypotesis was big-endian):')
        print(psi_out_little)
        print('---------------------------------------------\nAgain Qiskit:')
        print(psi_out_qiskit)
        print('---------------------------------------------')

    if np.allclose(psi_out_qiskit,psi_out_little):
        print('---------------------------------------------\nConclusion:')
        
        if debug:

            print("The numerical vector is represented in big endian notation")
    
    else:
        if debug:
            print("The vectors represents different states")

else:
    print('---------------------------------------------\nConclusion:')
    print("The numerical vector is represented in little endian notation")

