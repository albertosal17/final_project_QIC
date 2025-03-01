from qiskit import  transpile,QuantumCircuit
from qiskit_aer import AerSimulator

import matplotlib.pyplot as plt
import numpy as np

def qiskit_quantum_circuit_GRN(theta, draw=False):

    # discarding angles with values less than 0,5 degree
    theta = np.where(theta > (np.pi*0.5)/180, theta, 0)
    nr_qubits = theta.shape[0]

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

    if draw:
        qc.draw("mpl")  
        plt.show()
    
    return qc

def run_Aer_simulation(quantum_circuit, N_trial=8192):
    simulator = AerSimulator()

    qc_compiled = transpile(quantum_circuit, simulator)  # Optimized for the simulator

    job = simulator.run(qc_compiled, shots=N_trial)  # CONTROLLA, LORO SETTANO shots=8192

    result = job.result()
    counts = result.get_counts(quantum_circuit)  # Get measurement results

    return counts


