from qiskit import  transpile,QuantumCircuit
from qiskit_aer import AerSimulator
from qiskit.visualization import plot_histogram

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from utils import retrieve_last_theta, Laplace_smoothing, binary_reshuffling_indeces


def qiskit_quantum_circuit_GRN(theta, draw=False):
    """
    Creates the Qiskit version of the quantum circuit simulating the Gene Regulatory Network (GRN).

    Args:
    theta : np.ndarray
        A square matrix of rotation angles (in radians) representing the coupling between qubits/genes.
    draw : bool, optional
        If True, the function will display a visualization of the circuit (default is False).

    Returns:
    QuantumCircuit
        A Qiskit quantum circuit with the specified encoding and correlation layers.

    Notes:
    - Angles smaller than 0.5 degrees are set to zero to avoid negligible rotations.
    - The circuit consists of an encoding layer (`L_enc`) and multiple correlation layers (`L_k`).
    - In `L_enc`, each qubit is rotated around the Y-axis by its corresponding `theta[i, i]`.
    - Each `L_k` layer introduces controlled rotations (`CRY`) where the k-th qubit acts as control.
    """

    # discarding angles with values less than 0.5°
    theta = np.where(theta > (np.pi*0.5)/180, theta, 0)

    # Retrieving the number of genes/qubits to be considered
    nr_qubits = theta.shape[0]

    # Building the quantum circuit using Qiskit library functions
    qc = QuantumCircuit(nr_qubits,nr_qubits) #All qubits initialized to state |0> of the computational basis

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

    qc.measure(range(nr_qubits), range(nr_qubits))  # Measure both qubits

    if draw:
        qc.draw("mpl")  
        plt.show()
    
    return qc


def run_Aer_simulation(quantum_circuit, N_trial=8192):
    """
    Runs a Qiskit quantum circuit simulation using AerSimulator.

    Args:
    quantum_circuit : QuantumCircuit
        The quantum circuit to be simulated.
    N_trial : int, optional
        The number of shots (simulated measurement repetitions) (default is 8192).

    Returns:
    dict
        A dictionary containing the measurement results (bitstring counts).

    Notes:
    - The function uses Qiskit's `AerSimulator` to run the simulation.
    """

    simulator = AerSimulator()

    qc_compiled = transpile(quantum_circuit, simulator)  # Circuit optimized for the specific simulator

    job = simulator.run(qc_compiled, shots=N_trial) 

    result = job.result()
    counts = result.get_counts(quantum_circuit)  # Get measurement results

    return counts


