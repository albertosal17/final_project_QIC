import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from qiskit_aer import AerSimulator
from qiskit import  transpile,QuantumCircuit
from qiskit.visualization import plot_histogram

from qiskit.quantum_info import Statevector  
from qiskit.visualization import plot_bloch_multivector

from utils import Laplace_smoothing, binary_reshuffling_indeces, binary_labels
from debugger_module import *
from quantum_circuit import quantum_circuit_GRN
from loss import loss_function, gradient_loss

debug = True

######################################################################################
### Loading the observed data
p_obs_smooth = pd.read_csv('../p_obs_smoothed.csv',  sep='\t', dtype={0: str, 1: float}, header=None,)
# Loading the observed data
checkpoint(f"p_obs loaded", debug = debug)
p_obs_smooth = pd.Series(p_obs_smooth[1].values, index=p_obs_smooth[0].values)
print(type(p_obs_smooth))
p_obs_smooth = binary_reshuffling_indeces(p_obs_smooth)
checkpoint(f"p_obs reshuffled", debug = debug)

######################################################################################
### Initial parameters setting

# Reading the stored result regarding the activation ratios of each of the six genes
act_ratios_df = pd.read_csv('../act_ratios.csv', header=None) 
act_ratios = act_ratios_df[1].values 
genes_names = act_ratios_df[0].values
checkpoint(f"genes considered: {genes_names}\nactivation ratios: {act_ratios}", debug = debug)

# Initial theta matrix: 0 for all the off-diagonal elements
theta_in = np.diag(act_ratios) # Initial theta matrix: 0 for all the off-diagonal elements
checkpoint(f"theta_in loaded, with shape {theta_in.shape}", debug = debug)

#############################################################################################
### OPTIMIZATION PROCESS: FINDING THE BEST THETA

iterations = 1 # Number of iterations
learn_rate = 0.1 # Learning rate
rescale_factor = 1. # Rescale factor DI PROVA
loss_treshold = 0.1
nr_qubits = theta_in.shape[0] # One qubit per gene

# intial value for the matrix of parameters
theta = theta_in

# Loop over the iterations
for iter in range(iterations):
    ###
    ### QUANTUM CIRCUIT AND OUTPUT STATE
    ###
    # Construct the matrix associate with the quantum circuit 
    qc_opt = quantum_circuit_GRN(theta)
    checkpoint(f"Quantum circuit built (shape operator {qc_opt.shape})", debug = debug)

    psi_q0 = np.array([1,0], dtype=complex) #state |0>
    psi_in = psi_q0
    for ii in range(nr_qubits-1): #tensorizing the state |0> for each qubit
        psi_in = np.kron(psi_in, [1,0])
    checkpoint(f"Initial state computed (shape {psi_in.shape})", debug = debug)

    
    psi_out = np.dot(qc_opt, psi_in)
    checkpoint(f"Output state computed (shape {psi_out.shape})", debug = debug)

    ####
    ####OUTPUT PROBABILITY DISTRIBUTION
    ####
    # Computing the probability distribution associated to the output state
    p_out = np.abs(psi_out)**2
    checkpoint(f"Output probabilities computed (shape {p_out.shape})", debug = debug)

    # DUBBIO
    # Per ottenere le sequenze binarie associate alle probabilità è giusto convertire gli
    # indici da decimale in binario? E' giusto l'ordinamento così?
    p_out = pd.Series(p_out, index=binary_labels(p_out.shape[0])) # Converto in una pandas series, associandoci le labels
    N_out = p_out.shape[0]
    checkpoint(f"p_out computed, with {N_out} entries", debug = debug)
    p_out.pop('000000')
    N_out = p_out.shape[0]
    checkpoint(f"Popped out the non-relevant entries associated to 000000 label, now the entries are {N_out}", debug = debug)
    p_out = p_out / p_out.sum() #renormalization
    print(type(p_out))
    checkpoint(f"P_out renormalized: sum is {p_out.sum()}", debug = debug)
    # Check if psi_out is real, otherwise the formula to compute the gradiend used next is not valid
    if not np.all(np.isreal(psi_out)):
        raise ValueError("The output state of the quantum circuit is not real. The gradient cannot be computed.")

    # Laplace smoothing
    # CONTROLLA, NON SONO PER NIENTE SICURO DI USARE PER N TRIALS IL NUMERO DI CELLULE
    # CHE VIENE DAL PREPROCESS (DEVO FARE COSI IN QUANTO HO LE PROBABILITA' DATE DALLA MECCANICA QUANTISTICA, NON DA UNA SERIE DI MISURE)
    #PIUTTOSTO EVITEREI DI FARE IL LAPLACE SMOOTH
    n_cells = 24828 
    p_out_smooth = Laplace_smoothing(distribution=p_out, N_trials=24828, N_classes=N_out)
    checkpoint(f"Smoothing of p_out computed (check normalization: {p_out_smooth.sum()})", debug = debug)
    p_out_smooth = binary_reshuffling_indeces(p_out_smooth)
    checkpoint(f"p_out_smooth reshuffled", debug = debug)

    ###
    ### LOSS FUNCTION
    ###
    # Check that p_obs_smooth and p_out_smooth have the same dimension and same labels
    if not p_out_smooth.index.equals(p_obs_smooth.index): # both the values and the order of the indices
        raise ValueError(f"p_out_smooth and p_obs_smooth should have the same indices!")
    else:
        checkpoint(f"Both the distributions have the same labels", debug=debug)
        
    loss = loss_function(p_obs_smooth, p_out_smooth, theta, rescale_factor) 
    checkpoint(f"loss computed: {loss}", debug=debug)
   
    if loss < loss_treshold:
        break
     
    ###
    ### GRADIENT LOSS 
    ###   
    loss_gradient = gradient_loss(sequences=p_out_smooth.index, theta=theta, psi_in=psi_in, psi_out=psi_out, p_out=p_out_smooth, p_obs=p_obs_smooth, rescale_factor=rescale_factor)
    checkpoint(f"gradient computed: {loss_gradient.shape}", debug=debug)
    print(loss_gradient)
    #CONTROLLA:  NON E' ZERO SULLA DIAGONALE
    
    ###
    ### UPDATING THETA
    ###   
    #theta = theta -
    
    print('--------------------------------------------\n')



p_out_smooth.plot(kind='bar', width=1, label='p_out', color='blue', edgecolor='lightblue', alpha=0.5)
p_obs_smooth.plot(kind='bar', width=1, label='p_obs', color='red', edgecolor='tomato', alpha=0.5)

plt.xticks(rotation=90)
plt.ylabel('Probability')
plt.xlabel('Gene sequences')
plt.title('Distributions of genes sequences')
plt.legend()
plt.show()




#############################################################################################
### CIRCUIT SIMULATION, WITH OPTIMIZED THETA
raise ValueError("The following code is not implemented yet")

best_theta = theta # The best theta found during the optimization process

qc = QuantumCircuit(nr_qubits,nr_qubits) #ALL QUBITS INITIALIZED IN THE STATE |0> OF THE COMPUTATIONAL BASIS


#Defining the "encoding layer" L_enc
for ii in range(nr_qubits):
    qc.ry(best_theta[ii,ii], ii) 
qc.barrier() 

#Defining the correlation layers
for kk in range(nr_qubits):
    #Defining the L_k layer: it uses k-th qubit as control
    for pp in range(nr_qubits):
        if pp != kk:
            qc.cry(best_theta[kk,pp], kk, pp) #Documentation: cry(theta, control, target)
    qc.barrier()

#qc.draw("mpl")  
#plt.show()

N_trials = 24828 #The number of cells considered (comes from the preprocess section)
simulator = AerSimulator()

qc_compiled = transpile(qc, simulator)  # Optimized for the simulator

job = simulator.run(qc_compiled, shots=N_trials)  # CONTROLLA, LORO SETTANO shots=8192

result = job.result()
counts = result.get_counts(qc)  # Get measurement results

if '000000' in counts:
    counts.pop('000000')  # We are not interested in the case in which

# Retrieving p_out as an array
counts_values = np.array(list(counts.values()))
counts_keys = np.array(list(counts.keys()))

# Computing the probabilities associated to the counts and applying Laplace smoothing to the distribution
p_out = Laplace_smoothing(counts_values, N_trials=counts_values.sum(), N_classes=counts_values.shape[0])
#CHECK NORMALIZATION
if np.abs(p_out.sum() - 1) > 1e-10:
    raise ValueError('The output distribution is not normalized')

#Converting the distribution to a pandas Series is needed to use binary_reshuffling_indeces function
p_out = pd.Series(p_out, index=counts_keys)
p_out = binary_reshuffling_indeces(p_out)

ax = p_out.plot(kind='bar', label='p_out', color='olive', edgecolor='black', alpha=0.7)

plt.xticks(rotation=90)
plt.ylabel('Probability')
plt.xlabel('Gene sequences')
plt.title('Output distribution of genes sequences')
plt.legend()
plt.show()

#plot_histogram(counts, figsize=(10,5), color = 'purple')  # Plot results


