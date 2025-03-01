import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import time

from qiskit_aer import AerSimulator
from qiskit import  transpile,QuantumCircuit
from qiskit.visualization import plot_histogram

from qiskit.quantum_info import Statevector  
from qiskit.visualization import plot_bloch_multivector

from utils import Laplace_smoothing, binary_reshuffling_indeces, binary_labels, big_to_little_endian_vector_state
from debugger_module import *
from quantum_circuit import quantum_circuit_GRN
from loss import loss_function, gradient_loss
from visualization import plot_obs_vs_out_distr
from qiskit_quantum_circuit import *

debug = False


######################################################################################
### LOADING THE OBSERVED DATA

p_obs_smooth = pd.read_csv('../p_obs_smoothed.csv',  sep='\t', dtype={0: str, 1: float}, header=None,)
checkpoint(f"p_obs loaded", debug = debug)

#converting single column dataframe into a series
p_obs_smooth = pd.Series(p_obs_smooth[1].values, index=p_obs_smooth[0].values)

#ordering the indices-sequences in ascending order as binary numbers
p_obs_smooth = binary_reshuffling_indeces(p_obs_smooth)
checkpoint(f"p_obs reshuffled", debug = debug)

######################################################################################
### Initial parameters setting

# Reading the stored result for the activation ratio of each gene
act_ratios_df = pd.read_csv('../act_ratios.csv', header=None) 

# Separating the indices from the values
act_ratios = act_ratios_df[1].values 
genes_names = act_ratios_df[0].values
checkpoint(f"genes considered: {genes_names}\nactivation ratios: {act_ratios}", debug = debug)

# Initial theta matrix: 0 for all the off-diagonal elements
diagonal_theta = 2*np.arcsin(act_ratios**0.5)
theta_in = np.diag(diagonal_theta) 
checkpoint(f"theta_in loaded, with shape {theta_in.shape}", debug = debug)

#############################################################################################
### OPTIMIZATION PROCESS: FINDING THE BEST THETA

# Setting algorithm's parameters
#iterations = 1087 # Number of iterations (AS PAPER)
iterations = 1087 
learn_rate = 0.05 # Learning rate (AS PAPER)
loss_treshold = 1e-6 # AS PAPER
nr_qubits = theta_in.shape[0] # One qubit per gene

add_constraint =True


n_plots = 6 # number of times the distribution is displayed along the execution

n_cells = 24828 #From observed data

# intial value for the matrix of parameters
theta = theta_in

# Computing the initial state, with all qubits initialized to state |0>
# Oss. little endian and big endian representations for this state have the same representation
psi_q0 = np.array([1,0], dtype=complex) #state |0>
psi_in = psi_q0
for ii in range(nr_qubits-1): # tensorizing the state |0> for each qubit
    psi_in = np.kron(psi_in, [1,0])
checkpoint(f"Initial state computed (shape {psi_in.shape})", debug = debug)

    
# Creating an empty dataframe to store the entries of the matrix theta and the loss at each iteration
rows, cols = nr_qubits, nr_qubits 
column_names = [f"θ_{i},{j}" for i in range(rows) for j in range(cols)]
column_names.append('loss_total')
column_names.append('loss_kl')
df = pd.DataFrame(columns=column_names)

for iter in range(iterations): #for each iteration of the algorithm
    
    checkpoint(f"iteration: {iter}", debug=True)
    start_time = time.time() # recording initial time

    #############################################
    ### BUILDING QUANTUM CIRCUIT AND OUTPUT STATE
    ###

    # Computing the matrix associated with the quantum circuit 
    qc_opt = quantum_circuit_GRN(theta)
    checkpoint(f"Quantum circuit built (shape operator {qc_opt.shape})", debug = debug)

    # Computing output state from the circuit
    psi_out = np.dot(qc_opt, psi_in) #big-endian ordered (see test 2)
    checkpoint(f"Output state computed (shape {psi_out.shape})", debug = debug)
    
    psi_out = big_to_little_endian_vector_state(psi_out)
    checkpoint(f"Output state reordered as little endian", debug = debug)
    
    
    if not np.abs(np.linalg.norm(psi_out) - 1) < 1e-6:
        raise ValueError(f"psi_out is not normalized: norm is {np.linalg.norm(psi_out)}\n CONSIDERA DI NORMALIZZARLO")
    
    # Check if psi_out is real, otherwise the formula to compute the gradiend used next is not valid
    if not np.all(np.isreal(psi_out)):
        raise ValueError("The output state of the quantum circuit is not real. The gradient cannot be computed.")


    #############################################
    #### OUTPUT PROBABILITY DISTRIBUTION
    #### Computed from the theoretical output state as the absolute value of its entries
    ####

    # Computing the probability distribution associated to the output state
    p_out = np.abs(psi_out)**2
    checkpoint(f"Output probabilities computed (shape {p_out.shape})", debug = debug)

    # Associating to the probabilities the binary strings representing the state of the system
    # I do this considering that the vector is now represented in little-endian representation 
    # So if the vector is [c_0,c_1,c_2, .. , c_N]
    # c_0 represents state |00....0> 
    # c_1 represents state |00...01> 
    # c_2 represents state |00..010> 
    # c_N represents state |11....1>
    # Function bynary labels covert integers to binary strings so with that you are okay 
    p_out = pd.Series(p_out, index=binary_labels(p_out.shape[0], big_endian=False))  
    checkpoint(f"Labels associated with the distribution, following little-endian ordering", debug = debug)

    # removing entry associated with state in which each qubit is |0>: it represent the situation
    # in which each gene is inactivated, and it is not interesting for this study
    p_out.pop('000000') 
    checkpoint(f"Popped out the non-relevant entries associated to 000000 label, now the entries are {p_out.shape[0]}", debug = debug)
    
    # Renormazlizing
    p_out = p_out / p_out.sum()
    checkpoint(f"P_out renormalized: sum is {p_out.sum()}", debug = debug)

    # Laplace smoothing on p_out
    p_out_smooth, N_out_smooth = Laplace_smoothing(distribution=p_out, N_trials=n_cells)
    checkpoint(f"Smoothing of p_out computed (check normalization: {p_out_smooth.sum()})", debug = debug)
    # p_out_smooth = binary_reshuffling_indeces(p_out_smooth)
    # checkpoint(f"p_out_smooth reordered ", debug = debug)

    ###
    ### LOSS FUNCTION
    ###
    # Check that p_obs_smooth and p_out_smooth have the same dimension and same labels
    if not p_out_smooth.index.equals(p_obs_smooth.index): # both the values and the order of the indices
        raise ValueError(f"p_out_smooth and p_obs_smooth should have the same indices!")
    else:
        checkpoint(f"Both the distributions have the same labels", debug=debug)
    
    kull_leibl, loss_total, rescale_factor  = loss_function(observed_pdf=p_obs_smooth, output_pdf=p_out_smooth, theta=theta, add_constraint=add_constraint) 
    
    checkpoint(f"loss computed: {kull_leibl} (kull_leibl), {loss_total} (total)", debug=True)
    
    if add_constraint:
        if iter>0 and ( np.abs( loss_total - df.loc[iter-1]['loss_total'] ) < loss_treshold ):
            checkpoint("Loss threshold achieved")
            break
    else:
        if iter>0 and ( np.abs( kull_leibl - df.loc[iter-1]['loss_kl'] ) < loss_treshold ):
            
            checkpoint("Loss threshold achieved")
            print(kull_leibl)
            print(df.loc[iter-1]['loss_kl'])
            
            break
     
    ###
    ### GRADIENT LOSS 
    ###   
    loss_gradient = gradient_loss(sequences=p_out_smooth.index, theta=theta, psi_in=psi_in, psi_out=psi_out, p_out=p_out_smooth, N_out_smooth=N_out_smooth, p_obs=p_obs_smooth, rescale_factor=rescale_factor, add_constraint=add_constraint, debug=debug)
    checkpoint(f"gradient computed: {loss_gradient.shape}", debug=debug)
    print(loss_gradient)
    
    ###
    ### UPDATING THETA
    ###   
    theta = theta - learn_rate * ( loss_gradient + loss_gradient.T ) / 2
    checkpoint("theta updated", debug=debug)
    print(theta)

    theta_values = theta.flatten()
    results = np.append(theta_values, np.array([loss_total, kull_leibl]))
    df.loc[iter] = results
    stop_time = time.time()
    print(stop_time-start_time, " seconds")

    if iter in np.arange(n_plots)*int(iterations/n_plots):
        plot_obs_vs_out_distr(p_obs_smooth, p_out_smooth, limit=0.01)

df.to_csv(r'C:\Users\ASUS\Desktop\INFO QUANTISTICA\progettoQuantum\optimization_results.csv')

plot_obs_vs_out_distr(p_obs_smooth, p_out_smooth, limit=0.01, save=True) #final plot



#############################################################################################
### CIRCUIT SIMULATION, WITH OPTIMIZED THETA
raise ValueError("The following code is not implemented yet")

best_theta = theta # The best theta found during the optimization process
qc = qiskit_quantum_circuit_GRN(best_theta)

N_cells = 24828 #The number of cells considered (comes from the preprocess section)
counts = run_Aer_simulation(qc, N_cells)

if '000000' in counts:
    counts.pop('000000')  # We are not interested in the case in which all genes are not active

# Retrieving p_out as an array
counts_values = np.array(list(counts.values()))
counts_keys = np.array(list(counts.keys()))

# Computing the probabilities associated to the counts and applying Laplace smoothing to the distribution
p_out, _ = Laplace_smoothing(counts_values, N_trials=counts_values.sum())
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


