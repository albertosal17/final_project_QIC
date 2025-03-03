import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import time

from utils import Laplace_smoothing, binary_reshuffling_indeces, return_unique_edges
from debugger_module import checkpoint
from quantum_circuit import initial_ground_state, output_state, output_probability_distribution
from loss import loss_function, gradient_loss
from visualization import plot_obs_vs_out_distr, plot_loss_evolution, plot_theta_i_j_evolution
from qiskit_quantum_circuit import qiskit_quantum_circuit_GRN, run_Aer_simulation


plt.rcParams.update({'font.size': 17}) #setting font size for the plots
color_cycle = plt.cm.Set1.colors  # Setting a predefined colormap for the plots
markers = ['o', 's', 'D', '^', '|','v', '<', '>', 'p', 'h', 'H', '+', 'x', 'd', '|', '_']

debug = True
add_constraint =True # Flag to choose wheter to include the "contraint term" in the loss (and its gradient) computations
ok_simultation = True


######################################################################################
### LOADING THE OBSERVED DATA

p_obs_smooth = pd.read_csv('../results/p_obs_smoothed.csv',  sep='\t', dtype={0: str, 1: float}, header=None,)
#checkpoint(f"p_obs loaded", debug = debug)

#Converting single column dataframe into a series
p_obs_smooth = pd.Series(p_obs_smooth[1].values, index=p_obs_smooth[0].values)

#ordering the indices-sequences in ascending order as binary numbers
p_obs_smooth = binary_reshuffling_indeces(p_obs_smooth)
#checkpoint(f"p_obs reshuffled", debug = debug)


######################################################################################
### INITIAL PARAMETERS SETTING

# Reading the stored result for the activation ratio of each gene
act_ratios_df = pd.read_csv('../results/act_ratios.csv', header=None) 

# Separating the indices from the values
act_ratios = act_ratios_df[1].values 
genes_names = act_ratios_df[0].values
checkpoint(f"genes considered: {genes_names}\nactivation ratios: {act_ratios}", debug = debug)

# Initial theta matrix: 0 for all the off-diagonal elements
diagonal_theta = 2*np.arcsin(act_ratios**0.5) # see paper
theta_in = np.diag(diagonal_theta) 
#checkpoint(f"theta_in loaded, with shape {theta_in.shape}", debug = debug)


#############################################################################################
### OPTIMIZATION PROCESS: FINDING THE BEST THETA

# Setting algorithm's parameters
iterations = 200 
learn_rate = 0.05 # Learning rate (AS PAPER)
loss_treshold = 1e-8 # set None if you do not want to use it
n_plots = 1 # number of times the distribution is displayed along the execution

nr_qubits = theta_in.shape[0] # One qubit per gene
n_cells = 24828 # From experimental data


# Matrix of parameters for the first iteration of the algorithm
theta = theta_in

psi_in = initial_ground_state(nr_qubits)
checkpoint(f"Initial state computed (shape {psi_in.shape})", debug = debug)

    
# Creating an empty dataframe to store the entries of the matrix theta and the loss at each iteration
rows, cols = nr_qubits, nr_qubits 
column_names = [f"θ_{i},{j}" for i in range(rows) for j in range(cols)]
column_names.append('loss_total')
column_names.append('loss_kl')
df = pd.DataFrame(columns=column_names)

# Algorithm execution
for iter in range(iterations): 
    
    checkpoint(f"-------------------------------------------------\niteration: {iter}")
    start_time = time.time() # recording initial time

    #############################################
    ### BUILDING QUANTUM CIRCUIT AND OUTPUT STATE
    psi_out = output_state(theta, psi_in, debug)


    #############################################
    #### OUTPUT PROBABILITY DISTRIBUTION
    #### Computed from the theoretical output state as the absolute value of its entries
    p_out_smooth, N_out_smooth = output_probability_distribution(psi_out, n_cells, debug)

    # Check that p_obs_smooth and p_out_smooth have the same dimension and same labels
    if not p_out_smooth.index.equals(p_obs_smooth.index): # both the values and the order of the indices
        raise ValueError(f"p_out_smooth and p_obs_smooth should have the same indices!")
    

    #############################################
    ### LOSS FUNCTION COMPUTATION

    # computing the loss
    # Note: if add_contraint=True the loss total and rescale_factor variable will be returned as None
    kull_leibl, loss_total, rescale_factor  = loss_function(observed_pdf=p_obs_smooth, output_pdf=p_out_smooth, \
                                                             theta=theta, add_constraint=add_constraint) 
    checkpoint(f"loss computed: \n{kull_leibl} (kull_leibl)\n{loss_total} (total)")
    
    # Checking if convergence of the algorithm is reached: loss update smaller than fixed thrreshold.
    # Otherwise iterate again, until max number of iterations is reached
    if loss_treshold is not None and iter>0:
        if add_constraint: #use the total loss 
            if iter>0 and ( np.abs( loss_total - df.loc[iter-1]['loss_total'] ) < loss_treshold ):
                checkpoint("Loss threshold achieved")
                break
        else: #use the kl loss 
            if ( np.abs( kull_leibl - df.loc[iter-1]['loss_kl'] ) < loss_treshold ):
                checkpoint("Loss threshold achieved")
            break

    ###
    ### LOSS GRADIENT COMPUTATION
    loss_gradient = gradient_loss(theta=theta, psi_in=psi_in, psi_out=psi_out, p_out=p_out_smooth, N_out_smooth=N_out_smooth, \
                                   p_obs=p_obs_smooth, rescale_factor=rescale_factor, add_constraint=add_constraint, debug=debug)
    
    checkpoint(f"gradient computed: {loss_gradient.shape}")
    print(np.round(loss_gradient,5))
    

    ###
    ### UPDATING THETA
    theta = theta - learn_rate*( loss_gradient + loss_gradient.T )/2 #updating rule
    checkpoint("theta updated")
    print(np.round(theta, 5))

    # Storing results
    theta_values = theta.flatten()
    results = np.append(theta_values, np.array([loss_total, kull_leibl]))
    df.loc[iter] = results

    # Computing and printing execution time
    stop_time = time.time()
    checkpoint(f"{stop_time-start_time} seconds")

    # Plotting p_out vs. p_obs only for certain iterations, so that overral n_plots are displayed during execution
    if iter in np.arange(n_plots)*int(iterations/n_plots):
        plot_obs_vs_out_distr(p_obs_smooth, p_out_smooth, limit=0.01, save=True, image_name=f"p_obsVSp_out_iter{iter}")

# Saving the results dataframe
results_filename = '../results/optimization_results.csv'
df.to_csv(results_filename)

# Plot and save results
plot_obs_vs_out_distr(observed_distr=p_obs_smooth, output_distr=p_out_smooth, limit=0.01, save=True, image_name="p_obsVSp_out_final") #final plot
plot_loss_evolution(filename=results_filename)


# Computing a dictionary with each pair of genes names associated to two 
# unique integers associated with the specific genes
unique_edges = return_unique_edges(genes_names)
# For each unique pair, associate the names
edges_indices_to_names = {}
for edge in unique_edges:
    i,j=edge
    edges_indices_to_names[edge]=(genes_names[i], genes_names[j])

plot_theta_i_j_evolution(filename=results_filename, indices=edges_indices_to_names)



#############################################################################################
### CIRCUIT SIMULATION, WITH OPTIMIZED THETA
if ok_simultation:
    ############################################
    ### BUILDING THE CIRCUIT
    best_theta = theta # The best theta found during the optimization proces
    qc = qiskit_quantum_circuit_GRN(theta=best_theta, draw=False) 
    checkpoint("Qiskit circuit built, starting the simulation..", debug=debug)
    counts = run_Aer_simulation(qc)
    checkpoint("Simulation end", debug=debug)

    if '000000' in counts:
        counts.pop('000000')  # We are not interested in the case in which all genes are not active

    # Retrieving p_out as an array
    counts_values = np.array(list(counts.values()))
    counts_keys = np.array(list(counts.keys()))

    p_qiskit = counts_values / np.sum(counts_values)

    #Converting the distribution to a pandas Series is needed to use binary_reshuffling_indeces function
    p_qiskit = pd.Series(p_qiskit, index=counts_keys)
    p_qiskit = binary_reshuffling_indeces(p_qiskit)

    # Computing the probabilities associated to the counts and applying Laplace smoothing to the distribution
    p_qiskit_smooth, _ = Laplace_smoothing(p_qiskit, N_trials=counts_values.sum(), filename="p_qiskit_smoothed")
    #CHECK NORMALIZATION
    if np.abs(p_qiskit_smooth.sum() - 1) > 1e-8:
        raise ValueError('The output distribution is not normalized')
    checkpoint("Probability distribution of sequences computed", debug=debug)

    plot_obs_vs_out_distr(observed_distr=p_obs_smooth, output_distr=p_out_smooth, qiskit_distr=p_qiskit_smooth, limit=0.01, save=True, image_name="p_qiskitVSp_obsVSp_out") 

