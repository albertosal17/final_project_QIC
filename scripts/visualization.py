import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

color_cycle = plt.cm.Set1.colors  # Setting a predefined colormap for the plots

def plot_obs_vs_out_distr(observed_distr: pd.Series, output_distr: pd.Series, qiskit_distr= None, image_name="distributions_comparison", save = False, limit=0.):
    '''
    This function plots the 'observed distribution' (from experimental data) of gene sequences vs the analogous
    'output distribution', the probability distribution associated with the output state from the quantum circuit. 
    Optionally one can impose a limit on the heights of the observed distribution bars. From this operation only a subset
    of gene sequences probabilites will be represented. The function is written so that from the output distribution
    only these sequences are displayed.

    Additionaly, if passed as argument, the function superimpose the distribution obtained from the Qiskit simulation
    computed with the optimized value of theta, computed during training.

    Args:
    observed_distr: pd.Series, 
        the observed distribution
    output_distr: pd.Series, 
        the output distribution
    qiskit_distr: pd.Series or None,
        if it is not None, it is the distribution obtained from qiskit simulation 
    save: bool, 
        determines wether to save the plot or not
    filename: str,
        the eventual name for the image to be saved
    limit: 
        determines the minimum height for the bars of the observed distribution
    ''' 

    plt.figure(figsize=(15,8))

    # Filtering the observed distribution
    mask = observed_distr>limit
    observed_distr = observed_distr[mask]
    selected_indices = observed_distr.index  # Get indices where p_obs > limit

    # Retrieving the output distribution sequences correspondin to the filtered observed distribution sequences
    new_output_distr = pd.Series({idx: output_distr.get(idx, 0) for idx in selected_indices}, index=selected_indices)

    # Plotting
    new_output_distr.plot(kind='bar', width=0.8, label='p_out', color='red', edgecolor='tomato', alpha=0.5)
    observed_distr.plot(kind='bar', width=0.8, label='p_obs', color='blue', edgecolor='lightblue', alpha=0.5)

    if qiskit_distr is not None:
        new_qiskit_distr = pd.Series({idx: qiskit_distr.get(idx, 0) for idx in selected_indices}, index=selected_indices)
        new_qiskit_distr.plot(kind='bar', width=0.8, label='p_qiskit', color='green', edgecolor='lightgreen', alpha=0.5)

    # Plot settings
    plt.xticks(rotation=90)
    plt.ylabel('Probability')
    plt.xlabel('Gene sequences')
    plt.title('Distributions of genes sequences')
    plt.grid(True)
    plt.legend()

    if save: 
        plt.savefig('../results/' + image_name + '.svg', format='svg', bbox_inches='tight')
    plt.show()



def plot_loss_evolution(filename: str, color='navy', linestyle='-', save=True):
    """
    Plots the evolution of the KL divergence loss during training.

    Args:
    filename : str
        The path to the CSV file containing the loss data.
    color : str, optional
        The color of the plot line (default is 'navy').
    linestyle : str, optional
        The style of the plot line (default is '-').
    save: bool, 
        determines wether to save the plot or not.

    Notes:
    - The function reads a CSV file and extracts the 'loss_kl' column.
    - Assumes the CSV file contains a column named 'loss_kl'.
    
    """  

    plt.figure(figsize=(13,8))
 
    # Read the CSV file
    df = pd.read_csv(filename)  

    # Retrieving the data of the column containing the KL divergence values
    kl_values = df['loss_kl']

    kl_values.plot(kind='line', color=color, linestyle=linestyle, label='loss_kl')
    plt.legend()
    plt.xlabel('Epoch')
    plt.ylabel('K-L divergence $p_{out}$-$p_{obs}$')
    plt.title('Evolution of the KL term of the loss during training')
    plt.grid()
    
    if save: 
        plt.savefig(f'../results/loss_evolution.svg', format='svg', bbox_inches='tight')


    plt.show()
                    
    
def plot_theta_i_j_evolution(filename: str, indices: list, linestyle='-', save=True, show=True):
    """
    Plots the evolution of the θ_{i,j} angle (or a set of these angles) during training.

    Parameters:
    ----------
    filename : str
        The path to the CSV file containing the angles evolution data.
    indices: list
        list of tuples of two integers. These integeres represent the row and column of a specific angle in the 
        matrix of parameters to be learned theta.
    color : str
        The color of the plot line.
    linestyle : str, optional
        The style of the plot line (default is '-')
    save: bool, 
        determines wether to save the plot or not.
    show: bool, 
        determines wether to display the plot or not.

    Notes:
    ------
    - This function reads a CSV file and extracts the θ_{i,j} column.
    - Assumes the CSV file contains columns named in the format 'θ_i,j'.

    """ 
    nn=0 #counter
    for element in indices:
        index_i, index_j  = element

        # Read the CSV file and retrieving the evolution data for the angle element in position (index_i,index_j)
        df = pd.read_csv(filename)  
        column_name = 'θ_' + str(index_i) + ',' + str(index_j)
        theta_i_j_values = df[column_name]

        # plotting
        theta_i_j_values.plot(kind='line', color=color_cycle[nn], linestyle=linestyle, label=column_name)
        nn+=1

    plt.legend()
    plt.xlabel('Epoch')
    plt.ylabel('Angles $\theta_{i,j}$'+' (rad)')
    plt.title('Parameters evolution during training')
    plt.grid()

    if show:
        if save: 
            plt.savefig(f'../results/theta_evolution{indices}.svg', format='svg', bbox_inches='tight')

        plt.show()

def adjency_matrix_heatmap(theta, save=True):

    c = plt.imshow(theta, cmap='RdYlGn', vmin = theta.min(), vmax= theta.max(),  interpolation='nearest')

    if save: 
        plt.savefig(f'../results/heatmap_theta.svg', format='svg', bbox_inches='tight')  
    
    plt.colorbar(c) 
    plt.show()

# # Example of usage:
# plot_loss_evolution('../optimization_results.csv')




# # Example of usage:
# cmap = plt.cm.Set1
# num_colors = 11  # specify the number of colors you want
# colors = [cmap(i / num_colors) for i in range(num_colors)]
# plot_theta_i_j_evolution('..optimization_results.csv', 0, 1,color=colors[0], single_plot=False)
# plot_theta_i_j_evolution('..optimization_results.csv', 1, 0, color=colors[0],linestyle='dashed', single_plot=False)
# plot_theta_i_j_evolution('..optimization_results.csv', 0, 2,color=colors[1], single_plot=False)
# plot_theta_i_j_evolution('..optimization_results.csv', 2, 0, color=colors[1], linestyle='dashed', single_plot=False)
# plot_theta_i_j_evolution('..optimization_results.csv', 0, 3,color=colors[2], single_plot=False)
# plot_theta_i_j_evolution('..optimization_results.csv', 3, 0, color=colors[2], linestyle='dashed', single_plot=False)
# plot_theta_i_j_evolution('..optimization_results.csv', 3, 1,color=colors[3], single_plot=False)
# plot_theta_i_j_evolution('..optimization_results.csv', 1, 3, color=colors[3], linestyle='dashed', single_plot=False)
# plot_theta_i_j_evolution('..optimization_results.csv', 4, 5,color=colors[4], single_plot=False)
# plot_theta_i_j_evolution('..optimization_results.csv', 5, 4, color=colors[4], linestyle='dashed', single_plot=False)
# plot_theta_i_j_evolution('..optimization_results.csv', 0, 0,color=colors[5], single_plot=False)
# plot_theta_i_j_evolution('..optimization_results.csv', 1, 1, color=colors[6], single_plot=False)
# plot_theta_i_j_evolution('..optimization_results.csv', 2, 2,color=colors[7], single_plot=False)
# plot_theta_i_j_evolution('..optimization_results.csv', 3, 3, color=colors[8], single_plot=False)
# plot_theta_i_j_evolution('..optimization_results.csv', 4, 4,color=colors[9], single_plot=False)
# plot_theta_i_j_evolution('..optimization_results.csv', 5, 5, color=colors[10], single_plot=False)

# plt.show()

