import matplotlib.pyplot as plt
import igraph as ig
import numpy as np
import pandas as pd

from utils import return_unique_edges


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
    x = np.arange(len(selected_indices))
    if qiskit_distr is not None:
            # Retrieving the qiskit distribution sequences corresponding to the filtered observed distribution sequences
            new_qiskit_distr = pd.Series({idx: qiskit_distr.get(idx, 0) for idx in selected_indices}, index=selected_indices)
            
            #plotting            
            bar_width = 0.3
            plt.bar(x, new_qiskit_distr, width=bar_width, label="p_qiskit", color="green", edgecolor="green", alpha=0.6)
            plt.bar(x + bar_width, new_output_distr, width=bar_width, label="p_out", color="red", edgecolor="tomato", alpha=0.6)
            plt.bar(x + 2 * bar_width, observed_distr, width=bar_width, label="p_obs", color="blue", edgecolor="lightblue", alpha=0.6)
            plt.title('Distributions of genes sequences \n(experimental vs. numerical simulation vs. qiskit simulation)')

    else:
        bar_width = 0.4
        plt.bar(x + bar_width, new_output_distr, width=bar_width, label="p_out", color="red", edgecolor="tomato", alpha=0.6)
        plt.bar(x + 2 * bar_width, observed_distr, width=bar_width, label="p_obs", color="blue", edgecolor="lightblue", alpha=0.6)
        plt.title('Distributions of genes sequences \n(experimental vs. numerical simulation)')


    # Fix x-axis labels
    plt.xticks(x + bar_width, selected_indices, rotation=45)  # Center the labels and rotate for readability
    plt.ylabel('Probability')
    plt.xlabel('Gene sequences')
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
        dictionary. The keys are tuples of two integers univoquely associated to two genes. 
        These integeres represent the row and column of a specific angle in the matrix of 
        parameters theta. The values of the dictionary are tuples with the associated names 
        of the genes.
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
    plt.figure(figsize=(10,6))
    nn=0 #counter
    for tuple_indices, tuple_names in indices.items():
        index_i, index_j  = tuple_indices
        
        # Read the CSV file and retrieving the evolution data for the angle element in position (index_i,index_j)
        df = pd.read_csv(filename)  
        column_name = 'θ_' + str(index_i) + ',' + str(index_j)
        theta_i_j_values = df[column_name]

        # plot only if meaningful angles
        boolean_mask = np.abs(theta_i_j_values) > 0.03
        if boolean_mask.sum()>1:
            # plotting
            theta_i_j_values.plot(kind='line', color=color_cycle[nn], linestyle=linestyle, label=tuple_names)
            nn+=1
    plt.legend()
    plt.xlabel('Epoch')
    plt.ylabel(r'Angle $\theta_{i,j}$ (rad)')
    plt.title('Parameters evolution during training')
    plt.grid()

    if show:
        if save: 
            plt.savefig(f'../results/theta_evolution.svg', format='svg', bbox_inches='tight')

        plt.show()

def adjency_matrix_heatmap(matrix, genes_names, filename='heatmap_theta',save=True):
    """
    Generates and displays a heatmap for the matrix of parameters theta.
    
    Args:
    matrix : np.ndarray
        A 2D square numpy array representing the adjacency matrix.
    genes_names : np.ndarray
        A 1D numpy array containing the names of the genes (used as labels for x and y axes).
    filename : str, optional
        The filename to save the heatmap as an SVG file (default is 'heatmap_theta').
    save : bool, optional
        If True, saves the heatmap as an SVG file in the '../results/' directory (default is True).
    
    Notes:
    ------
    - The diagonal elements of the matrix (where `matrix == 0`) are masked and shown in black.
    - The colormap used is `RdYlGn`.
    - The values of the matrix are displayed on top of the heatmap with 3 decimal precision.
    """    
    plt.figure(figsize=(8,8))
    # Setting the colormap to be used
    cmap = plt.cm.RdYlGn  

    # discarding diagonal values and set their color to black
    masked_array = np.ma.masked_where(matrix == 0., matrix)
    cmap.set_bad(color='black')

    # plotting heatmap
    plt.imshow(masked_array, cmap=cmap, vmin = matrix.min(), vmax= matrix.max(),  interpolation='nearest')

    #displaying values on top of the heatmap
    for ii in range(matrix.shape[0]):
        for jj in range(matrix.shape[1]):
            plt.text(jj, ii, str(np.round(matrix[ii, jj],3)), ha='center', va='center', color='black', fontsize=10)
    
    plt.colorbar(shrink=0.65)  # Show color scale
    plt.xticks(range(genes_names.shape[0]), list(genes_names))
    plt.yticks(range(genes_names.shape[0]), list(genes_names))
    plt.title('Coupling angles after optimization (radians)')
    plt.xlabel('Gene names')
    plt.ylabel('Gene names')

    # Eventually save the plot
    if save: 
        plt.savefig(f'../results/{filename}.svg', format='svg', bbox_inches='tight')  

    plt.show()

def plot_GRN(theta, genes_names, save=True):
    """
    Plots a Gene Regulatory Network (GRN) from an adjacency matrix of coupling angles.
    
    Args:
    theta : np.ndarray
        The matrix with the coupling angles between qubits/genes
    genes_names : np.ndarray
        A 1D numpy array containing gene names, used as vertex labels.
    save : bool, optional
        If True, saves the plot as an SVG file in the '../results/' directory (default is True).
    
    Notes:
    ------
    - The edges are weighted by the average of symmetric matrix entries: (theta[i,j] + theta[j,i]) / 2.
    - Green edges represent positive coupling angles, and red edges represent negative ones.    
    """
    # Discarding simmetric the pairs of genes ("edges")
    # e.g. for [0,1] and [1,0] we keep only [1,0]
    unique_edges = return_unique_edges(genes_names)

    # For each unique pair, compute the average value between the two simmetric entries
    # e.g. for [0,1] and [1,0] we keep only [0,1] and associate to it the angle (theta[0,1]+theta[1,0])/2
    edges_theta_dict = {}
    for edge in unique_edges:
        i,j=edge
        edges_theta_dict[edge]=(theta[i,j]+theta[j,i])/2
    
    color_cycle = plt.cm.Pastel1.colors  # Setting a predefined colormap for the plots

    # Construct a graph with 5 vertices
    n_vertices = 6
    g = ig.Graph(n_vertices, unique_edges)

    # Set attributes for the nodes, and edges
    g.vs["name"] = genes_names
    g.es["strengths"] = np.array(list(edges_theta_dict.values()))

    # Plot in matplotlib
    # Note that attributes can be set globally (e.g. vertex_size), or set individually using arrays (e.g. vertex_color)
    fig, ax = plt.subplots(figsize=(5,5))
    ig.plot(
        g,
        target=ax,
        layout="circle", # print nodes in a circular layout
        vertex_size=50,
        vertex_color=color_cycle[:n_vertices],
        vertex_frame_width=2,
        vertex_frame_color="black",
        vertex_label=g.vs["name"],
        vertex_label_size=7.0,
        edge_width=np.dot(g.es["strengths"],10),
        edge_color=["green" if theta_ij>0 else "red" for theta_ij in g.es["strengths"]]
    )
    plt.title("Gene Regulatory Network recovered")

    if save:
        plt.savefig(f'../results/GRN.svg', format='svg', bbox_inches='tight')  
    plt.show()









