import matplotlib.pyplot as plt
import pandas as pd

def plot_obs_vs_out_distr(observed_distr: pd.Series, output_distr: pd.Series, save = False, limit=0.):

    mask = observed_distr>limit
    observed_distr = observed_distr[mask]

    selected_indices = observed_distr.index  # Get indices where p_obs > limit
    new_output_distr = pd.Series({idx: output_distr.get(idx, 0) for idx in selected_indices}, index=selected_indices)

    
    new_output_distr.plot(kind='bar', width=0.8, label='p_out', color='red', edgecolor='tomato', alpha=0.5)
    observed_distr.plot(kind='bar', width=0.8, label='p_obs', color='blue', edgecolor='lightblue', alpha=0.5)

    plt.xticks(rotation=90)
    plt.ylabel('Probability')
    plt.xlabel('Gene sequences')
    plt.title('Distributions of genes sequences')
    plt.grid(True)
    plt.legend()

    if save: 
        plt.savefig('../p_obsVSp_out_final.png')
    plt.show()



def plot_loss_evolution(filename: str, color='navy', linestyle='-'):
        # Read the CSV file
    df = pd.read_csv(filename)  # Adjust 'your_file.csv' to your file path

    column_name = 'loss_kl'

    # Access a specific column (e.g., the first column, index 0)
    column_data = df[column_name]

    column_data.plot(kind='line', color=color, linestyle=linestyle, label=column_name)
    plt.legend()
    plt.xlabel('Epoch')
    plt.ylabel('K-L divergence p_out vs p_obs')
    plt.title('KL term of the loss evolution during training')
    plt.grid()

    plt.show()

# # Example of usage:
# plot_loss_evolution('../optimization_results.csv')
                    
    


def plot_theta_i_j_evolution(filename: str, index_i: int, index_j: int, color:str, linestyle='-', single_plot=True):
        # Read the CSV file
    df = pd.read_csv(filename)  # Adjust 'your_file.csv' to your file path

    column_name = 'θ_' + str(index_i) + ',' + str(index_j)

    # Access a specific column (e.g., the first column, index 0)
    column_data = df[column_name]

    column_data.plot(kind='line', color=color, linestyle=linestyle, label=column_name)
    plt.legend()
    plt.xlabel('epoch')
    plt.ylabel('theta_i_j'+' (rad)')
    plt.title('Angle evolution during training')
    plt.grid()
    
    if single_plot:
        plt.show()


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

