import numpy as np

from debugger_module import *

from gates import *
from utils import *
from quantum_circuit import *    


def kl(p, q):
    """
    Kullback-Leibler divergence D(P || Q) for discrete distributions
    Args:
    p, q : array-like, dtype=float, shape=n
        Discrete probability distributions.
    """
    p = np.asarray(p, dtype=float)
    q = np.asarray(q, dtype=float)

    return np.sum(np.where(p != 0, p * np.log(p / q), 0)) #numpy.where(condition, [x, y]) Return elements, either from x or y, depending on condition.

def loss_function(observed_pdf, output_pdf, theta, add_constraint):
    '''
    Loss function for the GRN estimation problem.
    It consists of the Kullback-Leibler divergence between the observed (from data) and 
    output (from quantum circuit) pdfs.
    Additionally, it includes a constraint that penalizes the output pdf for values of theta
    that are close to pi/2. This is done to avoid the output pdf to be too peaked around pi/2, 
    a critical value as discussed in the paper.
    A dynamical rescaling factor is included to balance the two terms of the loss function.

    Args:
    - observed_pdf: array-like
        The probability distribution obtained from the data.
    - output_pdf: array-like
        The probability distribution obtained from the quantum circuit.
    - theta: np.ndarray
        The parameter of the quantum circuit that is being optimized. It consists of a square matrix 
        containing the angles of the gates.
    - add_constraint: bool
        Flag to choose wether to include the contraint term to the loss. If False all the variabbles related
        to it are returned as None-type
    '''

    # Computing the Kulback-Leibler divergence term
    L_kl = kl(output_pdf, observed_pdf) #ATTENTION TO THE ORDER OF THE ARGUMENTS: kl divergence is not symmetric  

    if add_constraint:
        # Computing the constraint term  
        L_cons = np.sum( 1 / ( theta**4 - (np.pi/2)**4 )**2 )

        # Total loss function
        rescale_factor = L_kl / L_cons # AS PAPER: not sure of validity of this point
        total_loss = L_kl + rescale_factor*L_cons

        return L_kl, total_loss, rescale_factor

    else:
        return L_kl, None, None
        


def der_loss_cons(theta, row_index, col_index):
    '''
    Computes the derivative of the constraint term of the loss function with respect to one of the
    angles (elements of the matrix theta).

    Args:
    theta: np.array, the matrix of angles
    row_index: int, the row index of the element of theta with respect to which we want to compute the derivative
    col_index: int, the column index of the element of theta with respect to which we want to compute the derivative 
    '''
    if not isinstance(theta, np.ndarray):
        raise TypeError(f"Argument 'theta' must be a np.array, got {type(theta).__name__}")
    if not isinstance(row_index, (int, np.integer)):
        raise TypeError(f"Argument 'row_index' must be an int, got {type(row_index).__name__}")
    if not isinstance(col_index, (int, np.integer)):
        raise TypeError(f"Argument 'col_index' must be an int, got {type(col_index).__name__}")
    
    if row_index != col_index:
        phi = theta[row_index, col_index] #angle of interest

        return (-8 * (phi**3) ) / (phi**4 + (np.pi/2)**4)**3 #analytical formula
    
    else: #We do not optimize on the diagonal
        return 0.


def der_loss_kl(theta, psi_in, psi_out, p_out, N_out_smooth, p_obs, row_index, col_index, debug):
    '''
    Computes the derivative of the Kullback-Leibler divergence term of the loss function with respect to one of the
    angles (elements of the matrix theta).

    Args:
    sequences: list of binary strings, 
        the sequences of genes observed 
    theta: np.array, 
        the matrix of angles
    psi_in, psi_out: np.array, 
        the input and output state of the quantum circuit, in the basis of the sequences
    p_out: pandas Series, 
        the output pdf, as abs value of elements from psi output. Laplace smoothing should be applied to it.
    N_out_smooth: int,
        the number of trial coming after the Laplace smoothing (see related function)
    p_obs: pandas Series, 
        the observed pdf from experimental data
    row_index: int, 
        the row index of the element of theta with respect to which we want to compute the derivative
    col_index: int, 
        the column index of the element of theta with respect to which we want to compute the derivative
    debug: bool,
        flag to be set for displaying debug messages
    '''
    # Safety checks
    if not isinstance(theta, np.ndarray):
        raise TypeError(f"Argument 'theta' must be a np.array, got {type(theta).__name__}")
    if not isinstance(psi_in, np.ndarray):
        raise TypeError(f"Argument 'theta' must be a np.array, got {type(psi_in).__name__}")
    if not isinstance(psi_out, np.ndarray):
        raise TypeError(f"Argument 'theta' must be a np.array, got {type(psi_out).__name__}")
    if not isinstance(p_out, pd.Series):
        raise TypeError(f"Argument 'p_out' must be a pandas, got {type(p_obs).__name__}")
    if not isinstance(p_obs, pd.Series):
        raise TypeError(f"Argument 'p_obs' must be a pandas, got {type(p_obs).__name__}")
    if not p_out.index.equals(p_obs.index):
        raise ValueError(f"Wrong shape: the shape of the derivative of the output state is {der_psi_out.shape}, while it is expected to be expected {expected_shape}")
    if not isinstance(row_index, (int, np.integer)):
        raise TypeError(f"Argument 'row_index' must be an int, got {type(row_index).__name__}")
    if not isinstance(col_index, (int, np.integer)):
        raise TypeError(f"Argument 'col_index' must be an int, got {type(col_index).__name__}")

    # No update on the diagonal elements
    if row_index == col_index:
        return 0.
    
    # if i is not equal to j.....
    ##################################################################################################
    ### COMPUTE THE OUTPUT STATE FROM THE DERIVATIVE OPERATOR APPLICATION WRT THETA_{i,j}
    der_psi_out = np.dot( der_operator_for_psi_in(theta, row_index=row_index, col_index=col_index), psi_in )    
    if not np.abs(np.linalg.norm(der_psi_out) - 1) < 1e-6:
        der_psi_out = der_psi_out / np.linalg.norm(der_psi_out) # NOT SURE OF THIS PASSAGE
    der_psi_out = big_to_little_endian_vector_state(der_psi_out) # converted to little endian notation, usefule for next operations                      
    
    # Safety checks
    expected_shape = psi_out.shape
    if der_psi_out.shape != expected_shape:
        raise ValueError(f"Wrong shape: the shape of the derivative of the output state is {der_psi_out.shape}, while it is expected to be expected {expected_shape}")
    if not np.all(np.isreal(der_psi_out)):
        raise ValueError("The derivative output state of the quantum circuit is not real. The gradient cannot be computed.")


    # ########################################################################################
    # # SIMMETRIC DERIVATIVE COMPUTATION
    # #NOT SURE TO USE THIS PART, NOT SURE IS CORRECT
    # simm_der_psi_out = np.dot(der_operator_for_psi_in(theta, row_index=col_index, col_index=row_index), psi_in)
    # checkpoint(f"der_psi_out computed (shape: {simm_der_psi_out.shape})", debug=debug)
    
    # simm_der_psi_out = big_to_little_endian_vector_state(simm_der_psi_out)                           
    
    # #CONTROLLA!! Rinormalizzo la derivata di psi_out!
    # simm_der_psi_out = simm_der_psi_out / np.linalg.norm(simm_der_psi_out)

    # if not np.abs(np.linalg.norm(simm_der_psi_out) - 1) < 1e-6:
    #     raise ValueError(f"der_psi_out is not normalized: norm is {np.linalg.norm(simm_der_psi_out)}")


    # expected_shape = psi_out.shape
    # if simm_der_psi_out.shape != expected_shape:
    #     raise ValueError(f"Wrong shape: the shape of the derivative of the output state is {der_psi_out.shape}, while it is expected to be expected {expected_shape}")
    
    # # Check if simm_der_psi_out is real
    # if not np.all(np.isreal(simm_der_psi_out)):
    #     raise ValueError("The derivative output state of the quantum circuit is not real. The gradient cannot be computed.")
    
    # simm_der_psi_out = simm_der_psi_out.real
    
    # # Taking the average derivative between the two simmetric derivatives
    # der_psi_out = (der_psi_out + simm_der_psi_out)/2
    # ########################################################################################

    ###########################################################################################
    # DERIVATIVE KL TERM FORMULA

    #   For the two output states we remove the first entry as the distributions p_out and p_obs
    #   disresgard the first entry, associated with the state |000000>, corresponding to the situation 
    #   with all gene not active (not interesting for our purposes)
    psi_out = psi_out[1:]
    der_psi_out = der_psi_out[1:]
    # Safety checke
    if psi_out.shape[0] != p_out.shape[0] and der_psi_out.shape[0] != p_out.shape[0]:
        raise ValueError("Inconsistent shapes")

    result = 0
    for idx in range(len(p_out)):
        #result += 2 * psi_out[idx+1] * der_psi_out[idx+1] * ( np.log( (p_out.iloc[idx]) / (p_obs.iloc[idx]) ) + 1) * (N_out/N_out_smooth) # Version with (N_out/N_out_smooth) term, not sure of the validity of this
        result += 2 * psi_out[idx] * der_psi_out[idx] * ( np.log( (p_out.iloc[idx]) / (p_obs.iloc[idx]) ) + 1)
    #checkpoint(f"derivative w.r.t. theta_{row_index},{col_index} computed: {result}", debug=debug)

    return result 

def der_loss_total(row_index, col_index, theta, psi_in, psi_out, p_out, N_out_smooth, p_obs, rescale_factor, add_constraint, debug):
    '''
    Computes the derivative of the total loss function with respect to one of the
    angles (elements of the matrix theta).

    Args:
    row_index: int, 
        the row index of the element of theta with respect to which we want to compute the derivative
    col_index: int, 
        the column index of the element of theta with respect to which we want to compute the derivative
    theta: np.array, 
        the matrix of angles
    psi_in, psi_out: np.array, 
        the input and output state of the quantum circuit, in the basis of the sequences
    p_out: np.array, 
        the output pdf, as abs value of elements from psi output. Laplace smoothing should be applied to it.
    N_out_smooth: int,
        the number of trial coming after the Laplace smoothing (see related function)
    p_obs: np.array, 
        the observed pdf from experimental data
    rescale_factor: float, 
        the factor that rescales the constraint term of the loss function
    add_constraint: bool
        Flag to choose wether to include the contraint term to the loss. If False all the variabbles related
        to it are returned as None-type
    debug: bool,
        flag to be set for displaying debug messages
    '''
    # Safety checks
    if not isinstance(theta, np.ndarray):
        raise TypeError(f"Argument 'theta' must be a np.array, got {type(theta).__name__}")
    if not isinstance(psi_in, np.ndarray):
        raise TypeError(f"Argument 'psi_in' must be a np.array, got {type(psi_in).__name__}")
    if not isinstance(p_obs, pd.Series):
        raise TypeError(f"Argument 'p_obs' must be a pandas, got {type(p_obs).__name__}")
    if not isinstance(row_index, (int, np.integer)):
        raise TypeError(f"Argument 'row_index' must be an int, got {type(row_index).__name__}")
    if not isinstance(col_index, (int, np.integer)):
        raise TypeError(f"Argument 'col_index' must be an np.int64, got {type(col_index).__name__}")
    if rescale_factor is not None:
        if not isinstance(rescale_factor, float):
            raise TypeError(f"Argument 'rescale_factor' must be a float, got {type(rescale_factor).__name}")

    # Computing the derivative of the KL term wrt theta_{row_index, col_index}
    der_L_kl = der_loss_kl(theta=theta, psi_in=psi_in, psi_out=psi_out, p_out=p_out, N_out_smooth=N_out_smooth, \
                           p_obs=p_obs, row_index=row_index, col_index=col_index, debug=debug)
    
    if add_constraint:
        der_L_c = der_loss_cons(theta=theta, row_index=row_index, col_index=col_index)

        der_total_loss = der_L_kl + rescale_factor*der_L_c

        return der_total_loss

    else:               
        return der_L_kl
    


def gradient_loss(theta, psi_in, psi_out, p_out, N_out_smooth, p_obs, rescale_factor, add_constraint, debug):

    '''
    This function computes the matrix representing the gradient of the total loss function with respect to the
    elements of the matrix theta.

    Args:
    theta: np.array, 
        the matrix of angles
    psi_in, psi_out: np.array, 
        the input and output state of the quantum circuit, in the basis of the sequences
    p_out: np.array, 
        the output pdf, as abs value of elements from psi output. Laplace smoothing should be applied to it.
    N_out_smooth: int,
        the number of trial coming after the Laplace smoothing (see related function)    
    p_obs: np.array, 
        the observed pdf from experimental data
    rescale_factor: float, 
        the factor that rescales the constraint term of the loss function
    add_constraint: bool
        Flag to choose wether to include the contraint term to the loss. If False all the variabbles related
        to it are returned as None-type
    debug: bool,
        flag to be set for displaying debug message
    '''
    # Safety checks
    if not isinstance(theta, np.ndarray):
        raise TypeError(f"Argument 'theta' must be a np.array, got {type(theta).__name__}")
    if not isinstance(psi_in, np.ndarray):
        raise TypeError(f"Argument 'psi_in' must be a np.array, got {type(psi_in).__name__}")
    if not isinstance(p_obs, pd.Series):
        raise TypeError(f"Argument 'p_obs' must be a pandas, got {type(p_obs).__name__}")
    if rescale_factor is not None:
        if not isinstance(rescale_factor, float):
            raise TypeError(f"Argument 'rescale_factor' must be a float, got {type(rescale_factor).__name__}")
    if theta.ndim != 2 or theta.shape[0] != theta.shape[1]:
        raise ValueError("Theta is not a square matrix.")
    
    # Retrieving the number of genes and so the dimensions of the theta-matrix and the correspondin gradient of the loss
    n_genes = theta.shape[0]
    rows = n_genes
    cols = n_genes

    # Computing the gradient of the loss function
    gradient = generate_matrix(der_loss_total, rows, cols, theta, psi_in, psi_out, p_out, N_out_smooth, p_obs, rescale_factor, add_constraint, debug)

    return gradient

