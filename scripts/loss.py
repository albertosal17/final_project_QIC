import numpy as np

from debugger_module import *

from gates import *
from utils import *
from quantum_circuit import *    


def kl(p, q):
    """Kullback-Leibler divergence D(P || Q) for discrete distributions
    Parameters
    ----------
    p, q : array-like, dtype=float, shape=n
        Discrete probability distributions.
    """
    p = np.asarray(p, dtype=float)
    q = np.asarray(q, dtype=float)

    return np.sum(np.where(p != 0, p * np.log(p / q), 0)) #numpy.where(condition, [x, y]) Return elements, either from x or y, depending on condition.

def loss_function(observed_pdf, output_pdf, theta, rescale_factor):
    '''
    Loss function for the GRN estimation problem.
    It consists of the Kullback-Leibler divergence between the observed (from data) and 
    output (from quantum circuit) pdfs.
    Additionally, it includes a constraint that penalizes the output pdf for values of theta
    that are close to pi/2. This is done to avoid the output pdf to be too peaked around pi/2, 
    a critical value as discussed in the paper.
    A dynamical rescaling factor is included to balance the two terms of the loss function.

    Parameters:
    - observed_pdf: array-like
        The probability distribution obtained from the data.
    - output_pdf: array-like
        The probability distribution obtained from the quantum circuit.
    - theta: np.ndarray
        The parameter of the quantum circuit that is being optimized. It consists of a square matrix 
        containing the angles of the gates.
    - rescale_factor: float
        The factor that rescales the constraint term of the loss function. 
        It is used to make L_cons of a value comparable to L_kl.
    '''

    # Computing the Kulback-Leibler divergence term
    L_kl = kl(output_pdf, observed_pdf) #ATTENTION TO THE ORDER OF THE ARGUMENTS: kl divergence is not symmetric  

    # Computing the constraint term  
    L_cons = np.sum( 1 / ( theta**4 - (np.pi/2)**4 )**2 )

    # Total loss function
    total_loss = L_kl + rescale_factor*L_cons

    return total_loss


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
    
    phi = theta[row_index, col_index] #angle of interest

    return -8*(phi**3) / (phi**4 + (np.pi/2)**4)**3


def der_loss_kl(sequences, theta, psi_in, psi_out, p_out, p_obs, row_index, col_index):
    '''
    Computes the derivative of the Kullback-Leibler divergence term of the loss function with respect to one of the
    angles (elements of the matrix theta).

    VEDI QUADERNO PER LA FORMULA ESPLICITA

    Args:
    sequences: list of binary strings, the sequences of genes observed 
    theta: np.array, the matrix of angles
    psi_in: np.array, the input state of the quantum circuit, in the basis of the sequences
    p_obs: np.array, the observed pdf from experimental data
    row_index: int, the row index of the element of theta with respect to which we want to compute the derivative
    col_index: int, the column index of the element of theta with respect to which we want to compute the derivative
    '''
    # Safety checks
    if not isinstance(sequences, pd.Index):
        raise TypeError(f"Argument 'sequences' must be a pandas Index, got {type(sequences).__name__}")
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
    if not isinstance(row_index, (int, np.integer)):
        raise TypeError(f"Argument 'row_index' must be an int, got {type(row_index).__name__}")
    if not isinstance(col_index, (int, np.integer)):
        raise TypeError(f"Argument 'col_index' must be an int, got {type(col_index).__name__}")
    
    checkpoint(f"Computing for theta_{row_index},{col_index}")
    # Computing the derivative of the output state, with respect to the specific angle of interest
    # For this, we need to apply the derivative of the quantum circuit operator to the input state
    # The derivative of the quantum circuit operator is built using a function contained in module quantum_circuit.py
    der_psi_out = np.dot(der_operator_for_psi_in(theta, row_index=row_index, col_index=col_index), psi_in)
    checkpoint(f"der_psi_out computed (shape: {der_psi_out.shape})", debug=True)
    print(der_psi_out)
    #                              CONTROLLA SIA CORRETTO 
    der_psi_out = der_psi_out[1:] #discarding the first element, associated with the state |000000>
    expected_shape = (len(sequences),)
    if der_psi_out.shape != expected_shape:
        raise ValueError(f"Wrong shape: the shape of thederivative of the output state is {der_psi_out.shape}, while it is expected to be expected {expected_shape}")
    # Check if der_psi_out is real
    if not np.all(np.isreal(der_psi_out)):
        raise ValueError("The derivative output state of the quantum circuit is not real. The gradient cannot be computed.")
    # Check if psi_out is real
    if not np.all(np.isreal(psi_out)):
        raise ValueError("The output state of the quantum circuit is not real. The gradient cannot be computed.")
    der_psi_out, psi_out = der_psi_out.real, psi_out.real

    # Applying the formula for the derivative of the KL divergence term
    result = 0
    for idx in range(len(sequences)):
        result += 2 * psi_out[idx] * der_psi_out[idx] * (np.log(p_out.iloc[idx] / p_obs.iloc[idx]) + 1)
    checkpoint(f"derivative w.r.t. theta_{row_index},{col_index} computed: {result}")

    return result 

def der_loss_total(row_index, col_index, sequences, theta, psi_in, psi_out, p_out, p_obs, rescale_factor):
    '''
    Computes the derivative of the total loss function with respect to one of the
    angles (elements of the matrix theta).

    Args:
    row_index: int, the row index of the element of theta with respect to which we want to compute the derivative
    col_index: int, the column index of the element of theta with respect to which we want to compute the derivative
    sequences: list of binary strings, the sequences of genes observed
    theta: np.array, the matrix of angles
    psi_in: np.array, the input state of the quantum circuit, in the basis of the sequences
    p_obs: np.array, the observed pdf from experimental data
    rescale_factor: float, the factor that rescales the constraint term of the loss function
    '''
    # Safety checks
    if not isinstance(sequences, pd.Index):
        raise TypeError(f"Argument 'sequences' must be a pandas Index, got {type(sequences).__name__}")
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
    if not isinstance(rescale_factor, float):
        raise TypeError(f"Argument 'rescale_factor' must be a float, got {type(rescale_factor).__name}")
                                                                          
    der_L_c = der_loss_cons(theta, row_index, col_index)
    der_L_kl = der_loss_kl(sequences, theta, psi_in, psi_out, p_out, p_obs, row_index, col_index)

    der_total_loss = der_L_c + rescale_factor*der_L_kl
    if not isinstance(der_total_loss, float):
        raise TypeError(f"'der_total_loss' must be a float, got {type(der_total_loss).__name}")
                
    return der_total_loss


def gradient_loss(sequences, theta, psi_in, psi_out, p_out, p_obs, rescale_factor):
    '''
    This function computes the matrix representing the gradient of the total loss function with respect to the
    elements of the matrix theta.

    Args:
    sequences: list of binary strings, the sequences of genes observed
    theta: np.array, the matrix of angles
    psi_in: np.array, the input state of the quantum circuit, in the basis of the sequences
    p_obs: np.array, the observed pdf from experimental data
    rescale_factor: float, the factor that rescales the constraint term of the loss function

    '''
    # Safety checks
    if not isinstance(sequences, pd.Index):
        raise TypeError(f"Argument 'sequences' must be a pandas Index, got {type(sequences).__name__}")
    if not isinstance(theta, np.ndarray):
        raise TypeError(f"Argument 'theta' must be a np.array, got {type(theta).__name__}")
    if not isinstance(psi_in, np.ndarray):
        raise TypeError(f"Argument 'psi_in' must be a np.array, got {type(psi_in).__name__}")
    if not isinstance(p_obs, pd.Series):
        raise TypeError(f"Argument 'p_obs' must be a pandas, got {type(p_obs).__name__}")
    if not isinstance(rescale_factor, float):
        raise TypeError(f"Argument 'rescale_factor' must be a float, got {type(rescale_factor).__name__}")

    if theta.ndim != 2 or theta.shape[0] != theta.shape[1]:
        raise ValueError("Theta is not a square matrix.")
    n_genes = theta.shape[0]
    rows = n_genes
    cols = n_genes

    gradient = generate_matrix(der_loss_total, rows, cols, sequences, theta, psi_in, psi_out, p_out, p_obs, rescale_factor)

    return gradient

