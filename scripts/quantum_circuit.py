import numpy as np

from debugger_module import *

from gates import *
from utils import *



###################################################################################################
### LAYERS OF THE QUANTUM CIRCUIT

def encoder_layer(theta, debug = False):
    '''
    Computes the operator that perform the operation defined in the encoding layer of the quantum circuit:
    It applyes a rotation around the y-axis of the Bloch sphere on each qubit of the quantum circuit.    
    
    Args:
    theta: numpy array, matrix with rotation angles (see paper)

    returns: 
    numpy array, the operator that characterizes the evolution
    induced to the state by the encoder layer
    '''
    #Safety checks
    if not isinstance(theta, np.ndarray):
        raise TypeError(f"Argument 'theta' must be a numpy array, got {type(theta).__name__}")
    if len(theta.shape) != 2:
        raise ValueError(f"Argument 'theta' must be a 2D numpy array, got {len(theta.shape)}D")
    if theta.shape[0] != theta.shape[1]:
        raise ValueError(f"Argument 'theta' must be a square matrix, got {theta.shape[0]}x{theta.shape[1]}")
    
    n_qubits = theta.shape[0] #one qubit associated to each gene

    # Filling a list with the matrix representation of the operators acting on each qubit
    operators = []
    for ii in range(n_qubits): #FOLLOWING BIG ENDIAN ORDER (NOT AS QISKIT)
        Ry = rotation_y(theta[ii, ii])
        operators.append(Ry)
    
    # Computing the tensor product of the operators
    encoder_operator = tensor_product(operators)

    # Safety check
    expected_shape = (2**n_qubits, 2**n_qubits)
    if encoder_operator.shape != expected_shape:
        raise ValueError(f"Wrong shape for the encoder layer matrix: the shape is {encoder_operator.shape}, while it is expected to be {expected_shape}")
    
    return encoder_operator


def correlation_layer_m(theta, control):
    '''
    Computes the operator performing the operation defined in the correlation layer of the quantum circuit:
    It applyies a controlled rotation around the y-axis of the Bloch sphere on each qubit of the quantum circuit.
    It uses the same qubit as control qubit

    Args:
    theta: numpy array, matrix with rotation angles (see paper)
    control: int, the index of the qubit that is used as control qubit
    '''
    #Safety checks
    if not isinstance(theta, np.ndarray):
        raise TypeError(f"Argument 'theta' must be a numpy array, got {type(theta).__name__}")
    if len(theta.shape) != 2:
        raise ValueError(f"Argument 'theta' must be a 2D numpy array, got {len(theta.shape)}D")
    if theta.shape[0] != theta.shape[1]:
        raise ValueError(f"Argument 'theta' must be a square matrix, got {theta.shape[0]}x{theta.shape[1]}")
    if not isinstance(control, int):
        raise TypeError(f"Argument 'control' must be an int, got {type(control).__name__}")

    # Retrieving the number of qubits composing the circuit
    n_qubits = theta.shape[0] #one qubit associated to each gene
    # Safety check
    if control < 0 or control >= n_qubits:
        raise ValueError(f"Argument 'control' must be an integer in the range [0, {n_qubits}), got {control}")   

    # Filling a list with the matrix representation of the operators 
    # e.g. m=3 --> [c-Ry(theta_m,0), c-Ry(theta_m,1), c-Ry(theta_m,2), c-Ry(theta_m,4),  c-Ry(theta_m,5)]
    control_rotation_gates = []
    for jj in range(n_qubits):  #for each qubit
        if jj != control: # only if the target qubit is different from the control qubit
            phi = theta[control, jj] #rotation angle

            operator = control_r_y(phi, n_qubits, control = control, target = jj)
            control_rotation_gates.append(operator)
    # Multiplying each operator from the list to get the full action of the L_m layer
    correlation_lk_operator = matrix_multiplication(control_rotation_gates)

    # Safety check
    if correlation_lk_operator.shape != (2**n_qubits, 2**n_qubits):
        raise ValueError(f"Wrong shape: the shape is {correlation_lk_operator.shape}, while it is expected to be expected {(2**n_qubits, 2**n_qubits)}")

    return correlation_lk_operator 


def quantum_circuit_GRN(theta):
    '''
    This function returns the matrix representation of the operator associated with the quantum circuit
    used to simulate the Gene Regulatory Network from the paper.
    
    Args:
    theta: numpy array, matrix with rotation angles (see paper)
    '''
    #Safety checks
    if not isinstance(theta, np.ndarray):
        raise TypeError(f"Argument 'theta' must be a numpy array, got {type(theta).__name__}")
    if len(theta.shape) != 2:
        raise ValueError(f"Argument 'theta' must be a 2D numpy array, got {len(theta.shape)}D")
    if theta.shape[0] != theta.shape[1]:
        raise ValueError(f"Argument 'theta' must be a square matrix, got {theta.shape[0]}x{theta.shape[1]}")
    
    n_qubits = theta.shape[0] #one qubit associated to each gene

    # Filling a list with the matrix representation of the operators associated to each layer.
    # The order of filling follows the order of application: first L_enc, then L0, then L1, ...
    layers = []
    layers.append(encoder_layer(theta))
    for kk in range(n_qubits):
        layers.append(correlation_layer_m(theta, kk))

    qc_operator = matrix_multiplication(layers)

    return qc_operator



def initial_ground_state(nr_qubits: int):
    '''
    This function computes the initial state of a multi-qubit circuit, with all qubits initialized to state |0>
    in the computational basis.

    Args: 
    nr_qubit: int
        the number of qubits composing the circuit

    Note: 
    -little endian and big endian representations for this state have the same representation
    '''
    psi_q0 = np.array([1,0], dtype=complex) #state |0>
    psi_in = psi_q0
    for ii in range(nr_qubits-1): # tensorizing the state |0> for each qubit
        psi_in = np.kron(psi_in, [1,0])

    return psi_in


def output_state(theta: np.ndarray, psi_in: np.ndarray, debug: bool):
    """
    Computes the output quantum state of a circuit given an input state and parameterized angles.

    Args:
    theta : np.ndarray
        A square matrix containing rotation angles (in radians) that define the quantum circuit.
    psi_in : np.ndarray
        The input quantum state vector.
    debug : bool
        If True, prints debug messages at key computation steps.

    Returns:
    np.ndarray
        The output quantum state vector after applying the circuit.

    Notes:
    - The output state is reordered to little-endian convention.

    Raises:
    ValueError
        If the output state is not normalized or contains complex values.
    """

    # Computing the matrix associated with the quantum circuit 
    qc_opt = quantum_circuit_GRN(theta)
    checkpoint(f"Quantum circuit operator built (shape operator {qc_opt.shape})", debug = debug)

    # Computing output state from the circuit
    psi_out = np.dot(qc_opt, psi_in) #big-endian ordered (see test 2)
    checkpoint(f"Output state computed (shape {psi_out.shape})", debug = debug)
    
    psi_out = big_to_little_endian_vector_state(psi_out)
    #checkpoint(f"Output state reordered as little endian", debug = debug)
    
    # Safety check: psi_out msut be normalized and real (otherwise the formula to compute the gradiend used next is not valid)
    if not np.abs(np.linalg.norm(psi_out) - 1) < 1e-6:
        raise ValueError(f"psi_out is not normalized: norm is {np.linalg.norm(psi_out)}")
    if not np.all(np.isreal(psi_out)):
        raise ValueError("The output state of the quantum circuit is not real.")

    return psi_out


def output_probability_distribution(psi_out: np.ndarray, nr_cells: int, debug: bool):
    '''
    Computes the probability distribution from the output quantum state.

    Args:
    psi_out : np.ndarray
        The output state vector of the quantum circuit.
    nr_cells : int
        The number of cells from the experimental data. They are used 'fictitiously' as the number of measurements 
        from which the distribution considered in Laplace smoothing has been obtained.
    debug : bool
        If True, prints debug messages at key computation steps.

    Returns:
    -------
    Tuple[pd.Series, float]
        - `p_out_smooth`: A pandas Series representing the smoothed probability distribution, 
          indexed by binary strings corresponding to quantum states.
        - `N_out_smooth`: The renormalization factor after Laplace smoothing.
    '''

    # Computing the probability distribution associated to the output state
    p_out = np.abs(psi_out)**2
    #checkpoint(f"Output probabilities computed (shape {p_out.shape})", debug = debug)

    # Associating to the probabilities the binary strings representing the state of the system
    # I do this considering that the vector is now represented in little-endian representation 
    # So if the vector is [c_0,c_1,c_2, .. , c_N]
    # c_0 represents state |00....0> 
    # c_1 represents state |00...01> 
    # c_2 represents state |00..010> 
    # c_N represents state |11....1>
    # Function bynary labels covert integers to binary strings so with that you are okay 
    p_out = pd.Series(p_out, index=binary_labels(p_out.shape[0], big_endian=False))  
    #checkpoint(f"Labels associated with the distribution, following little-endian ordering", debug = debug)

    # removing entry associated with state in which each qubit is |0>: it represent the situation
    # in which each gene is inactivated, and it is not interesting for this study
    p_out.pop('000000') 
    #checkpoint(f"Popped out the non-relevant entries associated to 000000 label, now the entries are {p_out.shape[0]}", debug = debug)
    
    # Renormazlizing
    p_out = p_out / p_out.sum()
    #checkpoint(f"P_out renormalized: sum is {p_out.sum()}", debug = debug)

    # Laplace smoothing on p_out
    p_out_smooth, N_out_smooth = Laplace_smoothing(distribution=p_out, N_trials=nr_cells)
    checkpoint(f"Smoothing of p_out computed (check normalization: {p_out_smooth.sum()})", debug = debug)


    return p_out_smooth, N_out_smooth



###################################################################################################
### DERIVATIVES OF THE LAYERS 

def der_correlation_layer_m(theta, control, row_index, col_index):
    '''
    theta: numpy array, matrix with rotation angles 
    control: int, the index of the qubit that is used as control qubit
    row_index: int, row-index of the paramameter that is the variable with respect to which we want to compute the derivative
    col_index: int, column-index of the paramameter that is the variable with respect to which we want to compute the derivative

    returns:
    numpy array, the operator that characterizes the evolution
    induced to the state by the derivative of the correlation layer
    '''
    #Safety checks
    if not isinstance(theta, np.ndarray):
        raise TypeError(f"Argument 'theta' must be a numpy array, got {type(theta).__name__}")
    if len(theta.shape) != 2:
        raise ValueError(f"Argument 'theta' must be a 2D numpy array, got {len(theta.shape)}D")
    if theta.shape[0] != theta.shape[1]:
        raise ValueError(f"Argument 'theta' must be a square matrix, got {theta.shape[0]}x{theta.shape[1]}")
    
    n_qubits = theta.shape[0] #one qubit associated to each gene
    
    #VEDI QUADERNO PER LA FORMULA ESPLICITA CON TUTTI I CONTI
    if row_index == col_index:
        return np.zeros((2**n_qubits, 2**n_qubits)) # No update on the diagonal angles
    elif control != row_index: # Null derivative for angles whose control qubit is not the one considered here as argument
        return np.zeros((2**n_qubits, 2**n_qubits))
    else:
        return der_control_r_y(phi=theta[row_index,col_index], n_qubits=n_qubits, control = row_index, target = col_index)



def der_operator_for_psi_in(theta, row_index, col_index):
    '''
    Funzione che ti da la derivata di l'operatore da applicare a |psi_in> per ottenere la derivata
    di |psi_out> (rispetto un parametro theta_a1,a2)

    theta: numpy array, matrix with rotation angles (see paper)
    row_index: int, row-index of the paramameter that is the variable with respect to which we want to compute the derivative
    col_index: int, column-index of the paramameter that is the variable with respect to which we want to compute the derivative
    '''
    #Safety checks
    if not isinstance(theta, np.ndarray):
        raise TypeError(f"Argument 'theta' must be a numpy array, got {type(theta).__name__}")
    if len(theta.shape) != 2:
        raise ValueError(f"Argument 'theta' must be a 2D numpy array, got {len(theta.shape)}D")
    if theta.shape[0] != theta.shape[1]:
        raise ValueError(f"Argument 'theta' must be a square matrix, got {theta.shape[0]}x{theta.shape[1]}")
    
    n_qubits = theta.shape[0] #one qubit associated to each gene

    if row_index == col_index:
        return np.zeros((2**n_qubits, 2**n_qubits)) # No update on the diagonal angles
    
    else:
        layers = [] 
        layers.append(encoder_layer(theta)) # it is always the first applied, and we do not derive it as it depends only on diagonal values
        
        for kk in range(n_qubits): #posizione l'operatore derivato nella giusta posizione
            if kk == row_index:
                layers.append( der_correlation_layer_m(theta=theta, control=kk, row_index=row_index, col_index=col_index) )
            else:
                layers.append( correlation_layer_m(theta=theta, control=kk) )
        
        # computing resultant operator from the matrix multiplication of the operators 
        # e.g. L5⋅L4⋅L3⋅L2⋅L1⋅(dL0/dθ_{0,j})⋅Lenc OR  L5⋅L4⋅L3⋅L2⋅(dL1/dθ_{1,j})⋅L1⋅Lenc + ..
        operator_derived = matrix_multiplication(layers)

    return operator_derived
