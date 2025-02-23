import numpy as np

from debugger_module import *

from gates import *
from utils import *


#####################################################################################################
### GENERAL FUNCTIONS

def build_circuit_operator(layers):
    '''
    Use this function to build the operator that provides the evolution of the state associated
    to a generic quantum circuit, given the layers that compose it, in matrix representation.
    layers: list of numpy arrays, the matrices that characterize the  layers (e.g. encoder and correlation layers)

    returns: 
    numpy array, the operator that characterizes the evolution
    induced to the state by the quantum circuit
    '''
    quantum_circuit_operator = matrix_multiplication(layers)

    return quantum_circuit_operator

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
    checkpoint(f"Number of qubits: {n_qubits}", debug)
    # Filling a list with the matrix representation of the operators acting on each qubit
    operators = []
    for ii in range(n_qubits): #FOLLOWING BIG ENDIAN ORDER (NOT AS QISKIT)
        Ry = rotation_y(theta[ii, ii])
        checkpoint(f"Ry shape: {Ry.shape}\nRy type: {type(Ry)}", debug)
        operators.append(Ry)
    checkpoint(f"Operators:\ntype {type(operators)}", debug)
    
    # Computing the tensor product of the operators
    encoder_operator = tensor_product(operators)
    checkpoint(f"Encoder operator shape: {encoder_operator.shape}", debug)

    # Safety check
    expected_shape = (2**n_qubits, 2**n_qubits)
    if encoder_operator.shape != expected_shape:
        raise ValueError(f"Wrong shape: the shape is {encoder_operator.shape}, while it is expected to be expected {expected_shape}")
    
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
    
    n_qubits = theta.shape[0] #one qubit associated to each gene

    if not isinstance(control, int):
        raise TypeError(f"Argument 'control' must be an int, got {type(control).__name__}")
    if control < 0 or control >= n_qubits:
        raise ValueError(f"Argument 'control' must be an integer in the range [0, {n_qubits}), got {control}")

    control_rotation_gates = []
    for jj in range(n_qubits):  #for each qubit
        if jj != control: # only if the target qubit is different from the control qubit
            phi = theta[control, jj] #rotation angle

            operator = control_r_y(phi, n_qubits, control = control, target = jj)
            control_rotation_gates.append(operator)

            if operator.shape != (2**n_qubits, 2**n_qubits):
                raise ValueError(f"Wrong shape: the shape is {operator.shape}, while it is expected to be expected {(2**n_qubits, 2**n_qubits)}")

    correlation_lk_operator = matrix_multiplication(control_rotation_gates)
    if correlation_lk_operator.shape != (2**n_qubits, 2**n_qubits):
        raise ValueError(f"Wrong shape: the shape is {correlation_lk_operator.shape}, while it is expected to be expected {(2**n_qubits, 2**n_qubits)}")

    return correlation_lk_operator 

def quantum_circuit_GRN(theta):
    '''
    questa funzione costruisce l'operator associato al circuito quantistico 
    del paper
    '''
    #Safety checks
    if not isinstance(theta, np.ndarray):
        raise TypeError(f"Argument 'theta' must be a numpy array, got {type(theta).__name__}")
    if len(theta.shape) != 2:
        raise ValueError(f"Argument 'theta' must be a 2D numpy array, got {len(theta.shape)}D")
    if theta.shape[0] != theta.shape[1]:
        raise ValueError(f"Argument 'theta' must be a square matrix, got {theta.shape[0]}x{theta.shape[1]}")
    
    n_qubits = theta.shape[0] #one qubit associated to each gene
    
    layers = []
    layers.append(encoder_layer(theta))
    for kk in range(n_qubits):
        layers.append(correlation_layer_m(theta, kk))

    return build_circuit_operator(layers)


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
    elif control != row_index:
        return np.zeros((2**n_qubits, 2**n_qubits))
    else:
        return der_control_r_y(theta[row_index,col_index], n_qubits, control = row_index, target = col_index)



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
         return np.zeros((2**n_qubits, 2**n_qubits))
    
    operator = []
    for mm in range(n_qubits): #per ogni addendo
        if mm != row_index:
            addend = np.zeros((2**n_qubits, 2**n_qubits))
        else:
            layers = [] 
            layers.append(encoder_layer(theta))
            
            for kk in range(n_qubits): #costruisco il prodotto definisce l'addendo
                if kk == mm:
                    layers.append(der_correlation_layer_m(theta, control=kk, row_index=row_index, col_index=col_index))
                else:
                    layers.append(correlation_layer_m(theta, control=kk))
            addend = matrix_multiplication(layers)
        
        operator.append(addend)
        
    operator = sum(operator)

    return operator
