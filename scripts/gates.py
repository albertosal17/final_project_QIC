import numpy as np

from debugger_module import *

from utils import tensor_product


############################################################################
### ROTATION Y GATES

def rotation_y(phi):
    '''
    Matrix representation in the computational basis of the rotation operator around y-axis of the 
    Bloch sphere.

    Args:
    phi: float, the rotation angle
    '''
    # Safety check
    if not isinstance(phi, float):
        raise TypeError(f"Argument 'phi' must be a float, got {type(phi).__name__}")
        
    return np.array([[np.cos(phi/2), -np.sin(phi/2)],  \
                     [np.sin(phi/2),  np.cos(phi/2)]], \
                    dtype=complex)


def der_rotation_y(phi):
    '''
    Derivative of the matrix representing the rotation around the y-axis of the Bloch sphere.
    It is a matrix obtained by deriving each element of the original matrix.

    Args:
    phi: float, rotation angle
    '''
    # Safety check
    if not isinstance(phi, float):
        raise TypeError(f"Argument 'phi' must be a float, got {type(phi).__name__}")
    
    return np.array([[-np.sin(phi/2)/2, -np.cos(phi/2)/2], \
                     [ np.cos(phi/2)/2, -np.sin(phi/2)/2]], \
                     dtype=complex)

########################################################################
### CONTROLLED R_y GATES

def control_r_y(phi, n_qubits, control, target):
    '''
    Matrix representation in the computational basis of the control R-y gate between qubit, for a 
    quantum circuit with many qubits.
    It applyies the rotation around y on a 'target' qubit only if the 'control' qubit is in state |1>.
    The overall operator is given by the tensor product of identities (with dimension 2) and the 
    control R-y gate between the two qubits of interest

    c-Ry = 1 x .. x 1 x |0><0| x 1 x .. x 1 x 1 x 1 x .. x 1 +
            1 x .. x 1 x |1><1| x 1 x .. x 1 x R_y(phi) x 1 x .. x 1 
        
        where 1 represents the identity matrix with dimension 2 and x the tensor product operator
        and |0><0| and |1><1| are the projectors on the computational basis states of the control qubit.
    

    Args:
    phi: float, rotation angle
    n_qubits: int, total number of qubits in the quantum circuit
    control: int, control qubit index
    target: int, target qubit index
    '''
    # Safety checks
    if not isinstance(control, int):
        raise TypeError(f"Argument 'control' must be an int, got {type(control).__name__}")
    if not isinstance(target, int):
        raise TypeError(f"Argument 'target' must be an int, got {type(target).__name__}")
    if not isinstance(phi, float):
        raise TypeError(f"Argument 'phi' must be a float, got {type(phi).__name__}")
    if not isinstance(n_qubits, int):
        raise TypeError(f"Argument 'n_qubits' must be an int, got {type(n_qubits).__name__}")
    if control > n_qubits-1 or target > n_qubits-1 or control < 0 or target < 0:
        error("Control or target qubit index out of range.")

    # Creating the operators that will be tensor producted to create the c-Ry operator
    I = np.eye(2, dtype=complex) #identity operator
    Ry = rotation_y(phi) # Rotation operator
    control_0_matrix = np.array([[1, 0],  \
                                 [0, 0]], \
                                 dtype=complex) #|0><0| projector
    control_1_matrix = np.array([[0, 0], \
                                 [0, 1]],\
                                 dtype=complex) #|1><1| projector

    # Creating two lists that will contain the operators composing the two addends in the c-Ry formula
    # At the end I will take the tensor product of these operators and sum the two results.
    first_term_operators = [] 
    second_term_operators = [] 
    # FIlling the lists 
    for i in range(n_qubits): #for each qubit
        if i == control:
            first_term_operators.append(control_0_matrix)
            second_term_operators.append(control_1_matrix)
        elif i == target:
            first_term_operators.append(I)
            second_term_operators.append(Ry)
        else:
            first_term_operators.append(I)
            second_term_operators.append(I)
    
    # Taking the tensor product of the operators
    first_term = tensor_product(first_term_operators)   
    second_term = tensor_product(second_term_operators)

    return first_term + second_term
    

# Funzione che ti da l'operatore derivato c-R_y
def der_control_r_y(phi, n_qubits, control, target):
    '''
    Derivative of the matrix representing the controlled-R_y gate, for a quantum circuit with many qubits.
    It is a matrix obtained by deriving each element of the original matrix.
        
    Args:
    phi: float, rotation angle
    n_qubits: int, number of qubits
    control: int, control qubit index
    target: int, target qubit index

    Nota: 
    Il codice è simile a quello della funzione per creare l'operatore c-R_y.
    L'unica differenza è che nel calcolo della derivata l'unico termine che ha derivata non nulla è
    quello associato allo stato |1> del qubit di controllo, in cui al posto di R_y sul
    target qubit facciamo agire la derivata di R_y. 
    '''
    # Safety checks
    if not isinstance(control, int):
        raise TypeError(f"Argument 'control' must be an int, got {type(control).__name__}")
    if not isinstance(target, int):
        raise TypeError(f"Argument 'target' must be an int, got {type(target).__name__}")
    if not isinstance(phi, float):
        raise TypeError(f"Argument 'phi' must be a float, got {type(phi).__name__}")
    if not isinstance(n_qubits, int):
        raise TypeError(f"Argument 'n_qubits' must be an int, got {type(n_qubits).__name__}")
    if control > n_qubits-1 or target > n_qubits-1 or control < 0 or target < 0:
        error("Control or target qubit index out of range.")

    # Creating the operators that will be tensor producted to create the c-Ry operator
    I = np.eye(2, dtype=complex)
    der_Ry = der_rotation_y(phi)
    control_1_matrix = np.array([[0, 0],  \
                                 [0, 1]], \
                                 dtype=complex)   

    operators = [] #control qubit 1
    for i in range(n_qubits):
        if i == control:
            operators.append(control_1_matrix)
        elif i == target:
            operators.append(der_Ry)
        else:
            operators.append(I)
    derivative_operator = tensor_product(operators)

    # Safety check
    expected_shape = (2**n_qubits, 2**n_qubits)
    if derivative_operator.shape != expected_shape:
        raise ValueError(f"Wrong shape: the shape is {derivative_operator.shape}, while it is expected to be expected {expected_shape}")
    
    return derivative_operator
