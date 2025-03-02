import numpy as np
import pandas as pd

from debugger_module import *

############################################################################
### UTILS

def tensor_product(objects):
    '''
    Computes the tensor product of a list of numpy arrays.
    Args:
    objects: list of numpy arrays, the objects to be tensor producted (e.g. vectors or matrices)

    If objects = [A, B, C], the function returns the tensor product A x B x C
    memo: tensor product is commutative
    '''
    # Safety checks
    if not isinstance(objects, list):
        raise TypeError(f"Argument 'objects' must be a list, got {type(objects).__name__}")
    for obj in objects:
        if not isinstance(obj, np.ndarray):
            raise TypeError(f"The elements inside the 'objects' argument must be an np.ndarray, got {type(obj).__name__}")
    
    # Computing the tensor product, one object at a time
    result = objects[0] 
    for i in range(1, len(objects)):
        result = np.kron(result, objects[i])

    return result


def matrix_multiplication(objects):
    '''
    Computes the matrix product of a list of numpy arrays (matrices). Typically these matrices
    represent the gates of a quantum circuit. It is useful when you want to apply transformations
    in sequence on the whole circuit.

    Args:
    objects: list of numpy arrays, the objects to be tensor producted (e.g. vectors or matrices)

    If objects = [A, B, C], the function returns a matrix representation of the operator C⋅B⋅A 
    NOTE: the order of the matrices in the list is important: the first (index 0) should be the 
    one that is applied first, the last (index -1) should be the one that is applied last
    memo: matrix multiplication is non commutative in general
    '''
    # Safety checks
    if not isinstance(objects, list):
        raise TypeError(f"Argument 'objects' must be a list, got {type(objects).__name__}")
    for obj in objects:
        if not isinstance(obj, np.ndarray):
            raise TypeError(f"The elements inside the 'objects' argument must be an np.ndarray, got {type(obj).__name__}")
    
    # Computing the matrix product, one object at a time, from first to last
    result = objects[0]
    if len(objects)>1:
        for i in range(1, len(objects)):
            result = np.dot(objects[i], result)

    return result

def Laplace_smoothing(distribution, N_trials, filename=None, alpha=1.):
    '''
    Laplace smoothing for a discrete probability distribution.
    The idea is to add a small amount of probability to all classes, so that no class has 
    zero probability. This is done by adding a constant to the numerator and a multiple of 
    the number of classes to the denominator.
    Parameters:
    - distribution: pd series
        The original probability distribution
    - N_trials: int
        The total number of counts.
        In the case of GRN estimation, this is the number of cells considered
    - filename: str, optional
        The name of the file where the smoothed pdf values will be saved. If None the data are not saved.
    - alpha: float, optional
        The smoothing parameter. Default is 1 as it is the choice made by the paper's authors.

    Returns:
    - pdf_smoothed: array-like, shape=(N_classes,)
        The smoothed probability distribution.
    
    '''
    # Safety checks
    if not isinstance(distribution, pd.Series):
        raise TypeError(f"Argument 'distribution' must be a pandas Series, got {type(distribution).__name__}")
    if not isinstance(N_trials, int):
        raise TypeError(f"Argument 'N_trials' must be a int, got {type(N_trials).__name__}")

    if not isinstance(alpha, float):
        raise TypeError(f"Argument 'alpha' must be a float, got {type(alpha).__name__}")

    N_classes = distribution.shape[0]

    #Laplace smoothing formula
    frequencies = distribution*N_trials 
    pdf_smoothed = (frequencies + alpha) / (N_trials + N_classes*alpha)
    new_N_trials = N_trials + N_classes*alpha

    # Ensure it's a pandas Series and not dataframe
    if isinstance(pdf_smoothed, pd.DataFrame):  # If it's a single column DataFrame, extract the coulumn as a series
        checkpoint("Laplace smoothing of the pdf resulted in a dataframe. Squeezing to a series.")
        pdf_smoothed = pdf_smoothed.squeeze()
        
    #Save the smoothed pdf values to a file if requested
    if filename:
        pdf_smoothed.to_csv(f'../results/{filename}.csv', sep='\t', header=False)
        checkpoint(f"Smoothed pdf values saved to {filename}.") 

    return pdf_smoothed, new_N_trials


def binary_reshuffling_indeces(sequences_probability):
    '''
    This function reorder a set of binary sequences and their corresponding probabilities. 
    The sequences are reordered by associating them with decimal numbers
    and then oredering these numbers in ascending order.

    Parameters: 
    - sequences_probability: pandas Series
        The probability distribution associated to a set of binary sequences.
        The values of the series are the probabilities and the index are the binary sequences.
        The binary sequences are strings of 0s and 1s.
    '''
    # Safety check
    if not isinstance(sequences_probability, pd.Series):
        raise TypeError(f"Argument 'sequences_probability' must be a pandas Series, got {type(sequences_probability).__name__}")
                                                                                              
    # Converting the binary sequences to integers 
    int_list = [int(b, 2) for b in sequences_probability.index] 
    
    # Sorting the integers and corresponding values
    sorted_indices = sorted(zip(int_list, sequences_probability.values)) # Sort the integers and keep the sorted values
    sorted_values = [value for _, value in sorted_indices] 

    # COnverting back the integers to binary sequences
    sorted_binary_list = [bin(i)[2:].zfill(6) for i, _ in sorted_indices]
    new_sequences_probability = pd.Series(sorted_values, index=sorted_binary_list) # New series to plot

    return new_sequences_probability


def generate_matrix(func, rows: int, cols: int, *args) -> np.ndarray:
    '''
    This function generates a matrix by applying a function to each element of the matrix.
    The function func must take as input the row and column indices of the element and the additional arguments.
    
    Args:
    func: function, the function to be applied to each element of the matrix
    rows: int, the number of rows of the matrix
    cols: int, the number of columns of the matrix
    '''

    return np.fromfunction(np.vectorize(lambda i, j: func(int(i), int(j), *args)), (rows, cols), dtype=float)


def binary_labels(size: int, big_endian = True) -> list[str]:
    '''
    This function associates to the entries of a vector state representing a multi-qubit system 
    the binary strings that represents the states of the single qubits

    The ordering can be big endian or little endian, depending on the application (Qiskit by default
    used little endian ordering)

    By default it uses big endian ordering, meaning that we interpret the left-most bit (MSB) in the 
    string as the state of the first qubit (qubit 0), the bit next to it is the state of the 
    second qubit, and so on.. This is the representation that we get from the way we have built the 
    matrix operator associated with the quantum circuit.

    Example:

    ---------
    For a 3-qubit system (dim = 2³ = 8), the function returns:

    binary_labels(8, big_endian=True) -> ['000', '100', '010', '110', '001', '101', '011', '111']

    This means:
    - The first entry corresponds to |000⟩
    - The second entry corresponds to |100⟩
    - The third entry corresponds to |010⟩
    - The fourth entry corresponds to |110⟩
    - The fifth entry corresponds to |001⟩
    - The sixth entry corresponds to |101⟩
    - The seventh entry corresponds to |011⟩
    - The eighth entry corresponds to |111⟩

    If using big-endian ordering (as in Qiskit), the states would instead be indexed as:

    ['000', '001', '010', '011', '100', '101', '110', '111']

    This means:
    - The first entry corresponds to |000⟩
    - The second entry corresponds to |001⟩
    - The third entry corresponds to |010⟩
    - The fourth entry corresponds to |011⟩
    - The fifth entry corresponds to |100⟩
    - The sixth entry corresponds to |101⟩
    - The seventh entry corresponds to |110⟩
    - The eighth entry corresponds to |111⟩
    '''
    # Calculate the bit length for the largest number (size-1)
    bit_length = (size - 1).bit_length()
    
    # Generate binary labels
    if big_endian:
        # Big-endian ordering
        return [format(i, f'0{bit_length}b')[::-1] for i in range(size)]
    else:
        # Little-endian ordering (Qiskit style)
        return [format(i, f'0{bit_length}b') for i in range(size)]
    


def big_to_little_endian_vector_state(vector: np.array) -> np.array:
    '''
    Reorders a vector representing a multi-qubit state from big-endian to little-endian by 
    adjusting the positions of the vector elements corresponding to qubit states.

    Args:
        vector: List of vector entries (amplitudes) in big-endian order.

    Returns:
        List of vector entries in little-endian order.
    '''
    # Get the number of qubits from the size of the vector
    n_qubits = int(np.log2(vector.shape[0]))
    
    # Reordering the vector's indices
    little_endian_vector = np.array([vector[int(bin(i)[2:].zfill(n_qubits)[::-1], 2)] for i in range(vector.shape[0])])
    
    return little_endian_vector