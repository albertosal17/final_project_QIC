from debugger_module import *

from loss import gradient_loss

def new_theta(theta, learn_rate, sequences, psi_in, p_obs, rescale_factor):
    
    gradient = gradient_loss(sequences, theta, psi_in, p_obs, rescale_factor)

    #CONTROLLA CHE LA DIAGONALE DI GRADIENT SIA ZERO

    return theta + ( learn_rate * (gradient+gradient.T) ) / 2