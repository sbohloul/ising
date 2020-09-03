import numpy as np


def generate_states(num_sample, num_site, rand_seed=42):
    """
    Generates random ising configurations (S1,S2,S3,..,Sn) where Sj = -1 or +1.

    Parameters
    ----------
    num_sample : integer
        Number of random configurations.
    num_site : integer
        Number of ising sites.
    rand_seed: integer
        Random number generator seed.

    Returns
    -------
    2D array : integer 
        Matrix of random ising configurations of size (nsample, nsite). The ising
        spin of site j for sample k is given by Sj(k) = [k, j] element.
      
    Examples
    --------
    Five random configurations for three spin sites:
    >>> generate_states(2, 3)
        array([[-1,  1, -1],
               [-1, -1,  1]])
    """ 
    
    np.random.seed(rand_seed)

    return np.random.choice([-1, 1], size=(num_sample,num_site))


def interaction_matrix(state, interaction='pairwise'):
    """
    Interaction matrix for a given random configurations of ising states.  The size 
    of random configuration matrix is (nsample, nsite).

    Parameters
    ----------
    state : 2D array
        Random configurations of ising states.
    interaction : string
        Type of interaction between ising sites.

    Returns
    -------
    interaction matrix : 3D array  
        A 3D array of size (nsite, nsite, nsample) so interation of pairs (i, j) for 
        sample k is given by Iij(k) = [i, j, k] element.
      
    Examples
    --------
    """

    # nsample = state.shape[0]
    nsite = state.shape[1]
    state = np.swapaxes(state, 0, 1)

    return state.reshape(nsite, 1, -1) * state.reshape(1, nsite, -1)


def total_energy(state, J=1):
    """
    Calculates total energy for a given ising configuration matrix of 
    size (nsample, nsite) according to H = -J*\Sum_j S_j*S_j+1.

    Parameters
    ----------
    state : 2D array
        Random configurations of ising states.
    J : integer
        Interaction strengh

    Returns
    -------
    total energy : vector   
        A vector of with nsample elements.
      
    Examples
    --------
    """    

    ncol = state.shape[1]   # number of sites
    H = -J * state * state[:, np.arange(1, ncol+1)%ncol]
        
    return np.sum(H, axis=1)


# generate random states and calculate their total energies
nsite = 3    # number of ising sites     
nsample = 5  # number of samples
ising_state = generate_states(nsample, nsite, rand_seed=42)
ising_intmat = interaction_matrix(ising_state)
energy = total_energy(ising_state)    

# Prepare data for training


