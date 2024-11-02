import numpy as np
# original matlab code found here: http://www-users.med.cornell.edu/~jdvicto/spkdm.html
# translated to python in 2024. 


def VPdis(tli, tlj, cost):
    """
    Calculate the "spike time" distance (Victor & Purpura 1996) for a single cost.

    Parameters:
    tli : list or numpy array
        Vector of spike times for the first spike train.
    tlj : list or numpy array
        Vector of spike times for the second spike train.
    cost : float
        Cost per unit time to move a spike.

    Returns:
    float
        The spike time distance between the two spike trains.
    """
    nspi = len(tli)
    nspj = len(tlj)

    if cost == 0:
        return abs(nspi - nspj)
    elif cost == float('inf'):
        return nspi + nspj

    # Initialize the scoring matrix
    scr = np.zeros((nspi + 1, nspj + 1))

    # Initialize the margins with the cost of adding a spike
    scr[:, 0] = np.arange(nspi + 1)
    scr[0, :] = np.arange(nspj + 1)

    if nspi and nspj:
        for i in range(1, nspi + 1):
            for j in range(1, nspj + 1):
                scr[i, j] = min(scr[i - 1, j] + 1, 
                                scr[i, j - 1] + 1, 
                                scr[i - 1, j - 1] + cost * abs(tli[i - 1] - tlj[j - 1]))

    return scr[nspi, nspj]

