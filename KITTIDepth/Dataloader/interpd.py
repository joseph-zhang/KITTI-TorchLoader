import numpy as np
from scipy.interpolate import LinearNDInterpolator

def interpdepth(sparse_depth):
    indi, indj = np.where(sparse_depth > 0.)
    ij = list(zip(indi, indj))
    d = sparse_depth[sparse_depth > 0.]
    m, n = sparse_depth.shape
    f = LinearNDInterpolator(ij, d, fill_value=0)
    J, I = np.meshgrid(np.arange(n), np.arange(m))
    IJ = np.vstack([I.flatten(), J.flatten()]).T
    return f(IJ).reshape([m,n])
