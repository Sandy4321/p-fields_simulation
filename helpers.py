import ar2gas as gas
import pygeostat as gs
import math
import numpy as np
from scipy.spatial import KDTree

def gslibvar_to_ar2gasvar(var_str):
    var_str = str(var_str)
    cov = []
    splitted = var_str.splitlines()
    nugget = float(splitted[0].split()[1])
    cov.append(gas.compute.Covariance.nugget(nugget))
    n_struct = int(splitted[0].split()[0])
    for struc in range(n_struct):
        t = int(splitted[struc*2+1].split()[0])
        contribution = float(splitted[struc*2+1].split()[1])
        a1, a2, a3 = float(splitted[struc*2+1].split()[2]), float(splitted[struc*2+1].split()[3]), float(splitted[struc*2+1].split()[4])
        r1, r2, r3 = float(splitted[struc*2+2].split()[0]), float(splitted[struc*2+2].split()[1]), float(splitted[struc*2+2].split()[2])
        if t == 1:
            cov.append(gas.compute.Covariance.spherical(contribution, r1, r2, r3, a1, a2, a3))
        if t == 2:
            cov.append(gas.compute.Covariance.exponential(contribution, r1, r2, r3, a1, a2, a3))
        if t ==3:
            cov.append(gas.compute.Covariance.gaussian(contribution, r1, r2, r3, a1, a2, a3))
    return cov

def autogrid(x, y, z, sx, sy, sz, buffer=0):
    if z is None:
        nz = 1
        oz = 0
        max_z = 0
    else:
        oz = min(z) - buffer #+ sz/2
        max_z = max(z) + buffer + sz/2
        nz = math.ceil((max_z - oz)/sz)

    ox = min(x) - buffer #+ sx/2
    oy = min(y) - buffer #+ sy/2
    max_x = max(x) + buffer + sx/2
    max_y = max(y) + buffer + sy/2
    nx = math.ceil((max_x - ox)/(sx))
    ny = math.ceil((max_y - oy)/(sy))
    
    return gas.data.CartesianGrid(nx, ny, nz, sx, sy, sz, ox, oy, oz), gs.GridDef([nx, ox, sx, ny, oy, sy, nz, oz, sz])

def cat_random_sample(prob_list, u):

    position = 0
    probs = []
    for idx, prob in enumerate(prob_list):
        probs.append(prob)
        acc = sum(probs)
        if u <= acc:
            position = idx
            break

    return position

def cat_sampler(probs_matrix, codes, reals):
    probs_matrix = np.array(probs_matrix).T
    realizations = []
    for real_idx, r in enumerate(reals):
        realization = []
        for idx, b in enumerate(np.array(r).T):
            position = cat_random_sample(probs_matrix[idx], b)
            realization.append(int(codes[position]))
        realizations.append(realization)
    return realizations

def standardize(probs_matrix):
    probs_matrix = np.array(probs_matrix)
    probs_matrix = np.where(probs_matrix > 1, 1, probs_matrix)
    probs_matrix = np.where(probs_matrix < 0, 0, probs_matrix)
    probs_sum = sum(probs_matrix)
    std_probs = [i / probs_sum for i in probs_matrix]
    return std_probs

def reals_to_indicators(cat_reals):
    ind_reals = {}
    codes = np.unique(cat_reals[0])
    for c in codes:
        ind_c = []
        for real in cat_reals:
            real_ind = np.where( (real == c) == True, 1, 0)
            ind_c.append(real_ind)
        ind_reals['ind_{}'.format(c)] = ind_c
    return ind_reals

def samples_dist(x, y, z):
    if z == None:
        z = np.zeros(len(x))
    coords = []
    min_dist = []
    for i, j, k in zip(x, y, z):
        coords.append((i, j, k))

    kdtree = KDTree(coords)
    for p in coords:
        dist, neighs = kdtree.query(p, k=2)
        min_dist.append(dist[1])
    gs.histplt(min_dist, icdf=True, title='Distance to the nearest sample')

def u_coef(prob_list):
    prob_list_t = np.array(prob_list).T
    max_prob = [np.max(i) for i in prob_list_t]
    pmax, pmin = np.max(max_prob), np.min(max_prob)
    u = [(pmax-i)/(pmax-pmin) for i in max_prob]
    return np.array(u)