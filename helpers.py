import ar2gas as gas
import pygeostat as gs
import math

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