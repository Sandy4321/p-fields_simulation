import numpy as np 
import matplotlib.pyplot as plt
import ar2gas as gas

def ijk(nx, ny, nz):
    i = [i for i in range(nx)]
    j = [j for j in range(ny)]
    k = [k for k in range(nz)]
    return i, j, k

def pairs_i_dir(i, j, k, lag):
    pairs = []
    for ij in j:
        for ii in i:
            if (ij + lag) in j:
                for ik in k:
                    pair = ([(ii, ij, ik), (ii, (ij + lag), ik)])
                    pairs.append(pair)
    return pairs

def pairs_j_dir(i, j, k, lag):
    pairs = []
    for ii in i:
        for ij in j:
            if (ii + lag) in i:
                for ik in k:
                    pair = ([(ii, ij, ik), ((ii + lag), ij, ik)])
                    pairs.append(pair)
    return pairs

def pairs_k_dir(i, j, k, lag):
    pairs = []
    for ii in i:
        for ij in j:
            for ik in k:
                if (ik + lag) in k:
                    pair = ([(ii, ij, ik), (ii, ij, (ik + lag))])
                    pairs.append(pair)
    return pairs

def ijk_in_n(nx, ny, node):
    i, j, k = node[0], node[1], node[2]
    n = k*nx*ny+j*nx+i
    return n

def variance(pair):
    a, b = pair[0], pair[1]
    return (a-b)**2

def get_values(pairs_list, prop):
    values = []
    for pair in pairs_list:
        values.append([prop[pair[0]], prop[pair[1]]])
    return values

class Variogram_on_Grid:

    def __init__(self, nx, ny, nz, reals, n_lags, step):
        self.nx = nx
        self.ny = ny
        self.nz = nz
        self.reals = reals
        self.n_lags = n_lags
        self.step = step

    def calculate(self):
        i, j, k = ijk(self.nx, self.ny, self.nz)

        variances_horiz_list = []
        variances_vert_list = []

        for lag in range(1, self.n_lags+1, self.step):
            print("Calculating experimtnal variograms for step {}".format(lag))
            pairs_in_i = pairs_i_dir(i, j, k, lag)
            pairs_in_j = pairs_j_dir(i, j, k, lag)
            pairs_in_horiz = pairs_in_i + pairs_in_j
            pairs_in_horiz = [(ijk_in_n(self.nx, self.ny, node[0]), ijk_in_n(self.nx, self.ny, node[1]))  for node in pairs_in_horiz]
            if self.nz > 1:
                pairs_in_k = pairs_k_dir(i, j, k, lag)
                pairs_in_vert = [(ijk_in_n(self.nx, self.ny, node[0]), ijk_in_n(self.nx, self.ny, node[1]))  for node in pairs_in_k]
            
            variances_horiz = []
            variances_vert = []

            for idx, real in enumerate(self.reals):
                #print("Calculating experimental variogram for real {}".format(idx+1))
                values = get_values(pairs_in_horiz, real)
                variance_val = 1/2 * np.nanmean([variance(v) for v in values])
                variances_horiz.append(variance_val)
                if self.nz > 1:
                    values_v = get_values(pairs_in_vert, real)
                    variance_val_v = 1/2 * np.nanmean([variance(v) for v in values_v])
                    variances_vert.append(variance_val_v)
        
            variances_horiz_list.append(variances_horiz)
                
            if self.nz > 1:
            
                variances_vert_list.append(variances_vert)
        
        self.variances_horiz_list = np.array(variances_horiz_list).T
        self.variances_vert_list = np.array(variances_vert_list).T

    def plot(self, horiz_block, vert_block, model):
        x_axis_h = [lag*horiz_block for lag in range(1, self.n_lags+1, self.step)]
        
        if self.nz == 1:
            fig, axes = plt.subplots(1, 1, constrained_layout=True, figsize=(7.5,5))

            for real in self.variances_horiz_list:
                axes.plot(x_axis_h, real, color='grey')
                axes.set_xlabel('Lag distance (m)')
                axes.set_ylabel('Variance')
                axes.set_title('Omni horizontal')
                axes.grid(True)
            
            if model != None:
                cov = gas.compute.KrigingCovariance(1., model)
                sill = cov.compute([0,0,0],[0,0,0])
                model_var_horiz = [sill-cov.compute([0,0,0],[pt,0,0]) for pt in x_axis_h]
                axes.plot(x_axis_h, model_var_horiz, color='red')
            
            plt.show()
        
        else:
            x_axis_v = [lag*vert_block for lag in range(1, self.n_lags+1, self.step)]
            fig, axes = plt.subplots(1, 2, constrained_layout=True, figsize=(15,5))

            for idx in range(len(self.variances_horiz_list)):
                axes[0].plot(x_axis_h, self.variances_horiz_list[idx], color='grey')
                axes[0].set_xlabel('Lag distance (m)')
                axes[0].set_ylabel('Variance')
                axes[0].set_title('Omni horizontal')
                axes[0].grid(True)
                axes[1].plot(x_axis_v, self.variances_vert_list[idx], color='grey')
                axes[1].set_xlabel('Lag distance (m)')
                axes[1].set_ylabel('Variance')
                axes[1].set_title('Vertical')
                axes[1].grid(True)
            
            if model != None:
                cov = gas.compute.KrigingCovariance(1., model)
                sill = cov.compute([0,0,0],[0,0,0])
                model_var_horiz = [sill-cov.compute([0,0,0],[pt,0,0]) for pt in x_axis_h]
                model_var_vert = [sill-cov.compute([0,0,0],[0,0,pt]) for pt in x_axis_v]
                axes[0].plot(x_axis_h, model_var_horiz, color='red')
                axes[1].plot(x_axis_h, model_var_vert, color='red')

            plt.show()