import numpy as np 
import matplotlib.pyplot as plt
import ar2gas as gas
import time

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
    prop = np.array(prop)
    values = []
    for pair in pairs_list:
        values.append([prop[pair[0]], prop[pair[1]]])
    return values
    
def reals_to_indicators(cat_reals, codes):
    ind_reals = {}
    for c in codes:
        ind_c = []
        for real in cat_reals:
            real_ind = np.where( (np.array(real) == c) == True, 1, 0)
            ind_c.append(real_ind)
        ind_reals[c] = ind_c
    return ind_reals

class Variogram:

    def __init__(self, nx, ny, nz, reals, codes, n_lags, step, exhaust=None):
        self.nx = nx
        self.ny = ny
        self.nz = nz
        self.reals = reals
        self.codes = codes
        self.ind_reals = reals_to_indicators(self.reals, self.codes)
        self.variances_codes = {}
        self.n_lags = n_lags
        self.step = step
        self.lag_lst = [i*step for i in range(1, self.n_lags+1)]
        self.exhaust = exhaust

    def calculate(self):
        i, j, k = ijk(self.nx, self.ny, self.nz)

        horiz_pairs_lst = []
        vert_pairs_lst = []
        
        t1 = time.time()
        for lag in self.lag_lst:
            print('Getting pairs for step {}'.format(lag))
            pairs_in_i = pairs_i_dir(i, j, k, lag)
            pairs_in_j = pairs_j_dir(i, j, k, lag)
            pairs_in_horiz = pairs_in_i + pairs_in_j
            pairs_in_horiz = [(ijk_in_n(self.nx, self.ny, node[0]), ijk_in_n(self.nx, self.ny, node[1]))  for node in pairs_in_horiz]
            
            horiz_pairs_lst.append(pairs_in_horiz)
            
            if self.nz > 1:
                pairs_in_k = pairs_k_dir(i, j, k, lag)
                pairs_in_vert = [(ijk_in_n(self.nx, self.ny, node[0]), ijk_in_n(self.nx, self.ny, node[1]))  for node in pairs_in_k]
                
                vert_pairs_lst.append(pairs_in_vert)
        t2 = time.time()
        t = t2 - t1
        print('took {} seconds'.format(round(t, 2)))
        print('\n')
        
        t1 = time.time()
        for c in self.codes:
            print('Getting variances for code {}'.format(c))
        
            self.variances_codes[c] = {}
            self.variances_codes[c]['variances_horiz_list'] = []
            self.variances_codes[c]['variances_vert_list'] = []
        
            if self.exhaust is not None:
                print('Getting variances for exhaustive')
                exhaust_ind = np.where( (np.array(self.exhaust) == c) == True, 1, 0)
                values = [get_values(p, exhaust_ind) for p in horiz_pairs_lst]
                variance_val = [1/2 * np.nanmean([variance(v) for v in vals]) for vals in values]
                self.variances_codes[c]['variances_horiz_exhaust'] = variance_val
                if self.nz > 1:
                    values = [get_values(p, exhaust_ind) for p in vert_pairs_lst]
                    variance_val = [1/2 * np.nanmean([variance(v) for v in vals]) for vals in values]
                    self.variances_codes[c]['variances_vert_exhaust'] = variance_val
        
            for idx, real in enumerate(self.ind_reals[c]):
                print('Getting variances for realization {}'.format(idx))
                values = [get_values(p, real) for p in horiz_pairs_lst]
                variance_val = [1/2 * np.nanmean([variance(v) for v in vals]) for vals in values] 
                self.variances_codes[c]['variances_horiz_list'].append(variance_val)
                if self.nz > 1:
                    values = [get_values(p, real) for p in vert_pairs_lst]
                    variance_val = [1/2 * np.nanmean([variance(v) for v in vals]) for vals in values]
                    self.variances_codes[c]['variances_vert_list'].append(variance_val)
            print('\n')
        t2 = time.time()
        t = t2 - t1
        print('took {} seconds \n'.format(round(t, 2)))
        
    def plot(self, horiz_block, vert_block, models):
     
        x_axis_h = [lag*horiz_block for lag in self.lag_lst]
        x_axis_v = [lag*vert_block for lag in self.lag_lst] if self.nz > 1 else None

        for idxc, c in enumerate(self.codes):

            if self.nz == 1:
                fig, axes = plt.subplots(1, 1, constrained_layout=True, figsize=(7.5,5))

                for real in self.variances_codes[c]['variances_horiz_list']:
                    axes.plot(x_axis_h, real, color='grey')
                    axes.set_xlabel('Lag distance (m)')
                    axes.set_ylabel('Variance')
                    axes.set_title('Omni horizontal cat {}'.format(c))
                    axes.grid(True)
                
                if models != None:
                    cov = gas.compute.KrigingCovariance(1., models[idxc])
                    sill = cov.compute([0,0,0],[0,0,0])
                    model_var_horiz = [sill-cov.compute([0,0,0],[pt,0,0]) for pt in x_axis_h]
                    axes.plot(x_axis_h, model_var_horiz, color='red')

                if self.exhaust is not None:
                    axes.plot(x_axis_h, self.variances_codes[c]['variances_horiz_exhaust'], color='blue')
                
                plt.show()

            else:
                
                fig, axes = plt.subplots(1, 2, constrained_layout=True, figsize=(15,5))

                for idx in range(len(self.variances_codes[c]['variances_horiz_list'])):
                    axes[0].plot(x_axis_h, self.variances_codes[c]['variances_horiz_list'][idx], color='grey')
                    axes[0].set_xlabel('Lag distance (m)')
                    axes[0].set_ylabel('Variance')
                    axes[0].set_title('Omni horizontal cat {}'.format(c))
                    axes[0].grid(True)
                    axes[1].plot(x_axis_v, self.variances_codes[c]['variances_vert_list'][idx], color='grey')
                    axes[1].set_xlabel('Lag distance (m)')
                    axes[1].set_ylabel('Variance')
                    axes[1].set_title('Vertical cat {}'.format(c))
                    axes[1].grid(True)
                
                if models is not None:
                    cov = gas.compute.KrigingCovariance(1., models[idxc])
                    sill = cov.compute([0,0,0],[0,0,0])
                    model_var_horiz = [sill-cov.compute([0,0,0],[pt,0,0]) for pt in x_axis_h]
                    model_var_vert = [sill-cov.compute([0,0,0],[0,0,pt]) for pt in x_axis_v]
                    axes[0].plot(x_axis_h, model_var_horiz, color='red')
                    axes[1].plot(x_axis_v, model_var_vert, color='red')

                if self.exhaust is not None:
                    axes[0].plot(x_axis_h, self.variances_codes[c]['variances_horiz_exhaust'], color='blue')
                    axes[1].plot(x_axis_v, self.variances_codes[c]['variances_vert_exhaust'], color='blue')

                plt.show()