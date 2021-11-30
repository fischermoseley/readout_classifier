#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

@author: % P. M. Harrington
"""

import datetime
import numpy as np
import qutip
#from helpers import files
import helpers_qutip

"""
class Metadata:
    def __init__(self, folder_name=None):
        # create results folder
        self.path_results_parent = files.make_folder(folder='results')
        
        self.time_start = datetime.datetime.now()

        if folder_name is None:
            folder_name = datetime.datetime.now().strftime("%Y%m%d_%H%M%S") + '_results'
            
        self.path_results = files.make_folder(folder=folder_name, 
                                                path_root=self.path_results_parent)
        self.path_figs = self.path_results
        
        print(self.path_figs)
"""

class Parameters:
    def __init__(self,
                 num_sites=1,
                 num_fock=2,
                 times=None, 
                 time_dep=False,
                 ntraj=4,
                 nsubsteps=41,
                 random_seed=None):
        #
        self.num_sites = num_sites
        
        # create list of num_fock states per site
        if type(num_fock) is not list:
            self.num_fock = self.num_sites*[num_fock]
        else:
            self.num_fock = []
            for n_fock in num_fock:
                if n_fock < 2:
                    self.num_fock.append(None)
                else:
                    self.num_fock.append(n_fock)

        #
        self.unit = 2*np.pi
        self.times = times

        self.time_dep = time_dep
        
        # trajectories
        self.ntraj = ntraj
        self.nsubsteps = nsubsteps
        
        #
        self.random_seed = random_seed #3
        
    def print_parameters(self):
        print('\n')
        print('solver: {}'.format(self.solver))
        print('num_pts: {}'.format(len(self.times)))
        print('\n')
        
    #def save_parameters(self, path_save):
    #    x = files.format_dict_as_csv(self.__dict__)
    #    files.write_csv(x, path_root=path_save, file_name='parameters')
        
class Operators_qubit_only:
    def __init__(self, n_sites, excitations=None, swap_tensor_order=False):
        '''
            n_sites (int), the # of sites. Each site has a qubit and oscillator.
        '''
        # define basic qutip operators for constructing tensored operator
        if excitations is not None:
            _excitations = excitations
        else:
            _excitations = n_sites**2
            
        self.Sm = qutip.enr_destroy(n_sites*[2], _excitations)
        self.iden = qutip.enr_identity(n_sites*[2], _excitations)
        self.zero = 0.*qutip.enr_identity(n_sites*[2], _excitations)
        
        # dims = self.Sm[0].dims[0]

        # self.zero = qutip.qzero(dims)
        # self.iden = qutip.qeye(dims)

        # create other operators
        self.Sp = []
        self.Sz = []
        self.Sx = []
        self.Sy = []
        for n in range(n_sites):
            _Sm = self.Sm[n]

            self.Sp.append(_Sm.dag())
            self.Sz.append(_Sm*_Sm.dag() - _Sm.dag()*_Sm)
            self.Sx.append((_Sm + _Sm.dag())/2)
            self.Sy.append(-1j*(_Sm - _Sm.dag())/2)

        # define operators for other observables

        # Jordan-Wigner transformation    
        phase = n_sites*[self.Sz[0]]
        for n in range(n_sites):
            for k in range(n):
                phase[n] = phase[n]*(-self.Sz[k])
            
        self.f = []
        for n in range(n_sites):
            self.f.append(phase[n]*self.Sm[n])

class Operators:
    def __init__(self, parameters, swap_tensor_order=False):
        '''
            n_sites (int), the # of sites. Each site has a qubit and oscillator.
            n_fock_list (list), List of # of Fock states for each site, ie. [0, 2, 2, 1].
        '''
        # number of lattice sites
        n_sites = parameters.num_sites
        n_fock = parameters.num_fock        

        # define basic qutip operators for constructing tensored operator
        s_zero = qutip.qzero(2)
        s_iden = qutip.qeye(2)
        s_destroy = qutip.destroy(2)

        # create operators for each site according to n_fock for that site
        zero = []
        iden = []
        a = []
        Sm = []
        for _n_fock in n_fock:
            # define single site operators operators
            if _n_fock is not None:
                
                a_zero = qutip.qzero(_n_fock)
                a_iden = qutip.qeye(_n_fock)
                a_destroy = qutip.destroy(_n_fock)
                
                if swap_tensor_order:
                    zero.append(qutip.tensor(a_zero, s_zero))
                    iden.append(qutip.tensor(a_iden, s_iden))
                    a.append(qutip.tensor(a_destroy, s_iden))
                    Sm.append(qutip.tensor(a_iden, s_destroy))
                else:
                    zero.append(qutip.tensor(s_zero, a_zero))
                    iden.append(qutip.tensor(s_iden, a_iden))
                    a.append(qutip.tensor(s_iden, a_destroy))
                    Sm.append(qutip.tensor(s_destroy, a_iden))
            else:
                zero.append(s_zero)
                iden.append(s_iden)
                a.append(s_iden)
                Sm.append(s_destroy)

        #
        self.zero = []
        self.iden = []
        self.Sm = []
        self.a = []
        
        # create zero operator
        for n in range(n_sites):
            self.zero.append(qutip.tensor(zero))
            
        #
        for n in range(n_sites):
            # establish an operator list of identity
            op_list = []
            for m in range(n_sites):
                op_list.append(iden[m])

            # modify the identity operator list and assign operator
            op_list[n] = iden[n]
            self.iden.append(qutip.tensor(op_list))
            
            op_list[n] = Sm[n]
            self.Sm.append(qutip.tensor(op_list))
            
            op_list[n] = a[n]
            self.a.append(qutip.tensor(op_list))
            
        # create other operators
        self.Sp = []
        self.Sz = []
        self.Sx = []
        self.Sy = []
        self.x = []
        self.y = []
        self.n = []
        for n in range(n_sites):
            _Sm = self.Sm[n]
            _a = self.a[n]

            self.Sp.append(_Sm.dag())
            self.Sz.append(_Sm*_Sm.dag() - _Sm.dag()*_Sm)
            self.Sx.append(_Sm + _Sm.dag())
            self.Sy.append(-1j*(_Sm - _Sm.dag()))
            
            # use identity if this site does not have fock states
            if n_fock[n] is not None:
                self.x.append((_a + _a.dag())/np.sqrt(2))
                self.y.append(-1j*(_a - _a.dag())/np.sqrt(2))
                self.n.append(_a.dag()*_a)
            else:
                self.x.append(self.iden[n])
                self.y.append(self.iden[n])
                self.n.append(self.iden[n])
            
        # define operators for other observables

        # Jordan-Wigner transformation    
        phase = n_sites*[self.Sz[0]]
        for n in range(n_sites):
            for k in range(n):
                phase[n] = phase[n]*(-self.Sz[k])
            
        self.f = []
        for n in range(n_sites):
            self.f.append(phase[n]*self.Sm[n])


        # # resonator Fock state projection operators
        # self.Pi = []
        # for n in range(n_sites):
        #     Pi = []
        #     for k in range(n_fock):
        #         proj = qutip.tensor(qutip.basis(n_fock, k)*qutip.basis(n_fock, k).dag(), qutip.qeye(2))
                
        #         op_list = []
        #         for _ in range(n_sites):
        #             op_list.append(iden)
    
        #         op_list[n] = proj
        #         Pi.append(qutip.tensor(op_list))
                
        #     self.Pi.append(Pi)

def solve_time_evolution(p, hamiltonian, state_0, c_ops=[], h_parameters=None, progress_bar=True):
    times = p.times
    
    if (p.solver=="sesolve"):
        result_solve = qutip.sesolve(H=hamiltonian,
                                    psi0=state_0, 
                                    tlist=times, 
                                    progress_bar=progress_bar)
    
    elif (p.solver=="mesolve"):
        result_solve = qutip.mesolve(H=hamiltonian,
                                     rho0=state_0, 
                                     tlist=times, 
                                     c_ops=c_ops, 
                                     progress_bar=progress_bar)
        
    return result_solve

def solve_time_evolution_trajectories(p, hamiltonian, state_0, c_ops=[], sc_ops=[], ntraj=5, nsubsteps=1, progress_bar=True):
    ''' performs stochastic master equation solver '''

    #
    times = p.times
    
    #
    dt = (p.times[1] - p.times[0])/nsubsteps
    np.random.seed(p.random_seed)
    
    # random every step
    r = np.random.normal(size=(ntraj, len(times), nsubsteps, len(sc_ops)))
    noise = np.sqrt(dt)*r

    # stochastic master equation    
    # use milstein for 1d noise array, the default taylor1.5 gives a 2d noise array
    """
    result_solve = qutip.smesolve(hamiltonian,
                            state_0,
                            times,
                            c_ops=c_ops,
                            sc_ops=sc_ops, 
                            ntraj=ntraj,
                            nsubsteps=nsubsteps,
                            method='homodyne',
                            solver='milstein', # use milstein for 1d noise array, the default taylor1.5 gives a 2d noise array
                            noise=noise,
                            store_measurement=True) """

    result_solve = qutip.smesolve(hamiltonian,
                        state_0,
                        times,
                        c_ops=c_ops,
                        sc_ops=sc_ops, 
                        ntraj=ntraj,
                        nsubsteps=nsubsteps,
                        method='homodyne',
                        solver='milstein', # use milstein for 1d noise array, the default taylor1.5 gives a 2d noise array
                        store_measurement=True)

    # noise : int, array[int, 1d], array[double, 4d]
    # int : seed of the noise
    # array[int, 1d], length = ntraj, seeds for each trajectories
    # array[double, 4d] (ntraj, len(times), nsubsteps, len(sc_ops)*[1|2])
    #     vector for the noise, the len of the last dimensions is doubled for
    #     solvers of order 1.5. The correspond to results.noise
    
    # state_t = result_solve.states
    return result_solve
