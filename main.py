#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

@author: % P. M. Harrington
"""

import matplotlib.pyplot as plt
import qutip
import defs
import helpers_qutip

if __name__=="__main__":
    #m = helpers_qutip.Metadata('example_trajectory')
    
    #print('start: {}'.format(m.time_start))

    p = defs.get_parameters(num_sites=1, num_fock=[5])
    p.print_parameters()
    
    ops = helpers_qutip.Operators(p, swap_tensor_order=True)

    h_args = defs.get_hamiltonian(p, ops)
    hamiltonian = h_args['H0']
        
    c_ops = defs.get_collapse_operators(p, ops)   

    # time-evolution
    state_0 = defs.get_initial_state(p, ops, state_qb='+x')

    # trajectories
    result_sme = helpers_qutip.solve_time_evolution_trajectories(p,
                                                                 hamiltonian,
                                                                 state_0, 
                                                                 c_ops=[],#[c_ops[1]],
                                                                 sc_ops=[c_ops[0]],
                                                                 ntraj=p.ntraj,
                                                                 nsubsteps=p.nsubsteps)

    # nbar
    expect_op = ops.n[0]
    fig, ax = plt.subplots(figsize=(6,4))
    for _rho in result_sme.states:
        expect_traj = qutip.expect(expect_op, _rho)
        ax.plot(p.times, expect_traj)
    ax.set_xlabel('Time ($\mu$s)')
    ax.set_ylabel('$<n>$')
    
    # x
    expect_op = ops.x[0]
    fig, ax = plt.subplots(figsize=(6,4))
    for _rho in result_sme.states:
        expect_traj = qutip.expect(expect_op, _rho)
        ax.plot(p.times, expect_traj)
    ax.set_xlabel('Time ($\mu$s)')
    ax.set_ylabel('$<a + a^\dagger>$')
    
    # y
    expect_op = ops.y[0]
    fig, ax = plt.subplots(figsize=(6,4))
    for _rho in result_sme.states:
        expect_traj = qutip.expect(expect_op, _rho)
        ax.plot(p.times, expect_traj)
    ax.set_xlabel('Time ($\mu$s)')
    ax.set_ylabel('$<j(a - a^\dagger)>$')
    
    # Sz
    expect_op = ops.Sz[0]
    fig, ax = plt.subplots(figsize=(6,4))
    for _rho in result_sme.states:
        expect_traj = qutip.expect(expect_op, _rho)
        ax.plot(p.times, expect_traj)
    ax.set_ylim([-1, 1])
    ax.set_xlabel('Time ($\mu$s)')
    ax.set_ylabel('$<\sigma_z>$')

    plt.show()