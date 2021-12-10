# P. M. Harrington, F. Moseley

import numpy as np
import qutip
import helpers_qutip

def get_parameters(num_sites=1, num_fock=2):
    p = helpers_qutip.Parameters(num_sites=num_sites,
                                 num_fock=num_fock,
                                 times=np.linspace(0, 10, 501),
                                 time_dep=True,
                                 ntraj=50,
                                 nsubsteps=41,
                                 random_seed=10)

    p.chi = 1*p.unit
    p.kappa = 2*p.unit
    p.Delta_1 = 0.
    p.epsilon_1 = 0.35*p.unit

    p.solver = 'mesolve'
    return p

def get_collapse_operators(p, ops):
    c_ops = []

    for k in range(p.num_sites):
        c_ops.append(np.sqrt(p.kappa)*ops.a[k])
        c_ops.append(np.sqrt(p.kappa/5)*ops.Sm[k])

    return c_ops

def get_initial_state(p, state_qb):
    '''
    Define the initial state of the system.
    '''

    if state_qb=='+x':
        psi_q = (qutip.basis(2,0) + qutip.basis(2,1))/np.sqrt(2)
    elif state_qb=='-x':
        psi_q = (qutip.basis(2,0) - qutip.basis(2,1))/np.sqrt(2)
    elif state_qb=='+y':
        psi_q = (qutip.basis(2,0) + 1j*qutip.basis(2,1))/np.sqrt(2)
    elif state_qb=='-y':
        psi_q = (qutip.basis(2,0) - 1j*qutip.basis(2,1))/np.sqrt(2)
    elif state_qb=='+z':
        psi_q = qutip.basis(2,0)
    elif state_qb=='-z':
        psi_q = qutip.basis(2,1)

    # the zt states represent \tilde{z}, which are a slight pertubations
    # to z eigenstates that's used to get around the fixed point at
    # z = +1, -1. only works with density matricies
    elif state_qb=='+z_tilde':
        ket_gnd = qutip.basis(2, 0)
        ket_exc = qutip.basis(2, 1)

        psi_c = qutip.basis(p.num_fock, 0)
        # ground state
        op_ = []
        for site_num in range(p.num_sites):
            op_.append(qutip.tensor(psi_c, ket_gnd))
        state_gnd = qutip.ket2dm(qutip.tensor(op_))

        # excited state
        op_ = []
        for site_num in range(p.num_sites):
            op_.append(qutip.tensor(psi_c, ket_exc))
        state_exc = qutip.ket2dm(qutip.tensor(op_))

        delta = 0.01
        return (1-delta)*state_gnd + delta*state_exc

    elif state_qb=='-z_tilde':
        ket_gnd = qutip.basis(2, 0)
        ket_exc = qutip.basis(2, 1)

        psi_c = qutip.basis(p.num_fock, 0)
        # ground state
        op_ = []
        for site_num in range(p.num_sites):
            op_.append(qutip.tensor(psi_c, ket_gnd))
        state_gnd = qutip.ket2dm(qutip.tensor(op_))

        # excited state
        op_ = []
        for site_num in range(p.num_sites):
            op_.append(qutip.tensor(psi_c, ket_exc))
        state_exc = qutip.ket2dm(qutip.tensor(op_))

        delta = 0.01
        return (1-delta)*state_exc + delta*state_gnd

    elif state_qb=='arb':
        theta = np.pi/8
        c0 = np.cos(theta)
        c1 = np.sin(theta)
        psi_q = c0*qutip.basis(2,0) + c1*qutip.basis(2,1)

    else:
        raise RuntimeError('Qubit state not defined.')


    psi_c = qutip.basis(p.num_fock, 0)
    #psi_c = coherent(p.num_fock, 2)

    # vacuum state
    op_ = []
    for site_num in range(p.num_sites):
        op_.append(qutip.tensor(psi_c, psi_q))

    ket = qutip.tensor(op_)
    if (p.solver=='sesolve'): # the stochastic (monte carlo) solver acts on kets not density matricies
        return ket
    elif (p.solver=='mesolve'): # the SME solver operates on density matricies so return that instead
        return qutip.ket2dm(ket)

def get_hamiltonian(p, ops):
    '''
    Define the Hamiltonian.
    '''

    # define returns
    hamiltonian = []

    # operators
    iden = ops.iden
    Sz = ops.Sz
    a = ops.a

    # Hamiltonian

    # # resonator
    # h_a = 0
    # for k in range(p.num_sites):
    #     h_a += p.omega_a*a[k].dag()*a[k]

    # # qubit
    # h_q = 0
    # for k in range(p.num_sites):
    #     h_q += p.omega_q*(iden[k] - Sz[k])/2

    # local resonator-qubit interaction
    h_disp = 0
    for k in range(p.num_sites):
        h_disp += -p.chi*a[k].dag()*a[k]*Sz[k]

    # # resonator-resonator interaction
    # h_J = 0
    # for k in range(p.num_sites-1):
    #     h_J += J*(a[k]*a[k+1].dag() + a[k].dag()*a[k+1])

    # # drive qubit
    # h_drive_q_Delta = 0
    # h_drive_q_Omega = 0
    # for k in range(p.num_sites):
    #     h_drive_q_Delta += Delta_q*(iden[k] - Sz[k])/2
    #     h_drive_q_Omega += (Omega/2)*Sx[k]

    h_drive_Delta_1 = 0
    h_drive_epsilon_1 = 0
    for k in range(p.num_sites):
        h_drive_Delta_1 += p.Delta_1*a[k].dag()*a[k]
        h_drive_epsilon_1 += p.epsilon_1*a[k] + np.conj(p.epsilon_1)*a[k].dag()


    return h_disp + h_drive_Delta_1 + h_drive_epsilon_1