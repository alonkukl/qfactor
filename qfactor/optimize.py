"""This module implements the main optimize function."""

import logging

import numpy as np
import jax
import time
import jax.numpy as jnp

from qfactor import utils
from qfactor.gates import Gate
from qfactor.tensors import CircuitTensor


logger = logging.getLogger( "qfactor" )


def optimize ( circuit, target, diff_tol_a = 1e-12, diff_tol_r = 1e-6,
               dist_tol = 1e-10, max_iters = 100000, min_iters = 1000,
               slowdown_factor = 0.0 ):
    """
    Optimize distance between circuit and target unitary.

    Args:
        circuit (list[Gate]): The circuit to optimize.

        target (np.ndarray): The target unitary matrix.

        diff_tol_a (float): Terminate when the difference in distance
            between iterations is less than this threshold.
       
       diff_tol_r (float): Terminate when the relative difference in
            distance between iterations is iless than this threshold:
                |c1 - c2| <= diff_tol_a + diff_tol_r * abs( c1 )

        dist_tol (float): Terminate when the distance is less than
            this threshold.

        max_iters (int): Maximum number of iterations.

        min_iters (int): Minimum number of iterations.

        slowdown_factor (float): A positive number less than 1. 
            The larger this factor, the slower the optimization.

    Returns:
        (list[Gate]): The optimized circuit.
    """

    if not isinstance( circuit, list ):
        raise TypeError( "The circuit argument is not a list of gates." )

    if not all( [ isinstance( g, Gate ) for g in circuit ] ):
        raise TypeError( "The circuit argument is not a list of gates." )

    if not utils.is_unitary( target ):
        raise TypeError( "The target matrix is not unitary." )

    if not isinstance( diff_tol_a, float ) or diff_tol_a > 0.5:
        raise TypeError( "Invalid absolute difference threshold." )

    if not isinstance( diff_tol_r, float ) or diff_tol_r > 0.5:
        raise TypeError( "Invalid relative difference threshold." )

    if not isinstance( dist_tol, float ) or dist_tol > 0.5:
        raise TypeError( "Invalid distance threshold." )

    if not isinstance( max_iters, int ) or max_iters < 0:
        raise TypeError( "Invalid maximum number of iterations." )

    if not isinstance( min_iters, int ) or min_iters < 0:
        raise TypeError( "Invalid minimum number of iterations." )

    if slowdown_factor < 0 or slowdown_factor >= 1:
        raise TypeError( "Slowdown factor is a positive number less than 1." )

    ct = CircuitTensor( target, circuit )

    c1 = 0
    c2 = 1
    it = 0

    iter_jit = jax.jit(single_iteration)

    while True:

        # Termination conditions
        if it > min_iters:

            if jnp.abs(c1 - c2) <= diff_tol_a + diff_tol_r * jnp.abs( c1 ):
                diff = jnp.abs(c1 - c2)
                logger.info( f"Terminated: |c1 - c2| = {diff}"
                              " <= diff_tol_a + diff_tol_r * |c1|." )
                break;

            if it > max_iters:
                logger.info( "Terminated: iteration limit reached." )
                break;

        it += 1

        
        # c1, ct, circuit = iter_jit(circuit, slowdown_factor, ct, c1)
        # c1 = c1.block_until_ready()
        tic = time.perf_counter()
        # from right to left
        c1, ct = iter_jit(slowdown_factor, ct)
        c1 = c1.block_until_ready()

        toc = time.perf_counter()

        print(f"iteration took {toc-tic} seconeds")
        if c1 <= dist_tol:
            logger.info( f"Terminated: c1 = {c1} <= dist_tol." )
            return circuit

        if it % 100 == 0:
            logger.info( f"iteration: {it}, cost: {c1}" )

        if it % 40 == 0:
            ct.reinitialize()

    return ct.gate_list

def single_iteration(slowdown_factor, ct):
    circuit = ct.gate_list
    for k in range( len( circuit ) ):
        rk = len( circuit ) - 1 - k

            # Remove current gate from right of circuit tensor
        ct.apply_right( rk, inverse = True )

            # Update current gate
        if not circuit[rk].fixed:
            env = ct.calc_env_matrix( circuit[rk].location )
            circuit[rk].update( env, slowdown_factor )

            # Add updated gate to left of circuit tensor
        ct.apply_left( rk )

        # from left to right
    for k in range( len( circuit ) ):
            # Remove current gate from left of circuit tensor
        ct.apply_left( k, inverse = True )

            # Update current gate
        if not circuit[k].fixed:
            env = ct.calc_env_matrix( circuit[k].location )
            circuit[k].update( env, slowdown_factor )

            # Add updated gate to right of circuit tensor
        ct.apply_right(k)

    c1 = jnp.abs( jnp.trace( ct.utry ) )
    c1 = 1 - ( c1 / ( 2 ** ct.num_qubits ) )
    return c1, ct


def get_distance ( circuit, target ):
    """
    Returns the distance between the circuit and the unitary target.

    Args:
        circuit (list[Gate]): The circuit.

        target (np.ndarray): The unitary target.
    
    Returns:
        (float): The distance between the circuit and unitary target.
    """

    ct = CircuitTensor( target, circuit )
    num_qubits = utils.get_num_qubits( target )
    return 1 - ( jnp.abs( np.trace( ct.utry ) ) / ( 2 ** num_qubits ) )

