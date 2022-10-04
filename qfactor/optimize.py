"""This module implements the main optimize function."""

import logging

import numpy as np
import jax
import time
import jax.numpy as jnp

from qfactor import utils
from qfactor.gates import Gate, gate
from qfactor.tensors import CircuitTensor


logger = logging.getLogger( "qfactor" )


def optimize ( target, gates_location, gates_untrys, diff_tol_a = 1e-12, diff_tol_r = 1e-6,
               dist_tol = 1e-10, max_iters = 100000, min_iters = 1000,
               slowdown_factor = 0.0, check_inputs=False ):
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
    if check_inputs:
        if not isinstance( gates_location, list ):
            raise TypeError( "The gates_location argument is not a list." )

        if not all( [ isinstance( g, tuple ) for g in gates_location ] ):
            raise TypeError( "The gates_location argument is not a list of tuples." )

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


    c1s = [0]*len(gates_untrys)
    c2s = [1]*len(gates_untrys)
    it = 0

    iter_vmap = jax.vmap(single_iteration, in_axes=(None, None,  None, 0))
    
    while True:

        # Termination conditions
        if it > min_iters:

            if all((jnp.abs(c1 - c2) <= diff_tol_a + diff_tol_r * jnp.abs( c1 ) for c1, c2 in zip(c1s, c2s))):
                # diff = jnp.abs(c1 - c2)
                logger.info( f"Terminated: |c1 - c2| "
                              " <= diff_tol_a + diff_tol_r * |c1|. for all instances" )
                break;

            if it > max_iters:
                logger.info( "Terminated: iteration limit reached." )
                break;

        it += 1
            
        c2s = c1s
        tic = time.perf_counter()
        c1s, gates_untrys = iter_vmap(slowdown_factor, target, gates_location, gates_untrys)
        c1s = c1s.block_until_ready()
        toc = time.perf_counter()

        print(f"iteration took {toc-tic} seconeds")
        
        if any([c1 <= dist_tol for c1 in c1s]):
            logger.info( f"Terminated: c1 = {c1s} <= dist_tol." )
            return gates_untrys

        if it % 5 == 0:
            logger.info( f"iteration: {it}, cost: {c1s}" )

    return gates_untrys

def single_iteration(slowdown_factor, target, gates_location, gates_untrys):

    circuit = [Gate(utry, loc) for utry,loc in zip(gates_untrys, gates_location)]
    ct = CircuitTensor(target, circuit)
    
    for _ in range(40):
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
    return c1, jnp.array([g.utry for g in ct.gate_list])


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

