"""This module implements the CircuitTensor class."""

import logging
import jax
import jax.numpy as jnp
import numpy as np

from qfactor import utils
from qfactor.gates import Gate

logger = logging.getLogger( "qfactor" )


class CircuitTensor():
    """A CircuitTensor tracks an entire circuit as a tensor."""

    def __init__ ( self, utry_target, gate_list ):
        """
        CircuitTensor Constructor

        Args:
            utry_target (np.ndarray): Unitary target matrix

            gate_list (list[Gate]): The circuit's gate list.
        """
        if(False):
            if not utils.is_unitary( utry_target ):
                raise TypeError( "Specified target matrix is not unitary." )

            if not isinstance( gate_list, list ):
                raise TypeError( "Gate list is not a list." )

            if not all( [ isinstance( gate, Gate ) for gate in gate_list ] ):
                raise TypeError( "Gate list contains non-gate objects." )


            if not all( [ utils.is_valid_location( gate.location, self.num_qubits )
                        for gate in gate_list ] ):
                raise ValueError( "Gate location mismatch with circuit tensor." )

        self.utry_target = utry_target
        self.num_qubits = utils.get_num_qubits( self.utry_target )

        self.gate_list = gate_list
        self.reinitialize()


    def reinitialize ( self ):
        """Reconstruct the circuit tensor."""
        logger.debug( "Reinitializing CircuitTensor" )

        self.tensor = self.utry_target.conj().T
        self.tensor = self.tensor.reshape( [2] * 2 * self.num_qubits )

        for gate in self.gate_list:
            self.apply_right( gate )

    def  _tree_flatten(self):
        children = (self.gate_list,)  # arrays / dynamic values
        aux_data = ( self.utry_target,)  # static values
        return (children, aux_data)


    @classmethod
    def _tree_unflatten(cls, aux_data, children):
        return cls(*aux_data, *children)
    
    @property
    def utry ( self ):
        """Calculates this circuit tensor's unitary representation."""
        num_elems = 2 ** self.num_qubits
        utry = self.tensor.reshape( ( num_elems, num_elems ) )
        # paulis = pauli_expansion( unitary_log_no_i( utry, tol = 1e-12 ) )
        # print( paulis[0] )
        return utry

    def apply_right ( self, gate, inverse = False ):
        """
        Apply the specified gate on the right of the circuit.

             .-----.   .------.
          n -|     |---|      |-
        n+1 -|     |---| gate |-
             .     .   '------'
             .     .
             .     .
       2n-1 -|     |------------
             '-----'
        
        Note that apply the gate on the right is equivalent to
        multiplying on the gate on the left of the tensor.
        This operation is performed using tensor contraction.

        Args:
            gate (Gate): The gate to apply.

            inverse (bool): If true, apply the inverse of gate.
        """
        
        gate_tensor = gate.get_tensor_format()
 
        if inverse:
            offset = 0
            gate_tensor = gate_tensor.conj()
        else:
            offset = gate.gate_size
            
        gate_tensor_indexs = [i for i in range(2*gate.gate_size)]
        
        circuit_tensor_indexs = [2*gate.gate_size + i  for i in range(2*self.num_qubits)]        
        output_tensor_index = [ 2*gate.gate_size + i  for i in range(2*self.num_qubits)]
        for i, loc in enumerate(gate.location):
            circuit_tensor_indexs[loc] = offset + i
            output_tensor_index[loc] = (gate.gate_size - offset) + i
        
        self.tensor  = jnp.einsum(gate_tensor, gate_tensor_indexs, self.tensor, circuit_tensor_indexs, output_tensor_index)

    def apply_left ( self, gate, inverse = False ):
        """
        Apply the specified gate on the left of the circuit.

             .------.   .-----.
          2 -|      |---|     |-
          3 -| gate |---|     |-
             '------'   .     .
                        .     .
                        .     .
       2n-1 ------------|     |-
                        '-----'
        
        Note that apply the gate on the left is equivalent to
        multiplying on the gate on the right of the tensor.
        This operation is performed using tensor contraction.

        Args:
            gate (Gate): The gate to apply.

            inverse (bool): If true, apply the inverse of gate.
        """

        gate_tensor = gate.get_tensor_format()
        if inverse:            
            offset = gate.gate_size
            gate_tensor = gate_tensor.conj()
        else:
            offset = 0
            
        gate_tensor_indexs = [i for i in range(2*gate.gate_size)]
        
        circuit_tensor_indexs = [2*gate.gate_size + i  for i in range(2*self.num_qubits)]        
        output_tensor_index = [ 2*gate.gate_size + i  for i in range(2*self.num_qubits)]
        for i, loc in enumerate(gate.location):
            circuit_tensor_indexs[self.num_qubits + loc] = offset + i
            output_tensor_index[self.num_qubits + loc] = (gate.gate_size - offset) + i

        self.tensor  = jnp.einsum(gate_tensor, gate_tensor_indexs, self.tensor, circuit_tensor_indexs, output_tensor_index)
        
    def calc_env_matrix ( self, location ):
        """
        Calculates the environmental matrix of the tensor with
        respect to the specified location.

        Args:
            location (iterable): Calculate the environment for this
                set of qubits.

        Returns:
            (np.ndarray): The environmental matrix.
        """

        if not isinstance( location, tuple ):
            raise TypeError("Location given is not a tuple")
            
        if not utils.is_valid_location( location, self.num_qubits ):
            raise ValueError( "Gate location mismatch with circuit tensor.", location, self.num_qubits )
        

        contraction_indexs = list(range(self.num_qubits))+list(range(self.num_qubits))
        for i, loc in enumerate(location):            
            contraction_indexs[loc+self.num_qubits] = self.num_qubits + i + 1

        contraction_indexs_str = "".join([chr(ord('a')+i) for i in contraction_indexs])

        env_tensor = jnp.einsum(contraction_indexs_str, self.tensor)
        env_mat = env_tensor.reshape((2**len(location), -1))

        return env_mat


from jax import tree_util
tree_util.register_pytree_node(CircuitTensor,
                                CircuitTensor._tree_flatten,
                                CircuitTensor._tree_unflatten)