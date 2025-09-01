from __future__ import annotations # required for methods of a class to return a instance of the same class (e. g. Operator.apply())
import jax
import jax.numpy as jnp
import numpy as np
from typing import Optional, Union, Callable, TypeAlias
from Qubitly.states import WaveFunction

class SimpleOperator:
    '''
    Operator acting on one or more qubits
    Nothing more than a wrapper for a complex valued matrix with shape (2ⁿ, 2ⁿ)
    '''
    def __init__(self, matrix: jnp.ndarray, n_qubits: Optional[int] = None):
        if n_qubits is not None:
            if matrix.shape != (2**n_qubits, 2**n_qubits):
                raise ValueError("Matrix size doesn't match specified number of qubits")
            self.n_qubits = n_qubits
        else:
            (m1, m2) = matrix.shape
            if (m1 & (m1 - 1)) != 0: # Check if m1 is a power of two
                raise ValueError("Matrix size is incompatible with local Hilbert space dimension 2 (qubits)")
            if m1 != m2:
                raise ValueError("Matrix is not a square matrix")
            self.n_qubits = m1.bit_length() - 1
        
        self.matrix = matrix
        
    def apply(self, other: WaveFunction | SimpleOperator) -> WaveFunction | SimpleOperator:
        if other.n_qubits != self.n_qubits:
            raise ValueError("Operator size doesn't match the size of the object to which it's applied")

        if isinstance(other, WaveFunction):
            amplitudes = self.matrix @ wf.amplitudes
            return WaveFunction(amplitudes=amplitudes, n_qubits=self.n_qubits)
        elif isinstance(other, SimpleOperator):
             matrix = self.matrix @ other.matrix
             return SimpleOperator(matrix=matrix, n_qubits=self.n_qubits)
        else:
            return NotImplemented
            
    def __mul__(self, other: WaveFunction | SimpleOperator) -> WaveFunction | SimpleOperator:
        return self.apply(other)
            
    @property
    def is_hermitian(self) -> bool:
        return is_hermitian(self.matrix)

    @property
    def is_unitary(self) -> bool:
        return is_unitary(self.matrix)

class SimpleHadamard(SimpleOperator):
    def __init__(self):
        hadamard = (1 / jnp.sqrt(2)) * jnp.array([[1,  1], 
                                                  [1, -1]], dtype=jnp.complex64)
        super().__init__(matrix=hadamard, n_qubits=1)

class SimpleSigmaX(SimpleOperator):
    def __init__(self):
        sigmaX = jnp.array([[0.0, 1.0], 
                            [1.0, 0.0]], dtype=jnp.complex64)
        super().__init__(matrix=sigmaX, n_qubits=1)

class SimpleSigmaY(SimpleOperator):
    def __init__(self):
        sigmaY = jnp.array([[0.0, -1.0j], 
                            [1.0j, 0.0]], dtype=jnp.complex64)
        super().__init__(matrix=sigmaY, n_qubits=1)

class SimpleSigmaZ(SimpleOperator):
    def __init__(self):
        sigmaZ = jnp.array([[1.0, 0.0], 
                            [0.0, -1.0]], dtype=jnp.complex64)
        super().__init__(matrix=sigmaZ, n_qubits=1)

class SimplePhaseGate(SimpleOperator):
    def __init__(self, phase: complex):
        self.phase = phase
        matrix = jnp.array([[1.0, 0.0], 
                            [0.0, jnp.exp(1.0j * phase)]], dtype=jnp.complex64)
        super().__init__(matrix=matrix, n_qubits=1)

class SimpleSGate(SimpleOperator):
    def __init__(self):
        S = jnp.array([[1.0, 0.0], 
                       [0.0, 1.0j]], dtype=jnp.complex64)
        super().__init__(matrix=S, n_qubits=1)

class SimpleCNOT(SimpleOperator):
    def __init__(self):
        CNOT_matrix = jnp.array([[1.0, 0.0, 0.0, 0.0], 
                          [0.0, 0.0, 0.0, 1.0], 
                          [0.0, 0.0, 1.0, 0.0], 
                          [0.0, 1.0, 0.0, 0.0]] , dtype=jnp.complex64)
        super().__init__(matrix=CNOT_matrix, n_qubits=2)

class SimpleCZ(SimpleOperator):
    def __init__(self):
        CZ_matrix = jnp.array([[1.0, 0.0, 0.0, 0.0], 
                          [0.0, 1.0, 0.0, 0.0], 
                          [0.0, 0.0, 1.0, 0.0], 
                          [0.0, 0.0, 0.0, -1.0]] , dtype=jnp.complex64)
        super().__init__(matrix=CZ_matrix, n_qubits=2)



class Operator:
    def __init__(self, simple_op: SimpleOperator | list[SimpleOperator], sites: list[int]):
        if isinstance(simple_op, SimpleOperator):
            if len(sites) != simple_op.n_qubits:
                raise ValueError("Number of sites doesn't match SimpleOperator dimension")
        elif isinstance(simple_op, list):
            if not all(isinstance(op, SimpleOperator) for op in simple_op):
                raise TypeError("All elements must be SimpleOperator instances")
            if len(sites) != len(simple_op):
                raise ValueError("Number of sites doesn't match number of SimpleOperator instances provided")
            elif any(op.n_qubits != 1 for op in simple_op):
                raise ValueError("At least one of the SimpleOperators provided is not a single-qubit operator")

        # The case in which the user provides two single-qubit operators must be handled here in the initializer. In fact, the list->SimpleOperator type change would cause problems in the jitted __call__ method.
        if isinstance(simple_op, list):
            assert len(simple_op) == 2, f"Operator object currently supports one and two-qubit gates: received a list of SimpleOperators of length {len(simple_op)}"
            simple_op = SimpleOperator(np.kron(simple_op[1].matrix, simple_op[0].matrix)) # Remember that states are built taking tensor products of qubits in decreasing order
            
        self.simple_op = simple_op
        self.sites = sites
        
      
    # def apply(self, other: WaveFunction | Operator) -> WaveFunction | Operator:
    def apply(self, other: WaveFunction) -> WaveFunction:
        if isinstance(other, WaveFunction):
            
            # jax.lax.cond is not required here because n_sites is treated as a static argument in every instance of Operator
            if self.n_sites == 1:
                amplitudes = _apply_matrix_to_single_site(self.simple_op.matrix, self.sites[0], other.amplitudes)
            elif self.n_sites == 2:
                amplitudes = _apply_matrix_to_two_sites(self.simple_op.matrix, self.sites, other.amplitudes)
            else:
                amplitudes = _apply_matrix_to_sites(self.simple_op.matrix, jnp.array(self.sites), other.amplitudes)

            return WaveFunction(amplitudes=amplitudes, n_qubits=other.n_qubits)
        
        elif isinstance(other, Operator):
            assert self.sites == other.sites # Just for now
            simple_op = SimpleOperator(self.simple_op.matrix @ other.simple_op.matrix, n_qubits=1)
            
            return Operator(simple_op=simple_op, sites=self.sites)
        
        else:
            return NotImplemented
            
        
    # def __mul__(self, other: WaveFunction | Operator) -> WaveFunction | Operator:
    def __mul__(self, other: WaveFunction) -> WaveFunction:
        return self.apply(other)

    def __call__(self, key, wf: WaveFunction, user_vars: dict) -> tuple[jnp.ndarray, WaveFunction, dict]:
        return key, self.apply(wf), user_vars

    @property
    def n_sites(self) -> int:
       return len(self.sites)


class Hadamard(Operator):
    def __init__(self, site: int):
        super().__init__(simple_op=SimpleHadamard(), sites=[site])

class SigmaX(Operator):
    def __init__(self, site: int):
        super().__init__(simple_op=SimpleSigmaX(), sites=[site])

class SigmaY(Operator):
    def __init__(self, site: int):
        super().__init__(simple_op=SimpleSigmaY(), sites=[site])

class SigmaZ(Operator):
    def __init__(self, site: int):
        super().__init__(simple_op=SimpleSigmaZ(), sites=[site])

class PhaseGate(Operator):
    def __init__(self, phase: jnp.complex64, site: int):
        simple_phase_gate = SimplePhaseGate(phase)
        super().__init__(simple_op=simple_phase_gate, sites=[site])

class SGate(Operator):
    def __init__(self, site: int):
        super().__init__(simple_op=SimpleSGate(), sites=[site])



class CtrlGateMixin:
    def resolve_ctrl_and_target(self, **kwargs) -> list[int]:
        if 'sites' in kwargs:
            if 'control' in kwargs or 'target' in kwargs:
                raise ValueError("Specify either `sites` or `control`/`target`, not both.")
            return kwargs['sites']
        elif 'control' in kwargs and 'target' in kwargs:
            return [kwargs['control'], kwargs['target']]
        else:
            raise ValueError("Provide either `sites=[i, j]` or both `control` and `target`.")

class ClassicalCtrlGateMixin(CtrlGateMixin):
    def resolve_ctrl_and_target(self, **kwargs) -> list[int]:
        ctrl_and_target = super().resolve_ctrl_and_target(**kwargs)
        if not isinstance(ctrl_and_target[0], str):
            raise TypeError("Control gate of a classical controlled gate must be a user-variable name (str)")
        if not isinstance(ctrl_and_target[1], int):
            raise TypeError("Target gate of a classical controlled gate must be a qubit number (int)")
            
        return ctrl_and_target


class CNOT(CtrlGateMixin, Operator):
    def __init__(self, **kwargs):
        sites = self.resolve_ctrl_and_target(**kwargs)
        super().__init__(simple_op=SimpleCNOT(), sites=sites)

class CZ(CtrlGateMixin, Operator):
    def __init__(self, **kwargs):
        sites = self.resolve_ctrl_and_target(**kwargs)
        super().__init__(simple_op=SimpleCZ(), sites=sites)




# Design choice: Classical controlled operators do not inherit from Operator, but contain a instance of Operator
class ClassicalCtrlOperator(ClassicalCtrlGateMixin):
    def __init__(self, simple_op: SimpleOperator, **kwargs):
        control_var, site = self.resolve_ctrl_and_target(**kwargs)
        self.control_var = control_var
        self.site = site
        self.operator = Operator(simple_op, [site])
        
    def __call__(self, key, wf: WaveFunction, user_vars: dict) -> tuple[jnp.ndarray, WaveFunction, dict]:
        # One might worry that this non-jax control flow might fail... actually it doesn't: it is evaluated once for all at during tracing. If the keys of the dict were to change, this would trigger recompilation. (In our case this does not happen, since variable names are chosen in the definition of the circuit and never changed).
        # On the other hand, jax flow controls cannot be used to raise exceptions, since every path is explored while tracing, including the one that raises the exception!
        if self.control_var not in user_vars:
            raise KeyError(f"User-variable {self.control_var} not defined")

        control = user_vars[self.control_var]

        # An excpetion should be raised here if control is not 0 or 1; however, as pointed out before, exception raiseing in jitted functions can depend only on shapes or static arguments.
        # We could use checkify.check from jax.experimental (which requires decorating the function that contains it with @checkify.checkify), but we should keep track of an error variable and print it at the end of the calculation. 

        branches = [
            lambda wf: wf,
            lambda wf: self.operator.apply(wf),
        ]
        wf = jax.lax.switch(control, branches, wf)

        return key, wf, user_vars


class ClassicalCX(ClassicalCtrlOperator):
    def __init__(self, **kwargs):
        super().__init__(SimpleSigmaX(), **kwargs)

class ClassicalCZ(ClassicalCtrlOperator):
    def __init__(self, **kwargs):
        super().__init__(SimpleSigmaZ(), **kwargs)



SimpleOperatorFn: TypeAlias = Callable[[complex], SimpleOperator]

# Design choice: ParametrizedOperator does not inherit from operator, but it creates an instance of Operator at ciecuit-runtime on the fly
class ParametrizedOperator:
    def __init__(self, simple_op_fn: SimpleOperatorFn, param_name: str, sites: int | list[int]):
        # NOTE that input validation will be done at circuit-runtime, i. e. inside __call__(), i. e. inside the constructor of Operator
        self.simple_op_fn = simple_op_fn
        self.param_name = param_name

        if not isinstance(sites, list): # TODO Perhaps the same logic can be applied somewhere else
            sites = [sites]
        self.sites = sites

    def __call__(self, key, wf: WaveFunction, user_vars: dict) -> tuple[jnp.ndarray, WaveFunction, dict]:
        if self.param_name not in user_vars:
            raise KeyError(f"Parameter {self.param_name} not defined")
        
        simple_op = self.simple_op_fn(user_vars[self.param_name])
        op = Operator(simple_op, self.sites)
        return op(key, wf, user_vars)
        
        
class ParametrizedPhaseGate(ParametrizedOperator):
    def __init__(self, param_name: str, sites: list[int]):
        super().__init__(lambda phase: SimplePhaseGate(phase), param_name, sites)
        



    
    


# HELPER FUNCTIONS
@jax.jit
def is_hermitian(matrix: jnp.ndarray) -> bool:
    return jnp.allclose(matrix, jnp.conj(matrix.T))

@jax.jit
def is_unitary(matrix: jnp.ndarray) -> bool:
    return jnp.allclose(matrix @ jnp.conj(matrix.T), jnp.eye(matrix.shape[0], dtype=jnp.complex64))

# (Currently using implementation #4)
@jax.jit
def _apply_matrix_to_single_site(matrix: jnp.ndarray, site: int, vector: jnp.ndarray):

    def calculate_basis_state_contribution(p: int):
        site_bit = (p >> site) & 1

        mask = ~(1 << site)
        masked_p = p & mask
        
        def handle_matrix_element(i: int):
            idx = masked_p | (i << site)
            amp = matrix[i, site_bit] * vector[p]
            return idx, amp

        idxs_p, amps_p = jax.vmap(handle_matrix_element)(jnp.arange(2))
        return idxs_p, amps_p

    idxs, amps = jax.vmap(calculate_basis_state_contribution)(jnp.arange(len(vector)))
    
    result = jnp.zeros_like(vector)
    result = result.at[idxs].add(amps)
    return result  


# (Currently using implementation #2)
@jax.jit
def _apply_matrix_to_two_sites(matrix: jnp.ndarray, sites: list[int], vector: jnp.ndarray):

    def calculate_basis_state_contribution(p: int):
        least_important_bit = (p >> sites[0]) & 1
        most_important_bit = (p >> sites[1]) & 1
        sites_number = (most_important_bit << 1) | least_important_bit

        mask = ~( (1 << sites[0]) | (1 << sites[1]) )
        masked_p = p & mask
        
        def handle_matrix_element(i: int):
            idx = masked_p | ((i & 1) << sites[0]) | ((i >> 1) << sites[1])
            amp = matrix[i, sites_number] * vector[p]
            return idx, amp

        idxs_p, amps_p = jax.vmap(handle_matrix_element)(jnp.arange(4))
        return idxs_p, amps_p


    idxs, amps = jax.vmap(calculate_basis_state_contribution)(jnp.arange(len(vector)))
    
    result = jnp.zeros_like(vector)
    result = result.at[idxs].add(amps)
    return result

@jax.jit
def _apply_matrix_to_sites(matrix: jnp.ndarray, sites: jnp.array, vector: jnp.ndarray):
    # NOTE that argument "sites" cannot be a regular list. In fact, index "i" of fori_loops is a tracer and cannot be used to access elements of a list instead of a concrete integer: TracerIntegerConversionError is raised.

    def calculate_basis_state_contribution(p: int):
        affected_substate_number = jax.lax.fori_loop(0, len(sites), 
                                      lambda i, state: state | ((p >> sites[i]) & 1) << i, 
                                      0)

        mask = ~ jax.lax.fori_loop(0, len(sites), 
                                   lambda i, state: state | (1 << sites[i]), 
                                   0)
        masked_p = p & mask
        
        def handle_matrix_element(s: int):
            bitfiled_update = jax.lax.fori_loop(0, len(sites), 
                                                lambda i, state: state | ((s >> i) & 1) << sites[i], 
                                                0)
            idx = masked_p | bitfiled_update
            amp = matrix[s, affected_substate_number] * vector[p]
            return idx, amp

        idxs_p, amps_p = jax.vmap(handle_matrix_element)(jnp.arange(2**len(sites)))
        return idxs_p, amps_p


    idxs, amps = jax.vmap(calculate_basis_state_contribution)(jnp.arange(len(vector)))
    
    result = jnp.zeros_like(vector)
    result = result.at[idxs].add(amps)
    return result
