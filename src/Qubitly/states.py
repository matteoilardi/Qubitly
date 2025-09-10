from __future__ import annotations # required for method from_superposition()
import jax
import jax.numpy as jnp
import jax.random as jrand
import numpy as np
from typing import Sequence, Optional, Union

class WaveFunction():
    def __init__(self, amplitudes: jnp.ndarray | Sequence, n_qubits: Optional[int] = None):
        if not isinstance(amplitudes, jnp.ndarray):
            amplitudes = jnp.array(amplitudes, dtype=jnp.complex64)  

        if n_qubits is not None:
            if amplitudes.shape != (2**n_qubits,):
                raise ValueError("List of amplitudes doesn't match local Hilbert space dimension (qubits)")
            self.n_qubits = n_qubits
        else:
            m = amplitudes.shape[0]
            if (m & (m - 1)) != 0: # Check if m is a power of two
                raise ValueError("List of amplitudes is incompatible with local Hilbert space dimension 2 (qubits)")
            self.n_qubits = m.bit_length() - 1

        self.amplitudes = amplitudes
        #self.normalize()

    @classmethod
    def from_string(cls, bit_str: str):
        n_qubits = len(bit_str)
        base_state = int(bit_str, 2)
        amplitudes = jnp.zeros(2**n_qubits, dtype=jnp.complex64).at[base_state].set(1.0)
        return cls(amplitudes, n_qubits)

    @classmethod
    def from_superposition(cls, wfs: list[WaveFunction], weights: Optional[list[complex]] = None) -> WaveFunction:
        if weights is not None and len(wfs) != len(weights):
            raise ValueError("List of WaveFunction's and of weights provided have different lengths")

        if weights is None:
            weights = jnp.ones(len(wfs), dtype=jnp.complex64)
        else:
            weights = jnp.array(weights, dtype=jnp.complex64)

        wfs_amps = jnp.stack([wf.amplitudes for wf in wfs])
        weights = weights[:, None]  # broadcast to match shape
        result_amps = jnp.sum(wfs_amps * weights, axis=0)
        result_amps = normalize_array(result_amps)

        return WaveFunction(result_amps)

    def normalize(self):
        # A method that normalizes a vector must be defined separately in order to jit it
        self.amplitudes = normalize_array(self.amplitudes)


    @property
    def norm(self):
        return jnp.linalg.norm(self.amplitudes)

    @property
    def dim(self):
        return 2**self.n_qubits

    def overlap(self, other: WaveFunction) -> jnp.complex64:
        return jnp.dot(self.amplitudes, other.amplitudes)

    def __str__(self):
        return f"WaveFunction: {self.amplitudes}"

    def __eq__(self, other):
        if not isinstance(other, WaveFunction):
            return NotImplemented
        return jnp.allclose(self.amplitudes, other.amplitudes) # TODO Consider checking also n_qubits

    # NOTE flatten/unflatten methods are required to register WaveFunction as a pytree node. This in turn is necessary to jit a function (e. g. a QuantumCircuit) that accepts a WaveFunction as an argument.
    def tree_flatten(self):
        children = (self.amplitudes,)
        aux_data = (self.n_qubits,)
        
        return children, aux_data

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        (amplitudes,) = children
        (n_qubits,) = aux_data

        # Bypassing __init__ for two reasons: avoid unnecessary validations; for some reason jax raises when trying to access amplitudes.shape inside __init__ (IndexError: tuple index out of range)
        wf = cls.__new__(cls)
        wf.amplitudes = amplitudes
        wf.n_qubits = n_qubits
        
        return wf
        

# Explicit pytree node registration
jax.tree_util.register_pytree_node(
    WaveFunction,
    WaveFunction.tree_flatten,
    WaveFunction.tree_unflatten
)


@jax.jit
def couple(wf1: WaveFunction, wf2: WaveFunction) -> WaveFunction:
    n_qubits_result = wf1.n_qubits + wf2.n_qubits

    def result_amplitude(p: int):
        p1 = (p >> wf2.n_qubits)
        p2 = (p & (2**wf2.n_qubits - 1))
        return wf1.amplitudes[p1] * wf2.amplitudes[p2]
    amps_result = jax.vmap(result_amplitude)(jnp.arange(2**n_qubits_result))
    
    return WaveFunction(amps_result)
    

# Dummy key to pass to circuits that don't involve random number generation
# DUMMY_KEY = jrand.key(0)
# However, probably due to key tracing overhead, some QuantumCircuits that require no randomness get a jitted version that's even slower than the non jitted one.
# It is way better to pass _NO_RANDOMNESS (object() cannot be passed because it's not traceable)

# Dummy traceable object to pass to circuits that don't involve random number generation
_NO_RANDOMNESS = jnp.array([-1], dtype=jnp.int32)


class CompBasisMeasurement:
    def __init__(self, user_var: str, qubit_to_measure: int):
        # TODO When simultaneous (i. e. in the same CircuitStep) measurement of multiple qubits will be supported, perform consintency checks (e. g. number of variables matches number of measured qubits)
        # Perhaps these checks should be done inside __call__() for jit-compatibility (see implementation of ClassicalCtrlOperator)

        self.user_var = user_var
        self.qubit_to_measure = qubit_to_measure


    def __call__(self, key, wf: WaveFunction, user_vars: dict) -> tuple[jnp.ndarray, WaveFunction, dict]:
        key, subkey = jrand.split(key)

        # TODO When implementing simultaneous measurement of multiple qubits use _measure_all_computational_basis if set(self.qubits_to_measure) == set(np.arange(wf.dim))
        key, outcome, measured_amplitudes = _measure_computational_basis(subkey, wf.amplitudes, jnp.array([self.qubit_to_measure]))

        user_vars[self.user_var] = outcome
        return key, WaveFunction(measured_amplitudes), user_vars

            

# ===== HELPER FUNCTIONS =====

@jax.jit
def normalize_array(arr: jnp.ndarray):
    norm = jnp.sqrt(jnp.sum(jnp.abs(arr)**2))
    return arr / norm

@jax.jit
def _measure_all_computational_basis(key, amplitudes: jnp.ndarray):
    key, subkey = jrand.split(key)
    r = jrand.uniform(subkey)
    
    probabilities = jnp.abs(amplitudes) ** 2
    cumulative = jnp.cumsum(probabilities)
    sampled_basis_state = jnp.searchsorted(cumulative, r) # Find the index of the smallest element of the array among those that are grater than r
    
    measured_amplitudes = jnp.zeros_like(amplitudes, dtype=jnp.complex64)
    measured_amplitudes = measured_amplitudes.at[sampled_basis_state].set(1.0)

    return key, sampled_basis_state, measured_amplitudes

@jax.jit
def _measure_computational_basis(key, amplitudes: jnp.ndarray, qubits_to_measure: jnp.ndarray):
    
    # Build probability vector for the measured subspace
    def extract_amp_and_state_num(p: int):
        amp = amplitudes[p]
        
        def add_qubit_contribution_to_state_num(i, old_state_num):
            bit_value = (p >> qubits_to_measure[i]) & 1
            new_state_num = old_state_num | (bit_value << i)
            return new_state_num
        state_num = jax.lax.fori_loop(0, len(qubits_to_measure), add_qubit_contribution_to_state_num, 0)

        return amp, state_num

    amps, state_nums = jax.vmap(extract_amp_and_state_num)(jnp.arange(amplitudes.shape[0]))
    probabilities = jnp.zeros(2 ** qubits_to_measure.shape[0])
    probabilities = probabilities.at[state_nums].add(jnp.abs(amps)**2)

    # Sample state in measured subspace
    key, subkey = jrand.split(key)
    r = jrand.uniform(subkey)

    cumulative = jnp.cumsum(probabilities)
    sampled_state_num = jnp.searchsorted(cumulative, r)

    # Compute amplitutudes after the measurement by retaining only amplitudes that are tied to the measure outcome
    result_amplitudes = jnp.where(state_nums == sampled_state_num, amplitudes, 0.0)
    result_amplitudes = normalize_array(result_amplitudes)

    return key, sampled_state_num, result_amplitudes

