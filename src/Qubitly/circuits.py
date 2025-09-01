import jax
import jax.numpy as jnp
from collections import Counter
import functools
from typing import Protocol, NamedTuple
from Qubitly.states import WaveFunction, _NO_RANDOMNESS
from Qubitly.gates import Operator

class CircuitError(Exception):
    """Custom exception for errors in quantum circuit construction or execution."""
    
    def __init__(self, message: str, *, qubits=None):
        self.message = message
        self.qubits = qubits
        super().__init__(self._format_message())

    def _format_message(self):
        context = []
        if self.qubits is not None:
            context.append(f"Qubits: {self.qubits}")
        context_info = " | ".join(context)
        return f"{self.message}" + (f" ({context_info})" if context_info else "")

    def __str__(self):
        return self._format_message()


class CircuitLayer():
    '''
    This class is intended to provide a user-friendly interface for circuit definition.
    From the point of view of the actual calculations performed under the hood, it makes no difference whether or how you organize your gates into layers.
    In fact, the codebase does not include any form of gate-merging optimization as of now.
    '''

    def __init__(self, *gates: Operator):
        # Check if there are two or more gates acting on the same qubit
        sites = [site for gate in gates for site in gate.sites]
        if len(sites) != len(set(sites)):
            counter = Counter(sites)
            duplicates = [item for item, count in counter.items() if count > 1]
            raise CircuitError("Overlapping gates on the same site", qubits=duplicates)

        self.sites = sites
        self.gates = gates


    def __call__(self, key, wf: WaveFunction, user_vars: dict) -> tuple[jnp.ndarray, WaveFunction, dict]:
        return functools.reduce(lambda state, gate: gate(*state), 
                                self.gates, (key, wf, user_vars))


class CircuitStep(Protocol):
    '''
    Basic interface of a step inside a QuantumCircuit: a CircuitStep can be an Operator or a CircuitLayer 
    '''
    def __call__(self, key, wf: WaveFunction, user_vars: dict) -> tuple[jnp.ndarray, WaveFunction, dict]: ...


class CircuitResult(NamedTuple):
    wf: WaveFunction
    classsical_vars: dict


class QuantumCircuit:
    def __init__(self, *steps: CircuitStep):
        self.steps = steps

        # Define a jitted version of __call__ and store in a data member
        # Unfortunately, it seems impossible to jit __call__ from within the class (can't transform __call__ inside __init__): one should jit instances instead
        @functools.partial(jax.jit)
        def jit_call_wrapper(key, wf: WaveFunction, user_vars: dict):
            final_result = functools.reduce(lambda state, step: step(*state), 
                                    self.steps, (key, wf, user_vars))
            return CircuitResult(*final_result[1:])
        self.jit_call_wrapper = jit_call_wrapper
        

    def __call__(self, wf: WaveFunction, key = _NO_RANDOMNESS, **user_vars) -> CircuitResult:
        final_result = functools.reduce(lambda state, step: step(*state), 
                                self.steps, (key, wf, user_vars))
        return CircuitResult(*final_result[1:])

    def jit_call(self, wf: WaveFunction, key = _NO_RANDOMNESS, **user_vars) -> CircuitResult:
        return self.jit_call_wrapper(key, wf, user_vars)
        
