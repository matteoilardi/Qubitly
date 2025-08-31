import jax
import jax.numpy as jnp
from Qubitly.states import WaveFunction, CompBasisMeasurement
from Qubitly.gates import *
from Qubitly.circuits import QuantumCircuit

# Maybe CompBasisMeasurement should be defined inside gates

# QUANTUM TELEPORTATION

def prepare_for_teleportation(wf: WaveFunction):
    new_amps = jnp.pad(wf.amplitudes, (0, 6), constant_values=0)
    return WaveFunction(new_amps)

def extract_teleported_qubit(wf: WaveFunction, user_vars: dict):
    def is_third_qubit_amp(p: int):
        return jnp.logical_and((p & 1) == user_vars["m0"], ((p >> 1) & 1) == user_vars["m1"])
    mask = jax.vmap(is_third_qubit_amp)(jnp.arange(8))
    new_amps = wf.amplitudes[mask]
    return WaveFunction(new_amps)

QuantumTeleportation = QuantumCircuit((
    # Bell state creation on qubits 1, 2
    Hadamard(1),
    CNOT(control=1, target=2),
    
    # Bell measurement on qubits 0, 1
    CNOT(control=0, target=1),
    Hadamard(0),

    CompBasisMeasurement("m0", 0),
    CompBasisMeasurement("m1", 1),

    # Measurement-controlled gates
    ClassicalCZ(control="m0", target=2),
    ClassicalCX(control="m1", target=2),
))

# add kwarg for second arg of CompBasisMeasurement
# Extract relevant ampolitudes from result, based on m0 and m1
