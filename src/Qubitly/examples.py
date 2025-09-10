import jax
import jax.numpy as jnp
from Qubitly.states import WaveFunction, CompBasisMeasurement
from Qubitly.gates import *
from Qubitly.circuits import QuantumCircuit

# QUANTUM TELEPORTATION

# Couple input qubit to |00>
def prepare_for_teleportation(_single_qubit_input: WaveFunction) -> WaveFunction:
    new_amps = jnp.pad(_single_qubit_input.amplitudes, (0, 6), constant_values=0)
    return WaveFunction(new_amps)

# Project the 3-qubit wavefunction onto the subspace defined by measurement outcomes (m0, m1),
# and return the resulting 1-qubit state (the teleported qubit).
def extract_teleported_qubit(_raw_output: WaveFunction, measurement_vars: dict) -> WaveFunction:
    def is_third_qubit_amp(p: int):
        return jnp.logical_and(
            (p & 1) == measurement_vars["m0"], 
            ((p >> 1) & 1) == measurement_vars["m1"]
        )
    mask = jax.vmap(is_third_qubit_amp)(jnp.arange(8))
    new_amps = _raw_output.amplitudes[mask]
    return WaveFunction(new_amps)

QuantumTeleportation = QuantumCircuit(
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
)
