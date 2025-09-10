# Qubitly

A minimal quantum simulator in pure Python, using JAX.

## Install

User installation:
```bash
pip install -e .
```

Add development tools:
```bash
pip install -r requirements.txt
```

## Test
After installing the package in your environment, run the test suite from the project root with:
```bash
pytest
```

## Example 1: Bell state preparation

```python
import Qubitly as qbl
import jax

# Define the circuit
bell_circuit = qbl.QuantumCircuit(
    qbl.Hadamard(0),
    qbl.CNOT(control=0, target=1)
)

# Jit it, if you want
bell_circuit_jit = jax.jit(bell_circuit)

# Define input state
_00 = qbl.WaveFunction.from_string('00')

# Perform calculation
_bell, _ = bell_circuit_jit(_00)

print(_bell)
```

Output:

``` 
WaveFunction: [0.70710677+0.j 0.        +0.j 0.        +0.j 0.70710677+0.j]
```

## Example 2: Quantum Teleportation

```python
import Qubitly as qbl
import jax
import jax.numpy as jnp
import jax.random as jrand

QuantumTeleportation = qbl.QuantumCircuit(
    # Bell state creation on qubits 1, 2
    qbl.Hadamard(1),
    qbl.CNOT(control=1, target=2),
    
    # Bell measurement on qubits 0, 1
    qbl.CNOT(control=0, target=1),
    qbl.Hadamard(0),

    qbl.CompBasisMeasurement("m0", 0),
    qbl.CompBasisMeasurement("m1", 1),

    # Measurement-controlled gates
    qbl.ClassicalCZ(control="m0", target=2),
    qbl.ClassicalCX(control="m1", target=2),
)

_0 = qbl.WaveFunction.from_string('0')
_1 = qbl.WaveFunction.from_string('1')
_00 = qbl.WaveFunction.from_string('00')

_input = qbl.WaveFunction.from_superposition([_0, _1], [1., -1.]) # Example input
print(_input)

_input = qbl.couple(_00, _input) # Prepare for teleportation
_output, measurement_outcomes = QuantumTeleportation(_input, jrand.key(8))

print(_output)
print(measurement_outcomes)
```

Output:

```
WaveFunction: [ 0.70710677+0.j -0.70710677+0.j]
WaveFunction: [ 0.        +0.j  0.70710677+0.j  0.        +0.j  0.        +0.j
  0.        +0.j -0.70710677+0.j  0.        +0.j  0.        +0.j]
{'m0': Array(1, dtype=int32), 'm1': Array(0, dtype=int32)}
``` 

## Example 3: Train a Parametrized Quantum Circuit to Prepare Bell States
```python
import Qubitly as qbl
import itertools as it
import functools as ft
import jax
import jax.numpy as jnp
import numpy as np


# Two-qubit variational quantum circuit with 3 repeated layers.
# Each layer consists of:
#   - Single-qubit rotations (Rx, Ry, Rz) on both qubits (0 and 1), each parametrized
#     by a unique angle "theta_{i}_{axis}{qubit}" where i = layer index;
#   - Two entangling CNOT gates to couple the qubits.
qc = qbl.QuantumCircuit(
    *[qbl.CircuitLayer(
        qbl.ParametrizedRx("theta_"+str(i)+"_x0", 0),
        qbl.ParametrizedRy("theta_"+str(i)+"_y0", 0),
        qbl.ParametrizedRz("theta_"+str(i)+"_z0", 0),
        qbl.ParametrizedRx("theta_"+str(i)+"_x1", 1),
        qbl.ParametrizedRy("theta_"+str(i)+"_y1", 1),
        qbl.ParametrizedRz("theta_"+str(i)+"_z1", 1),
        qbl.CNOT(control=0, target=1),
        qbl.CNOT(control=1, target=0),
    ) for i in range(3)],
)

# Inputs: computational basis states of two qubits
_00 = qbl.WaveFunction.from_string('00')
_01 = qbl.WaveFunction.from_string('01')
_10 = qbl.WaveFunction.from_string('10')
_11 = qbl.WaveFunction.from_string('11')
_inputs = [_00, _01, _10, _11]

# Targets: Bell states
_bell0 = qbl.WaveFunction.from_superposition([_00, _11])
_bell1 = qbl.WaveFunction.from_superposition([_00, _11], [1., -1.])
_bell2 = qbl.WaveFunction.from_superposition([_01, _10])
_bell3 = qbl.WaveFunction.from_superposition([_01, _10], [-1., 1.])
_targets = [_bell0, _bell1, _bell2, _bell3]


# Fidelity of circuit output and target, averaged over the four basis states
def average_fidelity(var_dict: dict) -> float:
    _outputs = [qc(_input, qbl._NO_RANDOMNESS, **var_dict).wf for _input in _inputs]
    fidelities = jnp.array([jnp.abs(_output.overlap(_target))**2 
                            for _output, _target in zip(_outputs, _targets)])
    
    return jnp.mean(fidelities)

# Plain vanilla gradient ascent
@jax.jit # Jit the core function!
def train_step(var_dict: dict, learning_rate: float) -> tuple[dict, dict]:
    avg_fidelity, delta_dict = jax.value_and_grad(average_fidelity)(var_dict)
    var_dict = jax.tree_util.tree_map(lambda value, delta: value + learning_rate*delta, var_dict, delta_dict)
    metrics = {"average fidelity": avg_fidelity} # Note that avg_fidelity actually refers to the previous step
    return var_dict, metrics

# Hyperparameters
TRAINING_STEPS = 16
LEARNING_RATE = 0.8

def train_loop(initial_vars) -> dict:
    var_dict = initial_vars
    for step in range(TRAINING_STEPS):
        var_dict, metrics = train_step(var_dict, LEARNING_RATE)
        metrics = {"before step": step} | metrics 
        print(*it.starmap("{}: {}".format, metrics.items()), sep=", \t")
    return var_dict

# Initialize parameters with random values
initial_vars = {
    f"theta_{i}_{axis}{j}": np.random.rand()*np.pi
    for axis in ('x', 'y', 'z')
    for i in range(3)
    for j in range(2)
}
final_vars = train_loop(initial_vars)
print("Final average fidelity: ", average_fidelity(final_vars))
```

Typical output:

```
before step: 0, 	average fidelity: 0.17282041907310486
before step: 1, 	average fidelity: 0.23148536682128906
before step: 2, 	average fidelity: 0.2829485833644867
before step: 3, 	average fidelity: 0.3262091875076294
before step: 4, 	average fidelity: 0.36809390783309937
before step: 5, 	average fidelity: 0.41574573516845703
before step: 6, 	average fidelity: 0.47418922185897827
before step: 7, 	average fidelity: 0.5457116365432739
before step: 8, 	average fidelity: 0.6289483308792114
before step: 9, 	average fidelity: 0.7185207605361938
before step: 10, 	average fidelity: 0.8052879571914673
before step: 11, 	average fidelity: 0.8781540393829346
before step: 12, 	average fidelity: 0.9297406077384949
before step: 13, 	average fidelity: 0.96100914478302
before step: 14, 	average fidelity: 0.9782591462135315
before step: 15, 	average fidelity: 0.9875431656837463
Final average fidelity:  0.99263567
```
