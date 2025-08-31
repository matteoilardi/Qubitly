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

## Example: Bell state preparation

```python
import Qubitly as qbl
import jax

# Define the circuit
bell_circuit = qbl.QuantumCircuit([
    qbl.Hadamard(0),
    qbl.CNOT(control=0, target=1)
])

# Jit it, if you want
bell_circuit_jit = jax.jit(bell_circuit)

# Define input state
_00 = qbl.WaveFunction.from_string('00')

# Perform calculation
_bell, _ = bell_circuit_jit(qbl._NO_RANDOMNESS, _00)

print(_bell) # Output: 
# WaveFunction: [0.70710677+0.j 0.        +0.j 0.        +0.j 0.70710677+0.j]
```

