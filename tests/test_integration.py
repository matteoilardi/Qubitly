import pytest
from hypothesis import given, settings, strategies as st
import jax
import jax.numpy as jnp
import jax.random as jrand
from Qubitly.states import WaveFunction, normalize_array
from Qubitly.examples import QuantumTeleportation, prepare_for_teleportation, extract_teleported_qubit

jnp_complex64_strategy = st.builds(
    lambda re, im: jnp.complex64(re + 1j * im),
    re=st.floats(min_value=-10.0, max_value=10.0, width=32, allow_nan=False, allow_infinity=False),
    im=st.floats(min_value=-10.0, max_value=10.0, width=32, allow_nan=False, allow_infinity=False),
)

WaveFunction_1_qubit_strategy = (
    st.lists(jnp_complex64_strategy, min_size=2, max_size=2)
    .filter(lambda lst: sum(abs(z)**2 for z in lst) > 1e-12)
    .map(lambda lst: normalize_array(jnp.array(lst, dtype=jnp.complex64)))
    .map(lambda amplitudes: WaveFunction(amplitudes))
)

# QUANTUM TELEPORTATION

@settings(max_examples=20, deadline=None) # NOTE kwarg deadline=None is required for some reason
@given(WaveFunction_1_qubit_strategy)
def test_quantum_teleportation(_input):
    key = jrand.key(2)
    _prepared_input = prepare_for_teleportation(_input)
    
    _raw_output, user_vars = QuantumTeleportation(_prepared_input, key)
    _output = extract_teleported_qubit(_raw_output, user_vars)
    assert _output == pytest.approx(_input)
