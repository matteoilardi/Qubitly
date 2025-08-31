import pytest
import jax.numpy as jnp
import jax.random as jrand

from Qubitly.gates import _apply_matrix_to_single_site, _apply_matrix_to_two_sites
from Qubitly.states import _measure_computational_basis, normalize_array


# NOTE arr stands for array

# STATES

@pytest.fixture
def _00_arr() -> jnp.ndarray:
    return jnp.array([1, 0, 0, 0], dtype=jnp.complex64)

@pytest.fixture
def _01_arr() -> jnp.ndarray:
    return jnp.array([0, 1, 0, 0], dtype=jnp.complex64)

@pytest.fixture
def _11_arr() -> jnp.ndarray:
    return jnp.array([0, 0, 0, 1], dtype=jnp.complex64)

@pytest.fixture
def _0plus_arr() -> jnp.ndarray:
    return jnp.array([1, 1, 0, 0], dtype=jnp.complex64)

@pytest.fixture
def _plus0_arr() -> jnp.ndarray:
    return jnp.array([1, 0, 1, 0], dtype=jnp.complex64)

@pytest.fixture
def _plusplus_arr() -> jnp.ndarray:
    return jnp.ones(2**6, dtype=jnp.complex64)

@pytest.fixture
def _bell0_arr() -> jnp.ndarray:
    return jnp.array([1, 0, 0, 1], dtype=jnp.complex64)

@pytest.fixture
def _bell2_arr() -> jnp.ndarray:
    return jnp.array([0, 1, 1, 0], dtype=jnp.complex64)
    

# OPERATORS

@pytest.fixture
def sigma_x_arr() -> jnp.ndarray:
    return jnp.array([[0.0, 1.0], 
                     [1.0, 0.0]], dtype=jnp.complex64)

@pytest.fixture
def CNOT_arr() -> jnp.ndarray:
    return jnp.array([[1.0, 0.0, 0.0, 0.0], 
                      [0.0, 0.0, 0.0, 1.0], 
                      [0.0, 0.0, 1.0, 0.0], 
                      [0.0, 1.0, 0.0, 0.0]] , dtype=jnp.complex64)

    

# NOTE using unnormalized states here (can do this because states are transformed with unitary operators)
def test_apply_matrix_to_single_site(sigma_x_arr, _00_arr, _01_arr, _bell0_arr, _bell2_arr, _plusplus_arr):
    assert _apply_matrix_to_single_site(sigma_x_arr, 0, _00_arr) == pytest.approx(_01_arr)
    assert _apply_matrix_to_single_site(sigma_x_arr, 1, _bell0_arr) == pytest.approx(_bell2_arr)
    assert _apply_matrix_to_single_site(sigma_x_arr, 5, _plusplus_arr) == pytest.approx(_plusplus_arr)


def test_apply_matrix_to_two_sites(CNOT_arr, _0plus_arr, _plus0_arr, _bell0_arr):
    assert _apply_matrix_to_two_sites(CNOT_arr, [0, 1], _0plus_arr) == pytest.approx(_bell0_arr)
    assert _apply_matrix_to_two_sites(CNOT_arr, [1, 0], _0plus_arr) == pytest.approx(_0plus_arr)
    assert _apply_matrix_to_two_sites(CNOT_arr, [0, 1], _plus0_arr) == pytest.approx(_plus0_arr)
    assert _apply_matrix_to_two_sites(CNOT_arr, [1, 0], _plus0_arr) == pytest.approx(_bell0_arr)

    _6plus_arr = jnp.ones(2**6, dtype=jnp.complex64)
    assert _apply_matrix_to_two_sites(CNOT_arr, [4, 2], _6plus_arr) == pytest.approx(_6plus_arr)

def test_measure_computational_basis(_00_arr, _11_arr, _0plus_arr, _bell0_arr):
    # Integrate better random number generation in tests!!!
    key = jrand.key(2)

    # SIMPLEST CASE
    key, _, _measured_0 = _measure_computational_basis(key, _00_arr, jnp.array([0]))
    key, _, _measured_1 = _measure_computational_basis(key, _00_arr, jnp.array([1]))

    assert _measured_0 == pytest.approx(_00_arr)
    assert _measured_1 == pytest.approx(_00_arr)
    
    key, _, _measured_0 = _measure_computational_basis(key, _bell0_arr, jnp.array([0]))
    key, _, _measured_1 = _measure_computational_basis(key, _bell0_arr, jnp.array([1]))

    assert _measured_0 == pytest.approx(_00_arr) or _measured_0 == pytest.approx(_11_arr)
    assert _measured_1 == pytest.approx(_00_arr) or _measured_1 == pytest.approx(_11_arr)

    # MORE COMPLICATED CASE
    _state_arr = normalize_array(jnp.array([1, 1, 0, 1], dtype=jnp.complex64))
    key, _, _measured_1 = _measure_computational_basis(key, _state_arr, jnp.array([1]))

    # Remember that measurement results are normalized, so normalizing by hand |0plus> and |11> is necessary here
    assert _measured_1 == pytest.approx(normalize_array(_0plus_arr)) or _measured_1 == pytest.approx(normalize_array(_11_arr))


# Better to divide test in smaller units with specific purposes
# Tests for randomness?
