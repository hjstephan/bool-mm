"""
Pytest Konfiguration und Fixtures.
"""

import pytest
import numpy as np


@pytest.fixture
def identity_matrix():
    """Fixture: 3x3 Einheitsmatrix."""
    return np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])


@pytest.fixture
def zero_matrix():
    """Fixture: 3x3 Nullmatrix."""
    return np.array([[0, 0, 0], [0, 0, 0], [0, 0, 0]])


@pytest.fixture
def sample_matrices():
    """Fixture: Beispiel-Matrizen aus der Arbeit."""
    A = np.array([
        [1, 0, 1],
        [0, 1, 0],
        [1, 1, 0]
    ])
    B = np.array([
        [0, 1],
        [1, 0],
        [1, 1]
    ])
    expected = np.array([
        [1, 1],
        [1, 0],
        [1, 1]
    ])
    return A, B, expected


@pytest.fixture
def multiplier():
    """Fixture: BooleanMatrixMultiplier Instanz."""
    return BooleanMatrixMultiplier()