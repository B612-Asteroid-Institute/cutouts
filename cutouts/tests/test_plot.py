import matplotlib.pyplot as plt
import numpy as np
import pytest

from ..plot import add_velocity_vector


def test_add_velocity_vector_raises():
    # Test that add_velocity_vector raises the correct errors
    # when ra, dec, vra, vdec, and dt are nan

    fig, ax = plt.subplots(1, 1)
    with pytest.raises(ValueError):
        add_velocity_vector(ax, np.NaN, 0, 0, 0, 0)

    with pytest.raises(ValueError):
        add_velocity_vector(ax, 0, np.NaN, 0, 0, 0)

    with pytest.raises(ValueError):
        add_velocity_vector(ax, 0, 0, np.NaN, 0, 0)

    with pytest.raises(ValueError):
        add_velocity_vector(ax, 0, 0, 0, np.NaN, 0)

    with pytest.raises(ValueError):
        add_velocity_vector(ax, 0, 0, 0, 0, np.NaN)
