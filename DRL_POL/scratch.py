import pytest
import sys
import os
from unittest.mock import patch, MagicMock

# Add the directory containing your python files to the system path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from jets import (
    heav_func,
)

class TestHeavFunc:
    def test_heav_func_basic(self):
        position_z = 1.0
        delta_z = 0.5
        result = heav_func(position_z, delta_z)
        expected = "heav((z-0.750)*(1.250-z))"
        assert result == expected